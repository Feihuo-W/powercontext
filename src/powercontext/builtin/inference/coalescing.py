# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Coalesce concurrent embedding work and cache successful vectors.

One wrapper instance owns the shared scheduling state for an event loop: texts
that are already queued or in flight are joined instead of requested again,
queued texts are grouped into bounded provider batches, and successful vectors
are kept in a bounded LRU cache. A caller keeps its own input order, its own
failure surface and its own usage attribution, while the underlying provider
work stays shared and independent of any single caller's cancellation.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field

from powercontext.builtin.artifacts.memory.models import EmbeddingProfile
from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(slots=True)
class _Batch:
    """One provider request shared by every slot it was grouped from."""

    texts: tuple[str, ...]
    usage: InferenceUsage | None = None


@dataclass(slots=True)
class _Slot:
    """One in-flight text whose vector several callers may await."""

    text: str
    future: asyncio.Future[EmbeddingVector] = field(default=None)  # type: ignore[assignment]
    batch: _Batch | None = None


class CoalescingEmbeddingModel:
    """Share in-flight embedding work and reuse successful vectors."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if cache_size < 0:
            raise ValueError("cache_size must not be negative")
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must not be negative")
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._pending: dict[str, _Slot] = {}
        self._queue: list[str] = []
        self._flusher: asyncio.Task[None] | None = None
        self._batches: set[asyncio.Task[None]] = set()

    @property
    def profile(self) -> EmbeddingProfile:
        return self._model.profile

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Embed one ordered batch, sharing work with concurrent callers."""

        ordered = tuple(texts)
        if not ordered:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        vectors: dict[str, EmbeddingVector] = {}
        slots: list[_Slot] = []
        scheduled = False

        for text in dict.fromkeys(ordered):
            cached = self._cache_get(text)
            if cached is not None:
                vectors[text] = cached
                continue
            slot = self._pending.get(text)
            if slot is None:
                slot = _Slot(text=text, future=asyncio.get_running_loop().create_future())
                slot.future.add_done_callback(_consume)
                self._pending[text] = slot
                self._queue.append(text)
                scheduled = True
            slots.append(slot)

        if scheduled:
            self._ensure_flusher()

        if slots:
            outcomes = await asyncio.gather(
                *(asyncio.shield(slot.future) for slot in slots),
                return_exceptions=True,
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
            for slot in slots:
                vectors[slot.text] = slot.future.result()

        return EmbeddingResult(
            vectors=tuple(vectors[text] for text in ordered),
            usage=_sum_usage(slot.batch for slot in slots),
        )

    def _ensure_flusher(self) -> None:
        if self._flusher is None or self._flusher.done():
            self._flusher = asyncio.get_running_loop().create_task(self._flush())

    async def _flush(self) -> None:
        try:
            if self._coalesce_seconds > 0:
                await asyncio.sleep(self._coalesce_seconds)
            while self._queue:
                batch = tuple(self._queue[: self._max_batch_size])
                del self._queue[: self._max_batch_size]
                task = asyncio.get_running_loop().create_task(self._run_batch(batch))
                self._batches.add(task)
                task.add_done_callback(self._batches.discard)
        finally:
            self._flusher = None

    async def _run_batch(self, texts: tuple[str, ...]) -> None:
        slots = [self._pending[text] for text in texts if text in self._pending]
        batch = _Batch(texts=texts)
        for slot in slots:
            slot.batch = batch
        try:
            result = await self._model.embed(texts)
        except BaseException as error:
            self._resolve(slots, error=error)
            raise
        if len(result.vectors) != len(texts):
            self._resolve(
                slots,
                error=ValueError(f"embedding provider returned {len(result.vectors)} vectors for {len(texts)} texts"),
            )
            return
        batch.usage = result.usage
        for slot, vector in zip(slots, result.vectors, strict=True):
            self._store(slot.text, vector)
        self._resolve(slots, vectors={slot.text: vector for slot, vector in zip(slots, result.vectors, strict=True)})

    def _resolve(
        self,
        slots: list[_Slot],
        *,
        error: BaseException | None = None,
        vectors: dict[str, EmbeddingVector] | None = None,
    ) -> None:
        for slot in slots:
            self._pending.pop(slot.text, None)
            if slot.future.done():
                continue
            if error is not None:
                slot.future.set_exception(error)
            elif vectors is not None:
                slot.future.set_result(vectors[slot.text])

    def _store(self, text: str, vector: EmbeddingVector) -> None:
        if self._cache_size == 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _cache_get(self, text: str) -> EmbeddingVector | None:
        if self._cache_size == 0:
            return None
        vector = self._cache.get(text)
        if vector is not None:
            self._cache.move_to_end(text)
        return vector


def _consume(future: asyncio.Future[EmbeddingVector]) -> None:
    """Mark a shared outcome as observed so abandoned waiters stay silent."""

    if not future.cancelled():
        future.exception()


def _sum_usage(batches: Iterable[_Batch | None]) -> InferenceUsage:
    seen: list[_Batch] = []
    for batch in batches:
        if batch is None or batch.usage is None:
            continue
        if not any(existing is batch for existing in seen):
            seen.append(batch)
    if not seen:
        return InferenceUsage(requests=0)
    tokens = [batch.usage.input_tokens for batch in seen if batch.usage is not None]
    output = [batch.usage.output_tokens for batch in seen if batch.usage is not None]
    return InferenceUsage(
        requests=sum(batch.usage.requests for batch in seen if batch.usage is not None),
        input_tokens=None if all(value is None for value in tokens) else sum(value or 0 for value in tokens),
        output_tokens=None if all(value is None for value in output) else sum(value or 0 for value in output),
    )


__all__ = ["CoalescingEmbeddingModel"]
