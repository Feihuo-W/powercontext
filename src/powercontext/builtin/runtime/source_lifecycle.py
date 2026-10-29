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

"""Source-driven lifecycle propagation for Memory and Topic Memory artifacts.

Dependency graphs are registered per scope. A Source event (withdrawal or
restoration) propagates through ``all``/``any`` policies to every descendant,
and the native Memory entry states (and Topic search visibility) follow the
propagated availability. Historical references keep resolving for audit
purposes even while a node is unavailable for generation.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryCitation
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.tables import LIFECYCLE_EVENTS_TABLE, LIFECYCLE_NODES_TABLE
from powercontext.builtin.runtime.models import GetMemoryEntryRequest
from powercontext.sources import SourceRef

__all__ = ["SourceLifecycleError", "SourceLifecycleEvent", "ScopedSourceLifecycleService"]

_MAX_NODE_ID_LENGTH = 128
_POLICIES = ("all", "any")


class SourceLifecycleError(ValueError):
    """A lifecycle graph, event, or resolution request violates its contract."""


class SourceLifecycleEvent(BaseModel):
    """One Source availability transition; replay-safe via per-source sequences."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=_MAX_NODE_ID_LENGTH)
    revision: int = Field(gt=0, strict=True)
    sequence: int = Field(gt=0, strict=True)
    available: bool = Field(strict=True)
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def require_visible_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must contain non-whitespace content")  # noqa: TRY003
        return value


class _Node(BaseModel):
    kind: Literal["source", "memory", "topic"]
    policy: str
    parents: tuple[str, ...]
    reference: str
    lifecycle_revision: int
    available: bool


def _reference_for_kind(kind: str, payload: str) -> SourceRef | MemoryCitation | ArtifactRef:
    model = {"source": SourceRef, "memory": MemoryCitation, "topic": ArtifactRef}[kind]
    return model.model_validate_json(payload)  # type: ignore[attr-defined]


def _availability(parents: tuple[str, ...], policy: str, states: dict[str, bool]) -> bool:
    support = (states[parent] for parent in parents)
    return all(support) if policy == "all" else any(support)


class ScopedSourceLifecycleService:
    """Register dependency graphs and propagate Source availability per scope."""

    def __init__(self, runtime: Any, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = scope_id

    @property
    def _database(self) -> AsyncDatabase:
        return self._runtime._provider.database

    async def _load_nodes(self, connection) -> dict[str, _Node]:
        table = LIFECYCLE_NODES_TABLE
        rows = (
            await connection.execute(
                select(
                    table.c.node_id,
                    table.c.kind,
                    table.c.policy,
                    table.c.parents,
                    table.c.reference,
                    table.c.lifecycle_revision,
                    table.c.available,
                ).where(table.c.scope_id == self.scope_id)
            )
        ).all()
        return {
            str(row[0]): _Node(
                kind=str(row[1]),
                policy=str(row[2]),
                parents=tuple(json.loads(str(row[3]))),
                reference=str(row[4]),
                lifecycle_revision=int(row[5]),
                available=bool(row[6]),
            )
            for row in rows
        }

    async def _load_last_event(self, connection, node_id: str) -> tuple[int, bool, str] | None:
        table = LIFECYCLE_EVENTS_TABLE
        rows = (
            await connection.execute(
                select(table.c.sequence, table.c.available, table.c.reason)
                .where(table.c.scope_id == self.scope_id, table.c.node_id == node_id)
                .order_by(table.c.sequence.desc())
                .limit(1)
            )
        ).all()
        if not rows:
            return None
        return int(rows[0][0]), bool(rows[0][1]), str(rows[0][2])

    async def _write_node(self, connection, node_id: str, node: _Node) -> None:
        table = LIFECYCLE_NODES_TABLE
        await connection.execute(
            table.insert().values(
                scope_id=self.scope_id,
                node_id=node_id,
                kind=node.kind,
                policy=node.policy,
                parents=json.dumps(node.parents),
                reference=node.reference,
                lifecycle_revision=node.lifecycle_revision,
                available=node.available,
            )
        )

    @staticmethod
    def _validate_new_registration(nodes: dict[str, _Node], node_id: str, parents: tuple[str, ...]) -> None:
        if not parents:
            raise SourceLifecycleError("parents-required")
        if len(set(parents)) != len(parents):
            raise SourceLifecycleError("parents-duplicate")
        for parent in parents:
            if parent == node_id:
                raise SourceLifecycleError("dependency-cycle")
            if parent not in nodes:
                raise SourceLifecycleError("parents-missing")

    async def _validate_source_registration(self, source_ref: SourceRef) -> None:
        records = self._runtime.records.for_scope(self.scope_id)
        await records.get_source(source_ref.source_type, source_ref.source_id)

    async def _validate_memory_registration(self, citation: MemoryCitation) -> None:
        memory = self._runtime.memory.for_scope(self.scope_id)
        try:
            await memory.get(GetMemoryEntryRequest(citation=citation))
        except SourceLifecycleError:
            raise
        except Exception as error:
            raise SourceLifecycleError(f"memory-citation-unreadable: {error}") from None

    async def _validate_topic_registration(self, artifact_ref: ArtifactRef) -> None:
        records = self._runtime.records.for_scope(self.scope_id)
        await records.get_artifact_revision(artifact_ref.family, artifact_ref.artifact_id, artifact_ref.revision)

    async def register_source(self, node_id: str, source_ref: SourceRef, /, *, revision: int) -> None:
        if type(revision) is not int or revision < 1:
            raise SourceLifecycleError("revision")
        reference = source_ref.model_dump_json()
        async with self._runtime._scoped_operation(self.scope_id), self._runtime._locked(self.scope_id):
            async with self._database.transaction() as connection:
                nodes = await self._load_nodes(connection)
                existing = nodes.get(node_id)
                if existing is not None:
                    if (
                        existing.kind == "source"
                        and existing.reference == reference
                        and existing.lifecycle_revision == revision
                    ):
                        return
                    raise SourceLifecycleError("node-identity")
                await self._validate_source_registration(source_ref)
                await self._write_node(
                    connection,
                    node_id,
                    _Node(
                        kind="source",
                        policy="",
                        parents=(),
                        reference=reference,
                        lifecycle_revision=revision,
                        available=True,
                    ),
                )

    async def register_memory(self, node_id: str, citation: MemoryCitation, /, *, parents, policy) -> None:
        await self._register_derived("memory", node_id, citation.model_dump_json(), tuple(parents), policy)

    async def register_topic(self, node_id: str, artifact_ref: ArtifactRef, /, *, parents, policy) -> None:
        await self._register_derived("topic", node_id, artifact_ref.model_dump_json(), tuple(parents), policy)

    async def _register_derived(
        self,
        kind: str,
        node_id: str,
        reference: str,
        parents: tuple[str, ...],
        policy: str,
    ) -> None:
        if policy not in _POLICIES:
            raise SourceLifecycleError("policy")
        async with self._runtime._scoped_operation(self.scope_id), self._runtime._locked(self.scope_id):
            async with self._database.transaction() as connection:
                nodes = await self._load_nodes(connection)
                existing = nodes.get(node_id)
                if existing is not None:
                    if (
                        existing.kind == kind
                        and existing.reference == reference
                        and existing.parents == parents
                        and existing.policy == policy
                    ):
                        return
                    raise SourceLifecycleError("node-identity")
                self._validate_new_registration(nodes, node_id, parents)
                if kind == "memory":
                    await self._validate_memory_registration(MemoryCitation.model_validate_json(reference))
                else:
                    await self._validate_topic_registration(ArtifactRef.model_validate_json(reference))
                available = _availability(
                    parents, policy, {key: item.available for key, item in nodes.items()}
                )
                await self._write_node(
                    connection,
                    node_id,
                    _Node(
                        kind=kind,
                        policy=policy,
                        parents=parents,
                        reference=reference,
                        lifecycle_revision=0,
                        available=available,
                    ),
                )
        if kind == "memory":
            await self._sync_memory_nodes((node_id,))

    async def apply(self, event: SourceLifecycleEvent, /) -> tuple[str, ...]:
        replay = False
        changed: tuple[str, ...] = ()
        nodes: dict[str, _Node] = {}
        async with self._runtime._scoped_operation(self.scope_id), self._runtime._locked(self.scope_id):
            async with self._database.transaction() as connection:
                nodes = await self._load_nodes(connection)
                node = nodes.get(event.node_id)
                if node is None or node.kind != "source":
                    raise SourceLifecycleError("unknown-source")
                if node.lifecycle_revision != event.revision:
                    raise SourceLifecycleError("revision-mismatch")
                last = await self._load_last_event(connection, event.node_id)
                if last is not None:
                    sequence, available, reason = last
                    if event.sequence < sequence:
                        raise SourceLifecycleError("event-sequence")
                    if event.sequence == sequence:
                        if available != event.available or reason != event.reason:
                            raise SourceLifecycleError("event-sequence")
                        replay = True
                if not replay:
                    events = LIFECYCLE_EVENTS_TABLE
                    await connection.execute(
                        events.insert().values(
                            scope_id=self.scope_id,
                            node_id=event.node_id,
                            sequence=event.sequence,
                            lifecycle_revision=event.revision,
                            available=event.available,
                            reason=event.reason,
                        )
                    )
                    nodes[event.node_id] = node.model_copy(update={"available": event.available})
                    node_table = LIFECYCLE_NODES_TABLE
                    await connection.execute(
                        node_table.update()
                        .where(
                            node_table.c.scope_id == self.scope_id,
                            node_table.c.node_id == event.node_id,
                        )
                        .values(available=event.available)
                    )
                    children: dict[str, list[str]] = {}
                    for key, item in nodes.items():
                        for parent in item.parents:
                            children.setdefault(parent, []).append(key)
                    pending = deque(children.get(event.node_id, ()))
                    queued = set(pending)
                    changed_keys: list[str] = []
                    while pending:
                        key = pending.popleft()
                        queued.discard(key)
                        item = nodes[key]
                        updated = _availability(
                            item.parents, item.policy, {name: value.available for name, value in nodes.items()}
                        )
                        if updated == item.available:
                            continue
                        nodes[key] = item.model_copy(update={"available": updated})
                        changed_keys.append(key)
                        node_table = LIFECYCLE_NODES_TABLE
                        await connection.execute(
                            node_table.update()
                            .where(node_table.c.scope_id == self.scope_id, node_table.c.node_id == key)
                            .values(available=updated)
                        )
                        for child in children.get(key, ()):
                            if child not in queued:
                                queued.add(child)
                                pending.append(child)
                    changed = tuple(key for key in changed_keys if nodes[key].kind == "memory")
        if replay:
            await self._sync_memory_nodes(
                tuple(key for key, item in nodes.items() if item.kind == "memory"),
                desired={key: item.available for key, item in nodes.items() if item.kind == "memory"},
            )
        else:
            await self._sync_memory_nodes(changed)
        return sorted(key for key, item in nodes.items() if item.available)

    async def _sync_memory_nodes(
        self,
        node_ids: tuple[str, ...],
        /,
        desired: dict[str, bool] | None = None,
    ) -> None:
        """Align native Memory entry states with lifecycle availability.

        Idempotent: entries already in the desired state are left untouched, so
        a successful replay never creates a new Memory revision, and a retry
        after a failed propagation completes the remaining writes.
        """

        if not node_ids:
            return
        async with self._runtime._scoped_operation(self.scope_id), self._runtime._locked(self.scope_id):
            async with self._database.transaction() as connection:
                nodes = await self._load_nodes(connection)
            async with self._runtime._context(self.scope_id) as context:
                service = context.artifacts.memory
                current = await service.head(context.artifacts.memory_artifact_id)
                if current is None:
                    raise SourceLifecycleError("memory-head")
                entries = {entry.entry_id: entry for entry in await service.entries(current)}
                states = {item.entry_id: item.state for item in current.content.manifest.entries}
                to_forget = []
                to_reactivate = []
                for node_id in node_ids:
                    node = nodes.get(node_id)
                    if node is None:
                        raise SourceLifecycleError("unknown-node")
                    citation = MemoryCitation.model_validate_json(node.reference)
                    entry = entries.get(citation.entry_id)
                    if entry is None:
                        raise SourceLifecycleError("memory-entry")
                    wanted = desired[node_id] if desired is not None else node.available
                    if states.get(citation.entry_id) == ("active" if wanted else "inactive"):
                        continue
                    (to_reactivate if wanted else to_forget).append(entry)
                updated = current
                if to_forget:
                    updated = await service.forget(updated, entries=to_forget, reason="source lifecycle")
                if to_reactivate:
                    updated = await service.reactivate(updated, entries=to_reactivate, reason="source lifecycle")
                del updated

    async def resolve(
        self, node_id: str, /, *, purpose: Literal["generation", "audit"]
    ) -> SourceRef | MemoryCitation | ArtifactRef:
        if purpose not in {"generation", "audit"}:
            raise SourceLifecycleError("purpose")
        async with self._database.transaction() as connection:
            nodes = await self._load_nodes(connection)
        node = nodes.get(node_id)
        if node is None:
            raise SourceLifecycleError("unknown-node")
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError("node-unavailable")
        return _reference_for_kind(node.kind, node.reference)


class LifecycleApplication:
    """Select the lifecycle service for one Scope."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycleService:
        return ScopedSourceLifecycleService(self._runtime, scope_id)
