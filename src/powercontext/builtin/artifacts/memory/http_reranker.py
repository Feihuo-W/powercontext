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

"""HTTP cross-encoder reranking with global rank mapping across batches."""

from __future__ import annotations

import math
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

DEFAULT_HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"
_SENTINEL = object()


class HTTPMemoryReranker:
    """Score coarse Memory candidates through one HTTP service, batch by batch.

    Global one-based ranks come from the caller's full candidate pool and stay
    stable across batches. Every batch response must cover exactly its own ranks
    once; selection then happens over the whole pool by descending score with
    original rank as the tie-breaker.
    """

    def __init__(
        self,
        endpoint: str,
        /,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = DEFAULT_HTTP_RERANK_POLICY_ID,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("reranker endpoint must be a non-empty string")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("reranker batch_size must be a positive integer")
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("reranker policy_id must be a non-empty string")
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Batch the pool, validate every response, then select globally."""

        _validate_bounds(len(candidates), limit)
        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            ranks = tuple(range(start + 1, start + 1 + len(batch)))
            payload = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": candidate.text} for rank, candidate in zip(ranks, batch, strict=True)
                ],
            }
            response = await self._client.post(self._endpoint, json=payload)
            requests += 1
            response.raise_for_status()
            scores.update(_parse_scores(response, ranks))

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        selected = tuple(rank for rank, _score in ordered[:limit])
        return MemoryRerankDecision(selected_ranks=selected, usage=InferenceUsage(requests=requests))


def _parse_scores(response: httpx.Response, expected_ranks: tuple[int, ...]) -> dict[int, float]:
    try:
        payload = response.json()
    except ValueError as error:
        raise ValueError("reranker response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("reranker response must be a JSON object")
    if set(payload) != {"scores"}:
        raise ValueError("reranker response must contain exactly the scores field")
    entries = payload["scores"]
    if not isinstance(entries, list):
        raise ValueError("reranker scores must be a list")

    scores: dict[int, float] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"rank", "score"}:
            raise ValueError("each reranker score must contain exactly rank and score")
        rank = entry["rank"]
        score = entry["score"]
        if type(rank) is not int:
            raise ValueError("reranker rank must be an integer")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("reranker score must be a finite number")
        if not math.isfinite(score):
            raise ValueError("reranker score must be finite")
        if rank not in expected_ranks:
            raise ValueError(f"reranker returned rank {rank} outside the requested batch")
        if rank in scores:
            raise ValueError(f"reranker returned duplicate rank {rank}")
        scores[rank] = float(score)

    missing = [rank for rank in expected_ranks if rank not in scores]
    if missing:
        raise ValueError(f"reranker response is missing ranks {missing}")
    return scores


def _validate_bounds(candidate_count: int, limit: int) -> None:
    if candidate_count < 1:
        raise ValueError("rerank requires at least one candidate")
    if type(limit) is not int or limit < 1 or limit > candidate_count:
        raise ValueError("rerank limit must be between one and the candidate count")


__all__ = ["DEFAULT_HTTP_RERANK_POLICY_ID", "HTTPMemoryReranker"]
