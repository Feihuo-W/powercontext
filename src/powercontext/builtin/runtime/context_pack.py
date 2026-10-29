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

"""Chunked context packs and prefix-reusing deltas for incremental transport.

A full pack carries every UTF-8-safe chunk of one prepared context plus a
SHA-256 digest over its wire fields. A delta records how many leading chunks the
receiver may keep from an existing pack and the target suffix it must append, so
a receiver rebuilds the exact target bytes from serialized data alone.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from typing import Any

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512

_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})


class ContextPackError(ValueError):
    """Raised when a pack, delta, or chunk boundary violates the wire contract."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """An immutable, self-verifying set of context chunks."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...] | list[bytes]) -> ContextPack:
        """Build a full pack from already-chunked byte items."""

        chunks = _validate_items(items)
        digest = _digest_for(chunks)
        # ``byte_count`` counts its own digits, so settle the fixpoint explicitly.
        byte_count = 0
        while True:
            payload = {
                "schema": PACK_SCHEMA,
                "items": _encode_items(chunks),
                "digest": digest,
                "byte_count": byte_count,
            }
            measured = len(_dumps(payload))
            if measured == byte_count:
                break
            byte_count = measured
        return cls(chunks, digest, byte_count)

    @classmethod
    def from_content(cls, content: str, /, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Split UTF-8 content into deterministic chunks of at most ``chunk_bytes``."""

        if not isinstance(content, str):
            raise ContextPackError("content must be a string")
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ContextPackError("chunk_bytes must be a positive integer")
        return cls.build(_chunk_utf8(content, chunk_bytes))

    @classmethod
    def from_bytes(cls, data: bytes, /) -> ContextPack:
        """Decode and verify a serialized full pack."""

        payload = _load_object(data)
        _require_fields(payload, _PACK_FIELDS, "pack")
        if payload["schema"] != PACK_SCHEMA:
            raise ContextPackError(f"unsupported pack schema {payload['schema']!r}")
        chunks = _decode_items(payload["items"], "items")
        digest = payload["digest"]
        if not isinstance(digest, str):
            raise ContextPackError("pack digest must be a string")
        if digest != _digest_for(chunks):
            raise ContextPackError("pack digest does not match its items")
        byte_count = payload["byte_count"]
        if type(byte_count) is not int or byte_count != len(data):
            raise ContextPackError("pack byte_count does not match the serialized size")
        return cls(chunks, digest, byte_count)

    def to_bytes(self) -> bytes:
        """Serialize this pack with the canonical wire encoding."""

        return _dumps(
            {
                "schema": PACK_SCHEMA,
                "items": _encode_items(self.items),
                "digest": self.digest,
                "byte_count": self.byte_count,
            }
        )


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """A prefix-preserving delta that rebuilds one target pack from a base pack."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Reuse the longest common leading chunk run, then carry the target suffix."""

        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("between requires two ContextPack values")
        limit = min(len(base.items), len(target.items))
        prefix = 0
        while prefix < limit and base.items[prefix] == target.items[prefix]:
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    @classmethod
    def from_bytes(cls, data: bytes, /) -> ContextPackDelta:
        """Decode a serialized delta and validate its wire contract."""

        payload = _load_object(data)
        _require_fields(payload, _DELTA_FIELDS, "delta")
        if payload["schema"] != DELTA_SCHEMA:
            raise ContextPackError(f"unsupported delta schema {payload['schema']!r}")
        base_digest = payload["base_digest"]
        target_digest = payload["target_digest"]
        if not isinstance(base_digest, str) or not isinstance(target_digest, str):
            raise ContextPackError("delta digests must be strings")
        prefix = payload["prefix_items"]
        if type(prefix) is not int:
            raise ContextPackError("prefix_items must be an integer")
        if prefix < 0:
            raise ContextPackError("prefix_items must not be negative")
        appended = _decode_items(payload["appended"], "appended")
        return cls(base_digest, target_digest, prefix, appended)

    def to_bytes(self) -> bytes:
        """Serialize this delta with the canonical wire encoding."""

        return _dumps(
            {
                "schema": DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": _encode_items(self.appended),
            }
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta, /) -> ContextPack:
    """Rebuild the target pack from a base pack plus a verified delta."""

    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("apply_context_pack_delta requires a pack and a delta")
    if base.digest != delta.base_digest:
        raise ContextPackError("delta base digest does not match the supplied pack")
    if type(delta.prefix_items) is not int or delta.prefix_items < 0:
        raise ContextPackError("prefix_items must be a non-negative integer")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("prefix_items exceeds the base pack chunk count")
    rebuilt = ContextPack.build((*base.items[: delta.prefix_items], *delta.appended))
    if rebuilt.digest != delta.target_digest:
        raise ContextPackError("rebuilt pack digest does not match the delta target")
    return rebuilt


def _chunk_utf8(content: str, chunk_bytes: int) -> tuple[bytes, ...]:
    chunks: list[bytes] = []
    current = bytearray()
    for character in content:
        encoded = character.encode("utf-8")
        if len(encoded) > chunk_bytes:
            raise ContextPackError(f"a single character exceeds the {chunk_bytes}-byte chunk limit")
        if len(current) + len(encoded) > chunk_bytes:
            chunks.append(bytes(current))
            current = bytearray()
        current.extend(encoded)
    if current:
        chunks.append(bytes(current))
    return tuple(chunks)


def _validate_items(items: object) -> tuple[bytes, ...]:
    if isinstance(items, (bytes, bytearray, str)):
        raise ContextPackError("pack items must be a sequence of byte strings")
    try:
        values = tuple(items)  # type: ignore[arg-type]
    except TypeError as error:
        raise ContextPackError("pack items must be a sequence of byte strings") from error
    chunks = []
    for value in values:
        if not isinstance(value, (bytes, bytearray)):
            raise ContextPackError("pack items must be byte strings")
        chunks.append(bytes(value))
    return tuple(chunks)


def _digest_for(items: tuple[bytes, ...]) -> str:
    canonical = _dumps({"schema": PACK_SCHEMA, "items": _encode_items(items)})
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _dumps(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _load_object(data: bytes) -> dict[str, Any]:
    if not isinstance(data, (bytes, bytearray)):
        raise ContextPackError("serialized payload must be bytes")
    try:
        payload = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError("payload is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ContextPackError("payload must be a JSON object")
    return payload


def _require_fields(payload: dict[str, Any], expected: frozenset[str], label: str) -> None:
    missing = expected - payload.keys()
    if missing:
        raise ContextPackError(f"{label} is missing fields: {sorted(missing)}")
    extra = payload.keys() - expected
    if extra:
        raise ContextPackError(f"{label} has unexpected fields: {sorted(extra)}")


def _encode_items(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decode_items(value: object, label: str) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError(f"{label} must be a list of Base64 strings")
    decoded = []
    for element in value:
        if not isinstance(element, str):
            raise ContextPackError(f"{label} entries must be Base64 strings")
        try:
            decoded.append(base64.b64decode(element.encode("ascii"), validate=True))
        except (binascii.Error, UnicodeEncodeError, ValueError) as error:
            raise ContextPackError(f"{label} contains invalid Base64 data") from error
    return tuple(decoded)


__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "DELTA_SCHEMA",
    "PACK_SCHEMA",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
