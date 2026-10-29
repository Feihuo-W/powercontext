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

"""Pi session history: one captured Source per visible event on the selected branch.

A Pi session archive is a JSONL append log whose records form a tree through
``parentId`` links. The branch selected by a leaf node is the only history the
model continued from, so this module captures exactly that ancestor chain as
immutable, individually citable evidence events. Control records keep the graph
connectivity but never become text evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from pydantic import BaseModel, Field, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.connectors import Connector, ConnectorRunCompletion, ConnectorRunSession, ConnectorRunStatus
from powercontext.sources.models import Source, SourceMaterialization

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
NonEmptyText = Annotated[str, Field(min_length=1)]

_MESSAGE_TYPE = "message"
_COMPACTION_TYPE = "compaction"
_BRANCH_SUMMARY_TYPE = "branch_summary"
_CUSTOM_MESSAGE_TYPE = "custom_message"
_SESSION_TYPE = "session"
_SESSION_VERSION = 3


class PiSessionInput(BaseModel):
    """A complete Pi v3 session archive plus the branch leaf to capture."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event on the selected branch, with its complete source record."""

    session_id: NonEmptyText
    event_id: NonEmptyText
    parent_id: str | None = None
    event_type: NonEmptyText
    role: NonEmptyText
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """A captured Pi event that can cite the exact original record."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def parse_pi_session(value: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    """Parse the selected branch of one Pi v3 session archive atomically.

    The archive must decode as UTF-8 (a leading BOM is accepted) JSONL whose
    first record is a ``type=session`` header at ``version=3`` with a non-empty
    ``id``. Every later record needs a unique ``id`` and a ``parentId`` that is
    either empty or an already-appeared node, so the file order is a valid
    topological order. Any malformed record, duplicate identity, unknown
    parent or missing leaf fails the whole archive with ``ValueError`` rather
    than returning a partial branch.
    """

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session archive must be valid UTF-8") from error

    records = _read_records(text)
    if not records:
        raise ValueError("Pi session archive must start with a session header")

    header = records[0]
    session_id = _session_id(header)

    parents: dict[str, str | None] = {}
    by_id: dict[str, dict[str, JsonValue]] = {}
    order: list[str] = []
    seen: set[str] = {session_id}

    for record in records[1:]:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("every Pi session record needs a non-empty string id")
        if event_id in seen:
            raise ValueError(f"duplicate Pi session record id {event_id!r}")
        parent = _parent_id(record)
        if parent is not None and parent not in seen:
            raise ValueError(f"record {event_id!r} refers to an unknown parent {parent!r}")
        seen.add(event_id)
        parents[event_id] = parent
        by_id[event_id] = record
        order.append(event_id)

    leaf = value.leaf_id if value.leaf_id is not None else (order[-1] if order else None)
    if leaf is None:
        raise ValueError("Pi session archive has no records to select a leaf from")
    if leaf not in by_id:
        raise ValueError(f"unknown Pi session leaf {leaf!r}")

    chain: list[str] = []
    cursor: str | None = leaf
    while cursor is not None:
        chain.append(cursor)
        cursor = parents[cursor]
    chain.reverse()

    events: list[PiEventInput] = []
    for event_id in chain:
        record = by_id[event_id]
        rendered = _render(record)
        if rendered is None:
            continue
        role, body = rendered
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=event_id,
                parent_id=parents[event_id],
                event_type=str(record.get("type")),
                role=role,
                content=f"{role}: {body}",
                record=record,
            )
        )
    return tuple(events)


def pi_event_source_name(session_id: str, event_id: str, /) -> str:
    """Build the delimiter-safe, bounded Source identity for one Pi event."""

    name = f"{PI_EVENT_SOURCE_NAME}:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(name) <= MAX_SOURCE_ID_LENGTH:
        return name
    digest = hashlib.sha256(f"{session_id}\x00{event_id}".encode()).hexdigest()
    return f"{PI_EVENT_SOURCE_NAME}:{digest}"


def _read_records(text: str) -> list[dict[str, JsonValue]]:
    records: list[dict[str, JsonValue]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Pi session line {index} is not valid JSON: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Pi session line {index} must be a JSON object")
        records.append(record)
    return records


def _session_id(header: dict[str, JsonValue]) -> str:
    if header.get("type") != _SESSION_TYPE:
        raise ValueError("the first Pi session record must have type=session")
    if header.get("version") != _SESSION_VERSION:
        raise ValueError(f"the Pi session header must declare version={_SESSION_VERSION}")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("the Pi session header needs a non-empty string id")
    return session_id


def _parent_id(record: dict[str, JsonValue]) -> str | None:
    parent = record.get("parentId")
    if parent is None:
        return None
    if not isinstance(parent, str) or not parent:
        raise ValueError("record parentId must be null or a non-empty string")
    return parent


def _render(record: dict[str, JsonValue]) -> tuple[str, str] | None:
    """Render one record as ``(role, body)``, or ``None`` without visible text."""

    record_type = record.get("type")
    if record_type == _MESSAGE_TYPE:
        return _render_message(record)
    if record_type in {_COMPACTION_TYPE, _BRANCH_SUMMARY_TYPE}:
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary:
            return None
        return str(record_type), summary
    if record_type == _CUSTOM_MESSAGE_TYPE:
        body = _visible_text(record.get("content"), allow_tool_calls=False)
        if not body:
            return None
        return "custom", body
    return None


def _render_message(record: dict[str, JsonValue]) -> tuple[str, str] | None:
    message = record.get("message")
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if not isinstance(role, str) or not role:
        return None

    if role == "bashExecution":
        command = _as_text(message.get("command"))
        output = _as_text(message.get("output"))
        body = f"$ {command}\n{output}" if output else f"$ {command}"
        return (role, body) if body.strip() else None

    if role == "toolResult":
        tool_name = _as_text(message.get("toolName"))
        tool_call_id = _as_text(message.get("toolCallId"))
        text = _visible_text(message.get("content"), allow_tool_calls=False)
        return role, f"[tool_result {tool_name} {tool_call_id}] {text}"

    body = _visible_text(message.get("content"), allow_tool_calls=True)
    if not body.strip():
        return None
    return role, body


def _visible_text(content: JsonValue, *, allow_tool_calls: bool) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif block_type == "toolCall" and allow_tool_calls:
            parts.append(_render_tool_call(block))
    return "\n".join(parts)


def _render_tool_call(block: dict[str, JsonValue]) -> str:
    name = _as_text(block.get("name"))
    call_id = _as_text(block.get("id"))
    arguments = block.get("arguments")
    if arguments is None:
        arguments = {}
    compact = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"[tool_call {name} {call_id}] {compact}"


def _as_text(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


class PiEventSourceAdapter:
    """Resolve and read Pi event Sources without changing the catalog."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_name(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose one Pi event as standard textual evidence for Memory pipelines."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PiEventSourceAdapter(),
    projections=(PiEventTextEvidenceProjection(),),
)


def _record_digest(record: dict[str, JsonValue]) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class PiSessionConnector(Connector):
    """Import the visible events of the selected Pi branch incrementally."""

    name = PI_SESSION_CONNECTOR_NAME
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        imported = self._imported(session.checkpoint)
        events = parse_pi_session(PiSessionInput(content=self.path.read_bytes(), leaf_id=self.leaf_id))

        for event in events:
            previous = imported.get(event.event_id)
            if previous is not None and previous != _record_digest(event.record):
                raise ValueError(f"imported Pi event {event.event_id!r} was rewritten")

        pending = [event for event in events if event.event_id not in imported]
        for event in pending:
            await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)

        checkpoint: dict[str, JsonValue] = {
            "version": 1,
            "events": {**imported, **{event.event_id: _record_digest(event.record) for event in pending}},
        }
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)

    @staticmethod
    def _imported(checkpoint: JsonValue | None) -> dict[str, str]:
        if not isinstance(checkpoint, dict):
            return {}
        events = checkpoint.get("events")
        if not isinstance(events, dict):
            return {}
        return {key: value for key, value in events.items() if isinstance(key, str) and isinstance(value, str)}


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PiEventInput",
    "PiEventSource",
    "PiEventSourceAdapter",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_name",
]
