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

"""Row-level CSV Sources with exact record identity and evidence projection.

Each data row of an offline CSV document becomes one ``CAPTURED`` Source whose
identity is derived from the document and the primary-key value, so citations
keep pointing at the same row across reordering and process restarts.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Annotated
from urllib.parse import quote

from pydantic import BaseModel, Field, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

CSV_ROW_SOURCE_NAME = "csv-row"
NonEmptyText = Annotated[str, Field(min_length=1)]


class CSVDocumentInput(BaseModel):
    """A complete offline CSV document to be parsed into row-level records."""

    document_id: NonEmptyText
    primary_key: NonEmptyText
    content: bytes


class CSVRowInput(BaseModel):
    """One parsed CSV record in original header order."""

    document_id: NonEmptyText
    primary_key: NonEmptyText
    primary_key_value: NonEmptyText
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """A captured CSV row that can cite the exact original record."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def parse_csv_document(value: CSVDocumentInput, /) -> tuple[CSVRowInput, ...]:
    """Parse and validate one CSV document atomically, returning its data rows.

    The document is decoded as strict UTF-8 (a leading BOM is accepted) and read
    with ``csv.reader`` semantics under ``strict=True``. The first record is the
    header; header fields must be non-blank and unique and are compared
    literally, so surrounding spaces are significant. Data record widths must
    match the header, primary-key values must be non-blank and unique, and any
    error fails the whole document with ``ValueError`` instead of returning
    partial rows. Completely empty physical records after the header are ignored.
    """

    document_id = value.document_id
    primary_key = value.primary_key
    _validate_document_id(document_id)
    _validate_primary_key(primary_key)

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be valid UTF-8") from error

    records = _read_records(text)
    if not records:
        raise ValueError("CSV document must contain a header record")

    header = records[0]
    _validate_header(header)

    try:
        key_index = header.index(primary_key)
    except ValueError:
        raise ValueError(f"primary_key {primary_key!r} is not an exact header column name") from None

    columns = tuple(header)
    width = len(columns)
    seen_keys: set[str] = set()
    rows: list[CSVRowInput] = []

    for record in records[1:]:
        if not record:
            continue
        if len(record) != width:
            raise ValueError(f"CSV record has {len(record)} fields, expected {width}")
        key_value = record[key_index]
        if not key_value.strip():
            raise ValueError("primary key values must be non-blank")
        if key_value in seen_keys:
            raise ValueError(f"duplicate primary key value {key_value!r} in document")
        seen_keys.add(key_value)
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=key_value,
                columns=columns,
                values=tuple(record),
            )
        )

    return tuple(rows)


def csv_row_source_id(document_id: str, primary_key_value: str, /) -> str:
    """Build the delimiter-safe, bounded Source identity for one CSV row."""

    _validate_document_id(document_id)
    if not isinstance(primary_key_value, str):
        raise ValueError("primary_key_value must be a string")
    if not primary_key_value.strip():
        raise ValueError("primary_key_value must be non-blank")

    source_id = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"CSV row source id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


def _read_records(text: str) -> list[list[str]]:
    reader = csv.reader(
        io.StringIO(text, newline=""),
        delimiter=",",
        quotechar='"',
        doublequote=True,
        strict=True,
    )
    try:
        return list(reader)
    except csv.Error as error:
        raise ValueError(f"malformed CSV document: {error}") from error


def _validate_document_id(document_id: str) -> None:
    if not isinstance(document_id, str):
        raise ValueError("document_id must be a string")
    if not document_id:
        raise ValueError("document_id must be non-empty")
    if document_id != document_id.strip():
        raise ValueError("document_id must not contain leading or trailing whitespace")


def _validate_primary_key(primary_key: str) -> None:
    if not isinstance(primary_key, str):
        raise ValueError("primary_key must be a string")
    if not primary_key:
        raise ValueError("primary_key must be non-empty")


def _validate_header(header: list[str]) -> None:
    if not header:
        raise ValueError("CSV header must not be empty")
    seen: set[str] = set()
    for column in header:
        if not column.strip():
            raise ValueError("CSV header column names must be non-blank")
        if column in seen:
            raise ValueError(f"duplicate CSV header column name {column!r}")
        seen.add(column)


def _row_mapping(columns: tuple[str, ...], values: tuple[str, ...]) -> dict[str, JsonValue]:
    return dict(zip(columns, values, strict=True))


class CSVRowSourceAdapter:
    """Resolve and read row-level CSV Sources without changing the catalog."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one CSV row as standard textual evidence for Memory pipelines."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = _row_mapping(source.columns, source.values)
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSVRowSourceAdapter(),
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowSourceAdapter",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
