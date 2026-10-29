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

"""Page-level evidence for complete financial-report PDFs.

Every physical page of the original PDF becomes one ``CAPTURED`` Source that
records the document identity, the digest of the exact bytes it came from and
its physical page number, so a citation always resolves back to one page of one
document revision. Retrieval answers a natural-language question with exact
page citations read back from the scoped Memory instead of synthesising text.
"""

from __future__ import annotations

import hashlib
import io
import re
from typing import Annotated, Any

from pydantic import BaseModel, Field
from pypdf import PdfReader

from powercontext.artifacts import MemoryCitation
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

PDF_PAGE_SOURCE_NAME = "pdf-page"
NonEmptyText = Annotated[str, Field(min_length=1)]

# A single natural-language question is turned into several short lexical probes:
# the shared admission floor accepts one matched term for probes of two terms or
# fewer, which keeps recall high while every hit is still a real Memory record.
_MAX_PROBE_TERMS = 8
_PROBE_POOL = 40
_MIN_TERM_LENGTH = 3

_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being below
    between both but by can cannot could did do does doing down during each few for from further had
    has have having he her here hers herself him himself his how i if in into is it its itself just
    me more most my myself no nor not now of off on once only or other ought our ours ourselves out
    over own same she should so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was we were what when where which while who
    whom why with would you your yours yourself yourselves
    any much many might must shall may also reported report reported has have had does did
    """.split()
)


class PDFDocumentInput(BaseModel):
    """A complete original PDF plus the identity it is captured under."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical page of one PDF revision."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """A captured PDF page that can cite the exact original page."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def parse_pdf_document(value: PDFDocumentInput, /) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of one PDF atomically.

    The whole document is opened and every page is extracted before any result
    is returned, so a damaged or truncated file fails with ``ValueError``
    instead of yielding a partial page list. Blank pages are kept in order with
    empty text; page numbers start at the first physical page. Documents that
    only carry an empty user password are readable without a secret.
    """

    document_id = value.document_id
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-blank string")
    content = value.content
    if not isinstance(content, (bytes, bytearray)) or not content:
        raise ValueError("PDF content must be non-empty bytes")

    digest = hashlib.sha256(content).hexdigest()
    reader = _open(content)
    try:
        total = len(reader.pages)
        if total < 1:
            raise ValueError("PDF document has no pages")
        pages: list[PDFPageInput] = []
        for index in range(total):
            pages.append(
                PDFPageInput(
                    document_id=document_id,
                    document_sha256=digest,
                    page_number=index + 1,
                    text=_page_text(reader.pages[index]),
                )
            )
    except ValueError:
        raise
    except Exception as error:  # noqa: BLE001 - any parser failure is a document failure
        raise ValueError(f"PDF document could not be fully parsed: {error}") from error
    return tuple(pages)


def _open(content: bytes) -> Any:
    try:
        reader = PdfReader(io.BytesIO(content), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF document requires a user password")
        return reader
    except ValueError:
        raise
    except Exception as error:  # noqa: BLE001 - pypdf raises a wide family of errors
        raise ValueError(f"PDF document could not be opened: {error}") from error


def _page_text(page: Any) -> str:
    try:
        text = page.extract_text()
    except Exception as error:  # noqa: BLE001 - a failing page fails the document
        raise ValueError(f"PDF page could not be extracted: {error}") from error
    return text if isinstance(text, str) else ""


def pdf_page_source_name(document_id: str, document_sha256: str, page_number: int, /) -> str:
    """Build the boundary-safe, bounded Source identity for one PDF page."""

    prefix = f"{PDF_PAGE_SOURCE_NAME}:{_quote(document_id)}:{document_sha256}:{page_number}"
    if len(prefix) <= MAX_SOURCE_ID_LENGTH:
        return prefix
    digest = hashlib.sha256(f"{document_id}\x00{document_sha256}\x00{page_number}".encode()).hexdigest()
    return f"{PDF_PAGE_SOURCE_NAME}:{digest}"


def _quote(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


class PDFPageSourceAdapter:
    """Resolve and read page-level PDF Sources without changing the catalog."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_name(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose one PDF page as standard textual evidence for Memory pipelines."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDFPageSourceAdapter(),
    projections=(PDFPageTextEvidenceProjection(),),
)


async def search_pdf_pages(memory: Any, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Return exact page citations for one natural-language question.

    The question is probed lexically through the public Memory search API with
    several short term sets, because a long natural-language query is admitted
    only when it covers many distinct terms. Candidates from every probe are
    merged, ranked by how much of the question they cover and then by the
    stored score, and finally read back so only distinct real pages are cited.
    """

    if limit < 1 or not isinstance(question, str) or not question.strip():
        return ()

    from powercontext.builtin.runtime.models import GetMemoryEntryRequest, SearchMemoryRequest

    terms = _probe_terms(question)
    probes = [" ".join(terms)] if terms else []
    probes.extend(terms[:_MAX_PROBE_TERMS])
    if not probes:
        probes = [question]

    pool: dict[str, Any] = {}
    for probe in probes:
        page = await memory.search(SearchMemoryRequest(query=probe, mode="fts", limit=_PROBE_POOL))
        for hit in page.hits:
            previous = pool.get(hit.entry_id)
            if previous is None or hit.score > previous.score:
                pool[hit.entry_id] = hit

    ranked = sorted(pool.values(), key=lambda hit: (-_coverage(hit.text, terms), -hit.score))

    citations: list[MemoryCitation] = []
    seen_pages: set[str] = set()
    for hit in ranked:
        if len(citations) >= limit:
            break
        citation = MemoryCitation(
            memory_ref=hit.memory_ref,
            entry_id=hit.entry_id,
            entry_version_id=hit.entry_version_id,
        )
        entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
        pages = [
            reference
            for reference in entry.sources
            if reference.source_type == PDF_PAGE_SOURCE_NAME and reference.source_id not in seen_pages
        ]
        if not pages:
            continue
        for reference in pages:
            seen_pages.add(reference.source_id)
            citations.append(citation)
            break
    return tuple(citations)


def _probe_terms(question: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[a-z]+|\d+", question.lower()):
        if token in _STOPWORDS:
            continue
        if not token.isdigit() and len(token) < _MIN_TERM_LENGTH:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def _coverage(text: str, terms: list[str]) -> int:
    if not terms:
        return 0
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "PDFPageSourceAdapter",
    "PDFPageTextEvidenceProjection",
    "parse_pdf_document",
    "pdf_page_source_name",
    "search_pdf_pages",
]
