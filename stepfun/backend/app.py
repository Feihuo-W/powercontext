# Copyright (c) 2026 OceanBase competition - StepFun challenge.
#
# Backend for the on-device mini knowledge base.
#
# All embedding inference happens IN THE BROWSER (Transformers.js + WebGPU/WASM).
# This backend only:
#   1. stores documents, their browser-computed vectors and metadata into seekdb;
#   2. executes hybrid (full-text + vector) search inside seekdb;
#   3. never calls any cloud inference API.
#
# seekdb connection is configured via environment variables (see .env.example).

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import pyseekdb
from pyseekdb import HNSWConfiguration

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEKDB_HOST = os.environ.get("SEEKDB_HOST", "127.0.0.1")
SEEKDB_PORT = int(os.environ.get("SEEKDB_PORT", "2881"))
SEEKDB_USER = os.environ.get("SEEKDB_USER", "root")
SEEKDB_PASSWORD = os.environ.get("SEEKDB_PASSWORD", "")
SEEKDB_DATABASE = os.environ.get("SEEKDB_DATABASE", "knowledge_base")
COLLECTION_NAME = os.environ.get("SEEKDB_COLLECTION", "mini_kb")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))  # all-MiniLM-L6-v2

app = FastAPI(title="StepFun mini knowledge base backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: dict[str, Any] = {"client": None, "collection": None, "error": None}


def _connect() -> None:
    client = pyseekdb.Client(
        host=SEEKDB_HOST,
        port=SEEKDB_PORT,
        database=SEEKDB_DATABASE,
        user=SEEKDB_USER,
        password=SEEKDB_PASSWORD,
    )
    # embedding_function=None: vectors are computed in the browser and passed
    # explicitly, so the server must never run an embedding model itself.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration=HNSWConfiguration(dimension=EMBEDDING_DIM, distance="cosine"),
        embedding_function=None,
    )
    _state["client"] = client
    _state["collection"] = collection
    _state["error"] = None


@app.on_event("startup")
def startup() -> None:
    try:
        _connect()
    except Exception as error:  # noqa: BLE001 - surface the reason to the UI
        _state["error"] = f"{type(error).__name__}: {error}"


def _collection():
    if _state["collection"] is None:
        try:
            _connect()
        except Exception as error:  # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail=(
                    "seekdb is not reachable. Start seekdb (OceanBase Desktop / "
                    f"Docker) first. Reason: {type(error).__name__}: {error}"
                ),
            ) from None
    return _state["collection"]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class DocumentIn(BaseModel):
    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[DocumentIn]


class SearchRequest(BaseModel):
    query_text: str = Field(min_length=1)
    query_embedding: list[float]
    top_k: int = Field(default=5, ge=1, le=50)
    # full-text candidate window; the fusion is performed by seekdb (RRF)
    n_fulltext: int = Field(default=20, ge=1, le=200)
    n_vector: int = Field(default=20, ge=1, le=200)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    connected = _state["collection"] is not None
    return {
        "connected": connected,
        "error": _state["error"],
        "host": SEEKDB_HOST,
        "port": SEEKDB_PORT,
        "database": SEEKDB_DATABASE,
        "collection": COLLECTION_NAME,
        "embedding_dim": EMBEDDING_DIM,
    }


@app.post("/api/ingest")
def ingest(request: IngestRequest) -> dict[str, Any]:
    if not request.documents:
        return {"added": 0, "elapsed_ms": 0.0}
    collection = _collection()
    started = time.perf_counter()
    try:
        collection.add(
            ids=[doc.id for doc in request.documents],
            documents=[doc.text for doc in request.documents],
            embeddings=[doc.embedding for doc in request.documents],
            metadatas=[doc.metadata for doc in request.documents],
        )
        # HNSW is maintained asynchronously on seekdb >= 1.3.0; make the new
        # vectors searchable right away so "import -> search" feels immediate.
        refresh = getattr(collection, "refresh_index", None)
        if callable(refresh):
            refresh()
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"seekdb write failed: {error}") from None
    return {
        "added": len(request.documents),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


@app.post("/api/search")
def search(request: SearchRequest) -> dict[str, Any]:
    collection = _collection()
    started = time.perf_counter()
    try:
        results = collection.hybrid_search(
            query={
                "where_document": {"$contains": request.query_text},
                "n_results": request.n_fulltext,
            },
            knn={
                "query_embeddings": [request.query_embedding],
                "n_results": request.n_vector,
            },
            rank={"rrf": {}},
            n_results=request.top_k,
            include=["documents", "metadatas"],
        )
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"seekdb search failed: {error}") from None
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    hits: list[dict[str, Any]] = []
    ids = results.get("ids", [[]])[0] if isinstance(results, dict) else getattr(results, "ids", [[]])[0]
    documents = results.get("documents", [[]])[0] if isinstance(results, dict) else getattr(results, "documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0] if isinstance(results, dict) else getattr(results, "metadatas", [[]])[0]
    distances = (
        results.get("distances", [[]])[0]
        if isinstance(results, dict)
        else getattr(results, "distances", [[]])[0]
    )
    for index, doc_id in enumerate(ids):
        hits.append(
            {
                "id": doc_id,
                "text": documents[index] if index < len(documents) else None,
                "metadata": metadatas[index] if index < len(metadatas) else {},
                "distance": distances[index] if index < len(distances) else None,
            }
        )
    return {"hits": hits, "elapsed_ms": elapsed_ms, "backend": "seekdb hybrid (RRF)"}


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    collection = _collection()
    count = 0
    try:
        count = collection.count()
    except Exception:  # noqa: BLE001 - count() is best-effort
        pass
    return {"count": count, "collection": COLLECTION_NAME}


# ---------------------------------------------------------------------------
# Static frontend (served by the same process for simplicity)
# ---------------------------------------------------------------------------

from fastapi.staticfiles import StaticFiles  # noqa: E402

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
