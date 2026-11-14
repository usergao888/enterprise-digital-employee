"""Vector store abstraction.

`MemoryVectorStore` is the default: numpy + JSON persistence, zero services to
run, which is what you want for a demo, CI and a laptop. `ChromaVectorStore`
and `QdrantVectorStore` are thin adapters with the same interface, enabled by
`DE_VECTOR_STORE=chroma|qdrant` when the SDK is installed.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    heading: str
    text: str
    ordinal: int = 0
    source_path: str = ""
    vector: list[float] = field(default_factory=list)

    def citation(self) -> str:
        if self.heading:
            return f"{self.title} · {self.heading}"
        return self.title

    def payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "heading": self.heading,
            "text": self.text,
            "ordinal": self.ordinal,
            "source_path": self.source_path,
        }


@dataclass
class Hit:
    chunk: Chunk
    score: float
    retriever: str = "dense"


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk]) -> int: ...
    def search(self, vector: Sequence[float], k: int) -> list[Hit]: ...
    def chunks(self) -> list[Chunk]: ...
    def clear(self) -> None: ...
    def stats(self) -> dict[str, Any]: ...


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class MemoryVectorStore:
    """In-process store with cosine similarity and JSON persistence."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._index_of: dict[str, int] = {}
        self._matrix: np.ndarray | None = None
        self._dim = 0

    # -- write ---------------------------------------------------------
    def upsert(self, chunks: Sequence[Chunk]) -> int:
        added = 0
        for chunk in chunks:
            existing = self._index_of.get(chunk.chunk_id)
            if existing is not None:
                self._chunks[existing] = chunk
                added += 1
                continue
            self._index_of[chunk.chunk_id] = len(self._chunks)
            self._chunks.append(chunk)
            added += 1
        self._rebuild()
        return added

    def clear(self) -> None:
        self._chunks.clear()
        self._index_of.clear()
        self._matrix = None
        self._dim = 0

    def delete_doc(self, doc_id: str) -> int:
        kept = [c for c in self._chunks if c.doc_id != doc_id]
        removed = len(self._chunks) - len(kept)
        if removed:
            self._chunks = kept
            self._index_of = {c.chunk_id: i for i, c in enumerate(self._chunks)}
            self._rebuild()
        return removed

    # -- read ----------------------------------------------------------
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, vector: Sequence[float], k: int) -> list[Hit]:
        if not self._chunks or self._matrix is None:
            return []
        query = np.asarray(vector, dtype=np.float32)
        if query.shape[0] != self._matrix.shape[1]:
            # Dimension mismatch (e.g. embedder swapped): fall back to per-item
            # cosine so the failure is degraded retrieval, never a crash.
            scored = [
                (chunk, _cosine(vector, chunk.vector)) for chunk in self._chunks
            ]
        else:
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm
            scores = self._matrix @ query
            scored = list(zip(self._chunks, scores.tolist()))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            Hit(chunk=chunk, score=float(score), retriever="dense")
            for chunk, score in scored[:k]
        ]

    def stats(self) -> dict[str, Any]:
        docs = {chunk.doc_id for chunk in self._chunks}
        return {
            "backend": "memory",
            "chunks": len(self._chunks),
            "documents": len(docs),
            "dim": self._dim,
        }

    # -- persistence ---------------------------------------------------
    def _rebuild(self) -> None:
        dims = {len(chunk.vector) for chunk in self._chunks if chunk.vector}
        self._dim = max(dims) if dims else 0
        if not self._chunks or self._dim == 0:
            self._matrix = None
            return
        matrix = np.zeros((len(self._chunks), self._dim), dtype=np.float32)
        for row, chunk in enumerate(self._chunks):
            if not chunk.vector:
                continue
            vec = np.asarray(chunk.vector, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            matrix[row, : vec.shape[0]] = vec[: self._dim]
        self._matrix = matrix

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "dim": self._dim,
            "chunks": [asdict(chunk) for chunk in self._chunks],
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "MemoryVectorStore":
        store = cls()
        if not path.exists():
            return store
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("chunks", []):
            chunk = Chunk(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                title=item.get("title", ""),
                heading=item.get("heading", ""),
                text=item.get("text", ""),
                ordinal=item.get("ordinal", 0),
                source_path=item.get("source_path", ""),
                vector=item.get("vector", []),
            )
            store._index_of[chunk.chunk_id] = len(store._chunks)
            store._chunks.append(chunk)
        store._rebuild()
        return store


# ---------------------------------------------------------------------------
# optional adapters
# ---------------------------------------------------------------------------
class ChromaVectorStore:
    """Adapter for ChromaDB (persistent client, cosine space)."""

    def __init__(self, persist_dir: Path, collection: str = "enterprise_kb") -> None:
        import chromadb  # noqa: F401  (optional dependency)

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[c.vector for c in chunks],
            metadatas=[{**c.payload(), "text": c.text} for c in chunks],
        )
        return len(chunks)

    def search(self, vector: Sequence[float], k: int) -> list[Hit]:
        result = self._collection.query(query_embeddings=[list(vector)], n_results=k)
        hits: list[Hit] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for i, chunk_id in enumerate(ids):
            meta = metas[i] or {}
            hits.append(
                Hit(
                    chunk=Chunk(
                        chunk_id=chunk_id,
                        doc_id=meta.get("doc_id", ""),
                        title=meta.get("title", ""),
                        heading=meta.get("heading", ""),
                        text=docs[i] or "",
                        ordinal=int(meta.get("ordinal", 0) or 0),
                    ),
                    score=1.0 - float(dists[i]) if i < len(dists) else 0.0,
                    retriever="dense",
                )
            )
        return hits

    def chunks(self) -> list[Chunk]:
        result = self._collection.get(include=["documents", "metadatas"])
        out: list[Chunk] = []
        for i, chunk_id in enumerate(result.get("ids") or []):
            meta = (result.get("metadatas") or [{}])[i] or {}
            out.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=meta.get("doc_id", ""),
                    title=meta.get("title", ""),
                    heading=meta.get("heading", ""),
                    text=(result.get("documents") or [""])[i] or "",
                    ordinal=int(meta.get("ordinal", 0) or 0),
                )
            )
        return out

    def clear(self) -> None:
        self._client.delete_collection(self._collection.name)

    def stats(self) -> dict[str, Any]:
        return {"backend": "chroma", "chunks": self._collection.count()}


class QdrantVectorStore:
    """Adapter for Qdrant (local `:memory:` or a remote server)."""

    def __init__(self, collection: str, dim: int, url: str | None = None) -> None:
        from qdrant_client import QdrantClient  # noqa: F401 (optional)
        from qdrant_client.models import Distance, VectorParams

        self._dim = dim
        self._collection = collection
        self._client = QdrantClient(url=url) if url else QdrantClient(":memory:")
        existing = {c.name for c in self._client.get_collections().collections}
        if collection not in existing:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=i, vector=list(c.vector), payload={**c.payload(), "text": c.text})
            for i, c in enumerate(chunks)
        ]
        if points:
            self._client.upsert(collection_name=self._collection, points=points)
        return len(points)

    def search(self, vector: Sequence[float], k: int) -> list[Hit]:
        result = self._client.query_points(
            collection_name=self._collection, query=list(vector), limit=k
        )
        hits: list[Hit] = []
        for point in result.points:
            payload = point.payload or {}
            hits.append(
                Hit(
                    chunk=Chunk(
                        chunk_id=payload.get("chunk_id", ""),
                        doc_id=payload.get("doc_id", ""),
                        title=payload.get("title", ""),
                        heading=payload.get("heading", ""),
                        text=payload.get("text", ""),
                    ),
                    score=float(point.score),
                    retriever="dense",
                )
            )
        return hits

    def chunks(self) -> list[Chunk]:
        points, _ = self._client.scroll(
            collection_name=self._collection, limit=10000, with_payload=True
        )
        out: list[Chunk] = []
        for point in points:
            payload = point.payload or {}
            out.append(
                Chunk(
                    chunk_id=payload.get("chunk_id", ""),
                    doc_id=payload.get("doc_id", ""),
                    title=payload.get("title", ""),
                    heading=payload.get("heading", ""),
                    text=payload.get("text", ""),
                )
            )
        return out

    def clear(self) -> None:
        self._client.delete_collection(self._collection)

    def stats(self) -> dict[str, Any]:
        return {"backend": "qdrant", "chunks": self._client.count(self._collection).count}


def build_store(name: str, index_dir: Path, dim: int) -> VectorStore:
    if name == "chroma":
        return ChromaVectorStore(index_dir / "chroma")
    if name == "qdrant":
        return QdrantVectorStore(collection="enterprise_kb", dim=dim)
    return MemoryVectorStore()
