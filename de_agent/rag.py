"""RAG stack: markdown-aware chunking, BM25, hybrid fusion, rerank, citations.

Design notes worth defending in an interview:

* **Chunking** is structure-first (headings) then size-bounded with sentence
  overlap. Fixed-size windows cut policy documents mid-clause and destroy the
  "which rule applies to whom" context that enterprise QA depends on.
* **Hybrid retrieval** matters because a hashing/dense embedding misses exact
  identifiers (工单号, 金额, 制度编号) while BM25 nails them; reciprocal-rank
  fusion needs no score calibration between the two.
* **Reranking** is lexical by default (no GPU, no model download) but the
  interface accepts a cross-encoder, so quality can be upgraded without
  touching the graph.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .config import Settings
from .embeddings import Embedder
from .textutil import (
    coverage,
    keyword_overlap,
    normalize,
    split_sentences,
    tokenize,
)
from .vectorstore import Chunk, Hit, VectorStore

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_MD_NOISE = re.compile(r"(`{3}.*?`{3})", re.DOTALL)


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------
def _join_sentences(sentences: Sequence[str]) -> str:
    out = ""
    for sentence in sentences:
        if not out:
            out = sentence
        elif re.search(r"[。！？；]$", out):
            out += sentence
        else:
            out += " " + sentence
    return out


def _window(text: str, size: int, overlap: int) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    windows: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if current and current_len + len(sentence) > size:
            windows.append(_join_sentences(current))
            tail: list[str] = []
            tail_len = 0
            for previous in reversed(current):
                if tail_len + len(previous) > overlap:
                    break
                tail.insert(0, previous)
                tail_len += len(previous)
            current = tail
            current_len = tail_len
        current.append(sentence)
        current_len += len(sentence)
    if current:
        windows.append(_join_sentences(current))
    return [w for w in windows if w.strip()]


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return [(heading, body)] preserving heading hierarchy labels."""
    sections: list[tuple[str, str]] = []
    heading_stack: list[tuple[int, str]] = []
    body: list[str] = []
    current_heading = ""

    def flush() -> None:
        joined = "\n".join(body).strip()
        if joined:
            sections.append((current_heading, joined))

    for line in text.splitlines():
        match = _HEADING.match(line.strip())
        if match:
            flush()
            body = []
            level = len(match.group(1))
            heading_stack = [(lv, h) for lv, h in heading_stack if lv < level]
            heading_stack.append((level, match.group(2).strip()))
            current_heading = " > ".join(h for _, h in heading_stack)
        else:
            body.append(line)
    flush()
    return sections


def chunk_document(
    doc_id: str, title: str, text: str, *, size: int, overlap: int, source_path: str = ""
) -> list[Chunk]:
    sections = _split_sections(text) or [("", text)]
    # A document's preamble ("本流程描述…由人力资源部负责执行。") is 30-60 chars of
    # pure throat-clearing. Left alone it becomes a standalone chunk that ranks
    # #1 for "入职要带什么材料" while containing no answer at all. Merge it into
    # the first real section instead.
    if len(sections) > 1 and len(sections[0][1]) < 90:
        preamble = sections[0][1]
        heading, body = sections[1]
        sections = [(heading, f"{preamble}\n{body}")] + sections[2:]

    chunks: list[Chunk] = []
    ordinal = 0
    for heading, body in sections:
        for window in _window(body, size, overlap):
            if len(window) < 40 and heading:
                window = f"{heading}\n{window}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{ordinal:03d}",
                    doc_id=doc_id,
                    title=title,
                    heading=heading,
                    text=window.strip(),
                    ordinal=ordinal,
                    source_path=source_path,
                )
            )
            ordinal += 1
    return chunks


def _doc_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped:
            return stripped[:60]
    return path.stem


def load_documents(kb_dir: Path) -> list[tuple[str, str, str, str]]:
    """Return [(doc_id, title, text, source_path)] for every .md/.txt under kb_dir."""
    documents: list[tuple[str, str, str, str]] = []
    if not kb_dir.exists():
        return documents
    for path in sorted(kb_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".markdown"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        doc_id = path.relative_to(kb_dir).as_posix()
        documents.append((doc_id, _doc_title(path, text), text, str(path)))
    return documents


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
class BM25Index:
    """Okapi BM25 over the chunk corpus, rebuilt lazily from the store."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._df: dict[str, int] = {}
        self._avgdl = 0.0
        self._n = 0
        self._dirty = True

    def build(self, chunks: Sequence[Chunk]) -> "BM25Index":
        self._chunks = list(chunks)
        self._tokens = []
        self._df = {}
        for chunk in self._chunks:
            # Title/heading terms are repeated so they count as high-value hits.
            raw = f"{chunk.title} {chunk.heading} {chunk.text}"
            tokens = tokenize(raw)
            weights = tokenize(chunk.title) * 2 + tokenize(chunk.heading)
            tokens = tokens + weights
            self._tokens.append(tokens)
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1
        self._n = len(self._chunks)
        self._avgdl = (
            sum(len(t) for t in self._tokens) / self._n if self._n else 0.0
        )
        self._dirty = False
        return self

    @property
    def size(self) -> int:
        return self._n

    def idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> list[Hit]:
        if self._dirty or self._n == 0:
            return []
        terms = tokenize(query)
        if not terms:
            return []
        scored: list[tuple[int, float]] = []
        for index, tokens in enumerate(self._tokens):
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            length = len(tokens) or 1
            score = 0.0
            for term in terms:
                freq = tf.get(term)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * length / (self._avgdl or 1))
                score += self.idf(term) * freq * (self.k1 + 1) / denom
            if score > 0:
                scored.append((index, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        top = scored[:k]
        if not top:
            return []
        best = top[0][1] or 1.0
        return [
            Hit(chunk=self._chunks[index], score=score / best, retriever="bm25")
            for index, score in top
        ]


# ---------------------------------------------------------------------------
# fusion + rerank
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[Hit]],
    k: int = 60,
    limit: int = 20,
    weights: Sequence[float] | None = None,
) -> list[Hit]:
    """Weighted RRF: `score(d) = Σ w_i / (k + rank_i(d))`.

    The weight matters for multi-query retrieval. Treating the user's original
    question and a 4-character sub-query as equal voters dilutes the main query
    by 1/3 and can shuffle a correct top-1 down to rank 2 — which measurably
    happened here before weights were introduced.
    """
    rankings = list(rankings)
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[str, float] = {}
    store: dict[str, Hit] = {}
    for weight, ranking in zip(weights, rankings):
        for rank, hit in enumerate(ranking, start=1):
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
            store.setdefault(cid, hit)
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:limit]
    return [
        Hit(chunk=store[cid].chunk, score=score, retriever="hybrid")
        for cid, score in ordered
    ]


class LexicalReranker:
    """CPU-only reranker: fuse retrieval score with lexical/structural overlap."""

    def __init__(self, dense_weight: float = 0.5, lexical_weight: float = 0.35,
                 heading_weight: float = 0.15) -> None:
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.heading_weight = heading_weight

    def rerank(self, query: str, hits: Sequence[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        best = max(hit.score for hit in hits) or 1.0
        out: list[Hit] = []
        for hit in hits:
            lexical = keyword_overlap(query, hit.chunk.text)
            heading = keyword_overlap(query, f"{hit.chunk.title} {hit.chunk.heading}")
            score = (
                self.dense_weight * (hit.score / best)
                + self.lexical_weight * lexical
                + self.heading_weight * heading
            )
            out.append(Hit(chunk=hit.chunk, score=score, retriever="rerank"))
        out.sort(key=lambda h: h.score, reverse=True)
        return out[:k]


class CrossEncoderReranker:
    """Optional upgrade path: BGE-reranker / any sentence-transformers CE model."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        from sentence_transformers import CrossEncoder  # noqa: F401 (optional)

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: Sequence[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        pairs = [(query, hit.chunk.text) for hit in hits]
        scores = self._model.predict(pairs)
        out = [
            Hit(chunk=hit.chunk, score=float(score), retriever="cross-encoder")
            for hit, score in zip(hits, scores)
        ]
        out.sort(key=lambda h: h.score, reverse=True)
        return out[:k]


# ---------------------------------------------------------------------------
# retriever facade
# ---------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)
    dense_candidates: int = 0
    sparse_candidates: int = 0
    queries: list[str] = field(default_factory=list)


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        settings: Settings,
        reranker: object | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.bm25 = BM25Index().build(store.chunks())
        self.reranker = reranker
        if settings.enable_rerank and self.reranker is None:
            self.reranker = LexicalReranker()

    # -- indexing ------------------------------------------------------
    def index_documents(self, kb_dir: Path | None = None) -> dict[str, object]:
        kb_dir = kb_dir or self.settings.kb_dir
        started = time.perf_counter()
        documents = load_documents(kb_dir)
        all_chunks: list[Chunk] = []
        for doc_id, title, text, source in documents:
            all_chunks.extend(
                chunk_document(
                    doc_id,
                    title,
                    text,
                    size=self.settings.chunk_size,
                    overlap=self.settings.chunk_overlap,
                    source_path=source,
                )
            )
        vectors = self.embedder.embed([chunk.text for chunk in all_chunks]) if all_chunks else []
        for chunk, vector in zip(all_chunks, vectors):
            chunk.vector = vector
        self.store.clear()
        self.store.upsert(all_chunks)
        self.bm25.build(self.store.chunks())
        return {
            "documents": len(documents),
            "chunks": len(all_chunks),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "backend": self.store.stats().get("backend", "memory"),
        }

    # -- search --------------------------------------------------------
    def search(self, query: str, k: int | None = None) -> RetrievalResult:
        k = k or self.settings.top_k_final
        query = normalize(query)
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        vector = self.embedder.embed([query])[0]
        dense_hits = self.store.search(vector, self.settings.top_k_dense)
        timings["dense_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        sparse_hits = self.bm25.search(query, self.settings.top_k_sparse)
        timings["sparse_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [dense_hits, sparse_hits], k=self.settings.rrf_k, limit=max(k * 4, 20)
        )
        if self.reranker is not None:
            final = self.reranker.rerank(query, fused, k)  # type: ignore[attr-defined]
        else:
            final = fused[:k]
        timings["fuse_rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return RetrievalResult(
            query=query,
            hits=final,
            timings_ms=timings,
            dense_candidates=len(dense_hits),
            sparse_candidates=len(sparse_hits),
            queries=[query],
        )

    def search_multi(self, queries: Sequence[str], k: int | None = None) -> RetrievalResult:
        """Multi-query retrieval: retrieve per sub-query, fuse, rerank on the original.

        Fusing at the *candidate* level (2N ranked lists into one RRF) rather than
        concatenating results keeps a noisy sub-query from crowding out the main
        one — RRF only rewards a chunk that several queries agree on.
        """
        k = k or self.settings.top_k_final
        cleaned: list[str] = []
        for query in queries:
            normalised = normalize(query)
            if normalised and normalised not in cleaned:
                cleaned.append(normalised)
        if not cleaned:
            return RetrievalResult(query="", hits=[], queries=[])
        if len(cleaned) == 1:
            return self.search(cleaned[0], k=k)

        timings: dict[str, float] = {}
        rankings: list[Sequence[Hit]] = []
        weights: list[float] = []
        dense_count = 0
        sparse_count = 0
        t0 = time.perf_counter()
        for position, query in enumerate(cleaned):
            vector = self.embedder.embed([query])[0]
            dense_hits = self.store.search(vector, self.settings.top_k_dense)
            sparse_hits = self.bm25.search(query, self.settings.top_k_sparse)
            dense_count += len(dense_hits)
            sparse_count += len(sparse_hits)
            rankings.append(dense_hits)
            rankings.append(sparse_hits)
            # The original question is the primary voter; sub-queries widen
            # recall but must not redefine relevance.
            weight = 1.0 if position == 0 else self.settings.sub_query_weight
            weights.extend([weight, weight])
        timings["multi_recall_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(
            rankings, k=self.settings.rrf_k, limit=max(k * 4, 20), weights=weights
        )
        # Always rerank against the user's original question — sub-queries exist
        # to widen recall, not to redefine what "relevant" means.
        if self.reranker is not None:
            final = self.reranker.rerank(cleaned[0], fused, k)  # type: ignore[attr-defined]
        else:
            final = fused[:k]
        timings["fuse_rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        return RetrievalResult(
            query=cleaned[0],
            hits=final,
            timings_ms=timings,
            dense_candidates=dense_count,
            sparse_candidates=sparse_count,
            queries=cleaned,
        )

    def stats(self) -> dict[str, object]:
        return {**self.store.stats(), "bm25_chunks": self.bm25.size}


# ---------------------------------------------------------------------------
# context packing / citations
# ---------------------------------------------------------------------------
CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def pack_context(hits: Sequence[Hit], max_chars: int = 3200) -> str:
    blocks: list[str] = []
    used = 0
    for index, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        header = f"[{index}] 《{chunk.title}》"
        if chunk.heading:
            header += f" · {chunk.heading}"
        body = chunk.text.strip()
        block = f"{header}\n{body}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def hits_to_sources(hits: Sequence[Hit]) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    for index, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        sources.append(
            {
                "index": index,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "heading": chunk.heading,
                "score": round(hit.score, 4),
                "retriever": hit.retriever,
                "preview": chunk.text[:120].replace("\n", " "),
                "text": chunk.text,
            }
        )
    return sources


def citation_report(
    answer: str,
    hits: Sequence[Hit],
    query: str,
    *,
    min_support: float = 0.10,
) -> dict[str, object]:
    """Verify every [n] marker: does index exist, and is the claim supported?

    Support is measured as query-term recall inside the cited chunk, which is a
    cheap, offline proxy for faithfulness. It catches the classic failure mode
    of a model citing [3] for something [3] never says.
    """
    indices = [int(m) for m in CITATION_RE.findall(answer)]
    by_index = {i: hit for i, hit in enumerate(hits, start=1)}
    terms = [t for t in tokenize(query) if len(t) > 1]
    checked: list[dict[str, object]] = []
    supported = 0
    for index in sorted(set(indices)):
        hit = by_index.get(index)
        if hit is None:
            checked.append({"index": index, "valid": False, "support": 0.0})
            continue
        support = coverage(
            terms, f"{hit.chunk.title} {hit.chunk.heading} {hit.chunk.text}"
        )
        ok = support >= min_support
        supported += 1 if ok else 0
        checked.append(
            {
                "index": index,
                "valid": True,
                "supported": ok,
                "support": round(support, 3),
                "title": hit.chunk.title,
            }
        )
    cited = len(set(indices))
    return {
        "cited": cited,
        "existing": cited > 0 and all(c["valid"] for c in checked),
        "supported": supported,
        "support_rate": round(supported / cited, 3) if cited else 0.0,
        "details": checked,
    }
