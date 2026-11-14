"""Text utilities: CJK-aware tokenisation, normalisation, sentence splitting.

Deliberately dependency-free — a Chinese BM25 that needs jieba is a BM25 that
cannot be installed in half the environments this project has to run in. We use
character bigrams for CJK, which is the standard fallback and works well for
retrieval (unigram+bigram recall beats jieba for short queries in practice).
"""

from __future__ import annotations

import re
import unicodedata

_CJK_RANGE = r"\u4e00-\u9fff\u3400-\u4dbf"
_LATIN_WORD = re.compile(r"[a-z0-9][a-z0-9_\-\.]*", re.IGNORECASE)
_CJK_RUN = re.compile(f"[{_CJK_RANGE}]+")
_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;\.])\s*|\n+")


def normalize(text: str) -> str:
    """Full-width -> half-width, lower case, collapsed whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return _WS.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Mixed Chinese/English tokeniser used by BM25 and the hashing embedder."""
    text = normalize(text).lower()
    if not text:
        return []
    tokens: list[str] = []
    for match in _LATIN_WORD.finditer(text):
        tokens.append(match.group(0))
    for run in _CJK_RUN.findall(text):
        tokens.extend(run)  # unigrams
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))  # bigrams
    return tokens


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text or "")]
    return [p for p in parts if p]


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix


def coverage(needle_terms: list[str], haystack: str) -> float:
    """Fraction of query terms present in `haystack` — the citation support check."""
    if not needle_terms:
        return 0.0
    hay_tokens = set(tokenize(haystack))
    hit = sum(1 for term in needle_terms if term in hay_tokens)
    return hit / len(needle_terms)


def keyword_overlap(query: str, text: str) -> float:
    """Length-normalised overlap, used by the lexical reranker."""
    q = set(tokenize(query))
    if not q:
        return 0.0
    t = set(tokenize(text))
    if not t:
        return 0.0
    return len(q & t) / (len(q) ** 0.5 * len(t) ** 0.5)
