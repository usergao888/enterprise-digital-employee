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
    """Length-normalised overlap over *content* terms, used for reranking.

    Single CJK characters are excluded on purpose. Matching on unigrams lets a
    short, irrelevant sentence win: "…拨打分机 8000 催单" outscored the real
    answer to "电脑开不了机怎么办" purely because both contain 机. Bigrams and
    latin words carry the actual meaning.
    """
    q = _content_terms(query)
    if not q:
        return 0.0
    t = _content_terms(text)
    if not t:
        return 0.0
    return len(q & t) / (len(q) ** 0.5 * len(t) ** 0.5)


def _content_terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if not (len(t) == 1 and _is_cjk(t))}


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff" or "\u3400" <= char <= "\u4dbf"


# ---------------------------------------------------------------------------
# query decomposition
# ---------------------------------------------------------------------------
_CLAUSE_SEP = re.compile(r"[，,；;。!！?？\n]+|(?:并且|同时|另外|以及|还有|顺便|然后)")
_LEADING = re.compile(r"^(?:请问|麻烦|帮我|我想知道|那么|那|另外|并且|同时|还有|以及|顺便)")
_TRAILING = re.compile(r"[？?！!。，,、\s]+$")


def decompose_question(question: str, max_parts: int = 3) -> list[str]:
    """Split a compound question into independently retrievable sub-queries.

    Rule-based on purpose: it is deterministic, needs no model, and acts as the
    fallback when the planner does not emit `sub_queries`. It only splits on
    clause boundaries that carry a question, so a single-clause question returns
    an empty list and the caller falls back to normal single-query retrieval.
    """
    text = normalize(question).strip()
    if not text:
        return []
    raw_parts = _CLAUSE_SEP.split(text)
    parts: list[str] = []
    for part in raw_parts:
        cleaned = _TRAILING.sub("", _LEADING.sub("", part.strip()))
        if len(cleaned) < 4:
            continue
        if cleaned == _TRAILING.sub("", text):
            continue
        if cleaned not in parts:
            parts.append(cleaned)
    if len(parts) < 2:
        return []
    return parts[: max(0, max_parts)]

