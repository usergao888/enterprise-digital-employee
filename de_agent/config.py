"""Runtime configuration for the Enterprise Digital Employee agent.

Everything is driven from environment variables so the same code runs against
DeepSeek / DashScope(Qwen) / vLLM / Ollama (all OpenAI-compatible) or fully
offline with the deterministic fake backend used by the test-suite.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KB_DIR = PROJECT_ROOT / "data" / "kb"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "var" / "index"

# Illustrative CNY per 1k tokens (public list prices, rounded). Used only for
# the routing cost report — never for billing.
_KNOWN_PRICES: dict[str, tuple[float, float]] = {
    "deepseek-chat": (0.001, 0.002),
    "deepseek-reasoner": (0.004, 0.016),
    "qwen-plus": (0.0008, 0.002),
    "qwen-max": (0.0024, 0.0096),
    "glm-4-plus": (0.005, 0.005),
    "gpt-4o-mini": (0.0011, 0.0044),
}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class ModelSpec:
    """One concrete model endpoint used by the multi-model router."""

    name: str
    label: str
    input_price_per_1k: float = 0.0   # CNY, used only for the cost report
    output_price_per_1k: float = 0.0


@dataclass
class Settings:
    # ---- LLM -------------------------------------------------------------
    llm_backend: str = "fake"            # fake | openai_compat
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-reasoner"
    llm_temperature: float = 0.2
    llm_timeout_s: int = 60
    llm_max_tokens: int = 1024

    # Cheap model used for the executor stage, strong one for plan/review.
    small_model: str = "deepseek-chat"
    router_enabled: bool = True
    router_long_question_chars: int = 120

    # ---- Embeddings ------------------------------------------------------
    embed_backend: str = "hashing"       # hashing | openai_compat
    embed_model: str = "text-embedding-v3"
    embed_dim: int = 512

    # ---- Retrieval -------------------------------------------------------
    chunk_size: int = 520
    chunk_overlap: int = 90
    top_k_dense: int = 12
    top_k_sparse: int = 12
    top_k_final: int = 5
    rrf_k: int = 60
    enable_rerank: bool = True
    min_citation_support: float = 0.30
    # Compound questions ("年假有几天，怎么申请") can be split into sub-queries and
    # fused with weighted RRF. Measured on the bundled 8-doc corpus it did not
    # help (see docs/eval_report.md §3), so it ships OFF and stays available for
    # larger corpora: DE_ENABLE_DECOMPOSITION=true.
    enable_decomposition: bool = False
    max_sub_queries: int = 3
    # Weight of each sub-query relative to the original question in RRF fusion.
    sub_query_weight: float = 0.5

    # ---- Agent graph -----------------------------------------------------
    max_iterations: int = 6              # tool-calling iterations inside executor
    max_revisions: int = 1               # reviewer -> revise -> re-execute rounds
    recursion_limit: int = 40
    enable_reviewer: bool = True
    # Park mutating tool calls for human confirmation instead of executing them.
    require_confirmation: bool = False

    # ---- Paths -----------------------------------------------------------
    kb_dir: Path = field(default_factory=lambda: DEFAULT_KB_DIR)
    index_dir: Path = field(default_factory=lambda: DEFAULT_INDEX_DIR)

    # ---- Server ----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8710

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            llm_backend=_env("DE_LLM_BACKEND", "fake"),
            llm_base_url=_env("DE_LLM_BASE_URL", "https://api.deepseek.com/v1"),
            llm_api_key=_env("DE_LLM_API_KEY", ""),
            llm_model=_env("DE_LLM_MODEL", "deepseek-reasoner"),
            llm_temperature=_env_float("DE_LLM_TEMPERATURE", 0.2),
            llm_timeout_s=_env_int("DE_LLM_TIMEOUT_S", 60),
            llm_max_tokens=_env_int("DE_LLM_MAX_TOKENS", 1024),
            small_model=_env("DE_SMALL_MODEL", "deepseek-chat"),
            router_enabled=_env_bool("DE_ROUTER_ENABLED", True),
            router_long_question_chars=_env_int("DE_ROUTER_LONG_QUESTION_CHARS", 120),
            embed_backend=_env("DE_EMBED_BACKEND", "hashing"),
            embed_model=_env("DE_EMBED_MODEL", "text-embedding-v3"),
            embed_dim=_env_int("DE_EMBED_DIM", 512),
            chunk_size=_env_int("DE_CHUNK_SIZE", 520),
            chunk_overlap=_env_int("DE_CHUNK_OVERLAP", 90),
            top_k_dense=_env_int("DE_TOP_K_DENSE", 12),
            top_k_sparse=_env_int("DE_TOP_K_SPARSE", 12),
            top_k_final=_env_int("DE_TOP_K_FINAL", 5),
            rrf_k=_env_int("DE_RRF_K", 60),
            enable_rerank=_env_bool("DE_ENABLE_RERANK", True),
            min_citation_support=_env_float("DE_MIN_CITATION_SUPPORT", 0.30),
            enable_decomposition=_env_bool("DE_ENABLE_DECOMPOSITION", False),
            max_sub_queries=_env_int("DE_MAX_SUB_QUERIES", 3),
            sub_query_weight=_env_float("DE_SUB_QUERY_WEIGHT", 0.5),
            max_iterations=_env_int("DE_MAX_ITERATIONS", 6),
            max_revisions=_env_int("DE_MAX_REVISIONS", 1),
            recursion_limit=_env_int("DE_RECURSION_LIMIT", 40),
            enable_reviewer=_env_bool("DE_ENABLE_REVIEWER", True),
            require_confirmation=_env_bool("DE_REQUIRE_CONFIRMATION", False),
            kb_dir=Path(_env("DE_KB_DIR", str(DEFAULT_KB_DIR))).resolve(),
            index_dir=Path(_env("DE_INDEX_DIR", str(DEFAULT_INDEX_DIR))).resolve(),
            host=_env("DE_HOST", "127.0.0.1"),
            port=_env_int("DE_PORT", 8710),
        )
        if settings.llm_backend == "openai_compat" and not settings.llm_api_key:
            raise ValueError(
                "DE_LLM_BACKEND=openai_compat 需要设置 DE_LLM_API_KEY"
            )
        return settings

    def model_catalog(self) -> dict[str, ModelSpec]:
        """Prices are illustrative CNY/1k tokens, used only for the cost report.

        The two models are deliberately different tiers: plan/review get the
        reasoning model, execute gets the cheap chat model. If you point both
        at the same model the router simply stops saving anything, which the
        report will show honestly.
        """
        prices = dict(_KNOWN_PRICES)
        default = (0.001, 0.002)
        spec_in, spec_out = prices.get(self.llm_model, default)
        small_in, small_out = prices.get(self.small_model, default)
        return {
            self.llm_model: ModelSpec(
                name=self.llm_model,
                label=f"{self.llm_model}(strong)",
                input_price_per_1k=spec_in,
                output_price_per_1k=spec_out,
            ),
            self.small_model: ModelSpec(
                name=self.small_model,
                label=f"{self.small_model}(small)",
                input_price_per_1k=small_in,
                output_price_per_1k=small_out,
            ),
        }
