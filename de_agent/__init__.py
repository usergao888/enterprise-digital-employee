"""Enterprise Digital Employee — a Python AI‑Agent reference implementation.

Layers (bottom-up):

    textutil / embeddings / vectorstore   retrieval primitives
    rag                                   chunking + BM25 + RRF + rerank
    tools                                 JSON-Schema tool calling
    graph                                 LangGraph-shaped state machine
    agents                                planner / executor / reviewer
    evaluation                            offline metrics
"""

from .agents import DigitalEmployee, build_digital_employee, guard_input
from .config import Settings
from .graph import END, START, GraphError, StateGraph
from .llm import FakeLLM, Message, MeteredRouter, OpenAICompatClient, ToolCall
from .rag import BM25Index, HybridRetriever, chunk_document, load_documents
from .tools import Tool, ToolRegistry, build_registry, safe_eval, validate_arguments
from .vectorstore import Chunk, Hit, MemoryVectorStore

__version__ = "0.3.1"

__all__ = [
    "DigitalEmployee",
    "build_digital_employee",
    "guard_input",
    "Settings",
    "StateGraph",
    "START",
    "END",
    "GraphError",
    "FakeLLM",
    "Message",
    "ToolCall",
    "MeteredRouter",
    "OpenAICompatClient",
    "BM25Index",
    "HybridRetriever",
    "chunk_document",
    "load_documents",
    "Tool",
    "ToolRegistry",
    "build_registry",
    "safe_eval",
    "validate_arguments",
    "Chunk",
    "Hit",
    "MemoryVectorStore",
    "__version__",
]
