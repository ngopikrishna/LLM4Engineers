"""RAG agent engine: wraps an existing FAISS index in a LlamaIndex ReActAgent.

The FAISS index and metadata (built by rag/create_index/create_index.py) are
loaded once at startup.  A custom LlamaIndex BaseRetriever bridges the existing
index format into LlamaIndex's retrieval pipeline, which is then exposed to a
ReActAgent (text-based ReAct loop, no tool-call API required) via AgentWorkflow.

Public API
----------
load_index(index_path)  — call once at startup; populates _workflow
index_info()            — returns a summary dict for the UI header
ask(question)           — synchronous agent call; returns (answer, sources)
reset()                 — clear the conversation history
"""

import asyncio
import pickle

import faiss
from llama_index.core import Settings
from llama_index.core.agent import AgentWorkflow, ReActAgent
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.ollama import Ollama
from sentence_transformers import SentenceTransformer

import config

# ── Module-level singletons ───────────────────────────────────────────────────

_workflow: AgentWorkflow | None = None
_retriever: "_FaissRetriever | None" = None
_index_meta: dict = {}
_chat_history: list[ChatMessage] = []


# ── Custom FAISS retriever ────────────────────────────────────────────────────

class _FaissRetriever(BaseRetriever):
    """Retrieves chunks from the existing FAISS index using SentenceTransformer.

    Also caches the last retrieved nodes so the Flask route can read them for
    the sources panel without a second search.
    """

    def __init__(
        self,
        index: faiss.Index,
        metadata: list[dict],
        embed_model: SentenceTransformer,
        top_k: int,
    ) -> None:
        self._index       = index
        self._metadata    = metadata
        self._embed_model = embed_model
        self._top_k       = top_k
        self.last_nodes: list[NodeWithScore] = []
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        vec = self._embed_model.encode([query_bundle.query_str]).astype("float32")
        _, indices = self._index.search(vec, self._top_k)

        nodes: list[NodeWithScore] = []
        for i in indices[0]:
            if i < 0 or i >= len(self._metadata):
                continue
            rec  = self._metadata[i]
            node = TextNode(
                text=rec["text"],
                metadata={
                    "file_name":  rec["filename"],
                    "page_label": str(rec["page"]),
                },
            )
            nodes.append(NodeWithScore(node=node, score=1.0))

        self.last_nodes = nodes
        return nodes


# ── Initialisation ────────────────────────────────────────────────────────────

def load_index(index_path: str) -> None:
    """Load the FAISS index and metadata, then build the ReActAgent workflow."""
    global _workflow, _retriever, _index_meta

    # Load the FAISS index and metadata pickle.
    print(f"[1/3] Loading FAISS index: {index_path}")
    faiss_index = faiss.read_index(index_path)
    meta_path   = index_path + ".meta"
    with open(meta_path, "rb") as f:
        metadata: list[dict] = pickle.load(f)
    print(f"      {faiss_index.ntotal} vectors, {len(metadata)} metadata records")

    # Embedding model for query-time retrieval (same model used at index build).
    embed_model = SentenceTransformer(config.EMBED_MODEL)

    # LLM for the agent reasoning loop.
    print(f"[2/3] Configuring LLM ({config.MODEL} via Ollama)")
    llm = Ollama(
        model=config.MODEL,
        base_url=config.OLLAMA_HOST,
        request_timeout=180.0,
    )
    # Disable LlamaIndex's own embedding pipeline — we use SentenceTransformer.
    Settings.embed_model = None

    # Build: retriever → query engine → QueryEngineTool → ReActAgent → workflow.
    print(f"[3/3] Building ReActAgent workflow")
    _retriever   = _FaissRetriever(faiss_index, metadata, embed_model, config.TOP_K)
    query_engine = RetrieverQueryEngine.from_args(_retriever, llm=llm)

    filenames = sorted({r["filename"] for r in metadata})
    tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="document_search",
            description=(
                f"Search the indexed document(s) ({', '.join(filenames)}) for relevant "
                "information. Always use this tool to look up facts before answering. "
                "Input should be a concise search query."
            ),
        ),
    )

    react_agent = ReActAgent(
        name="ReActAgent",
        tools=[tool],
        llm=llm,
    )

    _workflow = AgentWorkflow(agents=[react_agent], root_agent="ReActAgent")

    _index_meta = {
        "files":   filenames,
        "vectors": faiss_index.ntotal,
    }
    print(f"Agent ready.  Model: {config.MODEL}")


# ── Public helpers ────────────────────────────────────────────────────────────

def index_info() -> dict:
    return _index_meta


def ask(question: str) -> tuple[str, list[dict]]:
    """Run the agent and return (answer_text, list_of_source_dicts).

    Source dicts have keys: filename, page, text.
    """
    if _workflow is None:
        raise RuntimeError("Agent not initialised — call load_index() first.")

    async def _run() -> str:
        handler = _workflow.run(
            user_msg=question,
            chat_history=list(_chat_history),
        )
        result = await handler
        # Persist turns for multi-turn conversation.
        _chat_history.append(ChatMessage(role="user",      content=question))
        _chat_history.append(ChatMessage(role="assistant", content=result.response.content))
        return result.response.content or ""

    answer = asyncio.run(_run())

    # Read the nodes cached by the retriever during the agent's tool call.
    sources: list[dict] = []
    if _retriever is not None:
        for nws in _retriever.last_nodes:
            meta = nws.node.metadata or {}
            sources.append({
                "filename": meta.get("file_name", "unknown"),
                "page":     meta.get("page_label", "?"),
                "text":     nws.node.get_content()[:600],
            })

    return answer, sources


def reset() -> None:
    """Clear the conversation history."""
    _chat_history.clear()
    if _retriever is not None:
        _retriever.last_nodes = []
