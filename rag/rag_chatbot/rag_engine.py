"""RAG engine: loads the FAISS index once and handles retrieval + LLM streaming.

The FAISS index and metadata are loaded at import time from the path stored in
app.index_path (set before this module is used).  This avoids reloading the
index on every request.
"""

import pickle

import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

import config

# ── Loaded once at startup ────────────────────────────────────────────────────
_embed_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_metadata: list[dict] | None = None   # [{filename, page, text}, ...]
_index_path: str | None = None


def load_index(index_path: str) -> None:
    """Load the FAISS index and metadata file into memory."""
    global _embed_model, _index, _metadata, _index_path

    _embed_model = SentenceTransformer(config.EMBED_MODEL)
    _index       = faiss.read_index(index_path)
    meta_path    = index_path + ".meta"

    with open(meta_path, "rb") as f:
        _metadata = pickle.load(f)

    _index_path = index_path
    print(f"Index loaded: {index_path} ({_index.ntotal} vectors)")
    print(f"Metadata    : {len(_metadata)} records")


def index_info() -> dict:
    """Return a summary dict shown in the UI header."""
    if _metadata is None:
        return {}
    filenames = sorted({r["filename"] for r in _metadata})
    return {
        "files":   filenames,
        "vectors": _index.ntotal,
    }


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, k: int = config.TOP_K) -> list[dict]:
    """Embed *query* and return the top-k matching metadata records."""
    vec = _embed_model.encode([query]).astype("float32")
    distances, indices = _index.search(vec, k)
    return [_metadata[i] for i in indices[0] if i < len(_metadata)]


# ── Prompt building ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a precise research assistant.  Answer questions using ONLY the numbered
context excerpts provided below.  Follow these rules strictly:

1. After every sentence or claim, add an inline citation in the form
   [filename, page N] that identifies which excerpt supports it.
2. If multiple excerpts support a point, list all relevant citations:
   [filename, page 2][filename, page 7].
3. If the answer cannot be found in the excerpts, respond with exactly:
   "I don't have enough information in the provided documents to answer this."
4. Do not invent or infer facts beyond what the excerpts contain.
"""

def build_context_block(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] ({c['filename']}, page {c['page']}):\n{c['text']}")
    return "\n\n".join(lines)


def build_messages(
    history: list[dict],
    chunks: list[dict],
    user_question: str,
) -> list[dict]:
    """Assemble the full message list: system + context + history + new question."""
    context_block = build_context_block(chunks)
    system_content = SYSTEM_PROMPT + "\n\nContext excerpts:\n\n" + context_block

    messages = [{"role": "system", "content": system_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_question})
    return messages


# ── Streaming response ────────────────────────────────────────────────────────

def stream_answer(messages: list[dict]):
    """Yield text chunks from Ollama."""
    client = ollama.Client(host=config.OLLAMA_HOST)
    for chunk in client.chat(model=config.MODEL, messages=messages, stream=True):
        yield chunk["message"]["content"]
