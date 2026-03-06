"""RAG indexing script.

Reads a PDF, splits each page into overlapping text chunks, generates
embeddings with all-MiniLM-L6-v2, and writes a FAISS index plus a
metadata file that maps every vector back to its source.

Usage:
    python create_index.py path/to/document.pdf
    python create_index.py path/to/document.pdf --out path/to/index.faiss

Outputs (written next to the PDF unless --out is given):
    <name>.faiss          — FAISS flat-L2 index
    <name>.faiss.meta     — pickle: list of {filename, page, text} dicts
"""

import argparse
import os
import pickle
import sys

import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_SIZE    = 1024
CHUNK_OVERLAP = 16
EMBED_MODEL   = "all-MiniLM-L6-v2"


# ── Text extraction ──────────────────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for every page in the PDF.

    Page numbers are 1-based.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping chunks of *size* characters."""
    if not text:
        return []
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Return a float32 array of shape (N, dim)."""
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype("float32")


# ── FAISS storage ─────────────────────────────────────────────────────────────

def build_and_save_index(
    records: list[dict],
    embeddings: np.ndarray,
    index_path: str,
) -> None:
    """Write FAISS index and a matching metadata file to disk.

    Each record contains:
        filename   — base name of the source PDF
        page       — 1-based page number the chunk came from
        text       — original chunk text

    The i-th record corresponds to the i-th vector in the FAISS index.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
    faiss.write_index(index, index_path)

    meta_path = index_path + ".meta"
    with open(meta_path, "wb") as f:
        pickle.dump(records, f)

    print(f"  FAISS index : {index_path}  ({index.ntotal} vectors, dim={dimension})")
    print(f"  Metadata    : {meta_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def index_pdf(pdf_path: str, index_path: str) -> None:
    filename = os.path.basename(pdf_path)

    print(f"[1/4] Reading PDF: {pdf_path}")
    pages = extract_pages(pdf_path)
    print(f"      {len(pages)} page(s) with text")

    print(f"[2/4] Chunking  (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    records: list[dict] = []
    for page_num, page_text in pages:
        for chunk in chunk_text(page_text):
            records.append({
                "filename": filename,
                "page":     page_num,
                "text":     chunk,
            })
    print(f"      {len(records)} chunk(s) produced")

    if not records:
        print("No text could be extracted. Aborting.")
        sys.exit(1)

    print(f"[3/4] Embedding with '{EMBED_MODEL}'")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = embed([r["text"] for r in records], model)

    print(f"[4/4] Writing FAISS index")
    build_and_save_index(records, embeddings, index_path)

    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from a PDF for use in a RAG pipeline."
    )
    parser.add_argument("pdf", help="Path to the source PDF file")
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for the FAISS index file (default: <pdf_name>.faiss)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.pdf):
        print(f"Error: file not found: {args.pdf}")
        sys.exit(1)
    if not args.pdf.lower().endswith(".pdf"):
        print(f"Error: expected a .pdf file, got: {args.pdf}")
        sys.exit(1)

    if args.out:
        index_path = args.out
    else:
        base = os.path.splitext(args.pdf)[0]
        index_path = base + ".faiss"

    index_pdf(args.pdf, index_path)
