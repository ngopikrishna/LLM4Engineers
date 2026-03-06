"""Flask RAG chatbot.

Loads a FAISS index (built by rag/create_index.py) at startup and serves a
chat UI.  For every user question it:
  1. Retrieves the top-k most relevant chunks via vector search.
  2. Injects them as numbered context into the system prompt.
  3. Streams the LLM's answer (with citations) back to the browser.

Run:
    python app.py --index path/to/file.faiss
    python app.py --index path/to/file.faiss --port 5010
"""

import argparse
import json
import sys

import ollama
from flask import Flask, Response, render_template, request, stream_with_context

import config
import rag_engine

app = Flask(__name__)

# In-memory conversation history (user + assistant turns only; system is
# rebuilt on every request so the context is always fresh).
history: list[dict] = []


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", cfg=config, info=rag_engine.index_info())


@app.route("/chat", methods=["POST"])
def chat():
    data          = request.get_json()
    user_question = (data.get("user_prompt") or "").strip()

    if not user_question:
        return {"error": "Empty question"}, 400

    # Retrieve relevant chunks for this question.
    chunks = rag_engine.retrieve(user_question)

    # Build full message list (system + history + new question).
    messages = rag_engine.build_messages(history, chunks, user_question)

    # Accumulate the assistant reply so we can save it to history.
    assistant_reply: list[str] = []

    def generate():
        # Send the retrieved sources to the UI first.
        sources = [{"filename": c["filename"], "page": c["page"], "text": c["text"]}
                   for c in chunks]
        yield f"data: {json.dumps({'sources': sources})}\n\n"

        try:
            for chunk_text in rag_engine.stream_answer(messages):
                assistant_reply.append(chunk_text)
                yield f"data: {json.dumps({'text': chunk_text})}\n\n"

            history.append({"role": "user",      "content": user_question})
            history.append({"role": "assistant", "content": "".join(assistant_reply)})

        except ollama.ResponseError as e:
            yield f"data: {json.dumps({'error': f'Ollama error: {e.error}'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': f'Unexpected error: {e}'})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/clear", methods=["POST"])
def clear():
    history.clear()
    return "", 204


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RAG chatbot Flask server")
    parser.add_argument(
        "--index", required=True,
        help="Path to the .faiss index file produced by create_index.py",
    )
    parser.add_argument("--port", type=int, default=config.PORT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        rag_engine.load_index(args.index)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    app.run(debug=False, host=config.HOST, port=args.port)
