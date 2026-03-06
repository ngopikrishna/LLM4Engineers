"""Flask RAG-agent chatbot.

Loads a pre-built FAISS index at startup (produced by rag/create_index/create_index.py),
wraps it in a LlamaIndex ReActAgent, and serves a chat UI.  For every user
question the agent:
  1. Decides to call the document_search QueryEngineTool.
  2. Retrieves the top-k most relevant chunks from the FAISS index.
  3. Synthesises an answer and returns it with source attribution.

The answer is streamed word-by-word to the browser via Server-Sent Events so
the UI experience matches the rag_chatbot.

Run:
    python app.py --index path/to/file.faiss
    python app.py --index path/to/file.faiss --port 5011
"""

import argparse
import json
import sys

from flask import Flask, Response, render_template, request, stream_with_context

import config
import rag_agent_engine

app = Flask(__name__)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", cfg=config, info=rag_agent_engine.index_info())


@app.route("/chat", methods=["POST"])
def chat():
    data          = request.get_json()
    user_question = (data.get("user_prompt") or "").strip()

    if not user_question:
        return {"error": "Empty question"}, 400

    def generate():
        try:
            answer, sources = rag_agent_engine.ask(user_question)

            # Send retrieved sources first so the UI panel updates immediately.
            yield f"data: {json.dumps({'sources': sources})}\n\n"

            # Stream the answer word-by-word for a live-typing effect.
            words = answer.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                yield f"data: {json.dumps({'text': token})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/clear", methods=["POST"])
def clear():
    rag_agent_engine.reset()
    return "", 204


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RAG agent chatbot Flask server")
    parser.add_argument(
        "--index", required=True,
        help="Path to the .faiss index file produced by create_index.py",
    )
    parser.add_argument("--port", type=int, default=config.PORT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        rag_agent_engine.load_index(args.index)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    app.run(debug=False, host=config.HOST, port=args.port)
