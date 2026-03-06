"""Flask web application for Chatbot2 (Ollama backend).

Routes:
    GET  /       — Render the chat UI.
    POST /chat   — Accept JSON params, stream Ollama's response as SSE.

Run:
    python app.py
    # or: flask --app app run --port 5002
"""

import json
import ollama
from flask import Flask, render_template, request, Response, stream_with_context

import config
from ollama_client import stream_response

app = Flask(__name__)


def _parse_optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _parse_optional_int(value) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


@app.route("/")
def index():
    return render_template("index.html", cfg=config)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    system_prompt = data.get("system_prompt", config.SYSTEM_PROMPT)
    user_prompt   = data.get("user_prompt", "")
    model         = data.get("model", config.MODEL)
    num_predict   = int(data.get("num_predict", config.NUM_PREDICT))
    temperature   = float(data.get("temperature", config.TEMPERATURE))
    top_p         = _parse_optional_float(data.get("top_p"))
    top_k         = _parse_optional_int(data.get("top_k"))

    def generate():
        try:
            for chunk in stream_response(
                system_prompt, user_prompt, model, num_predict, temperature, top_p, top_k
            ):
                yield f"data: {json.dumps({'text': chunk})}\n\n"

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


if __name__ == "__main__":
    app.run(debug=True, host=config.HOST, port=config.PORT)
