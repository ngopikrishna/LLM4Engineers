"""Flask web application for Chatbot3 (Ollama backend, persistent history).

Routes:
    GET  /        — Render the chat UI.
    POST /chat    — Append user message, stream Ollama response as SSE,
                    then save assistant reply to history.
    POST /clear   — Reset conversation history.

Run:
    python app.py
    # or: flask --app app run --port 5003
"""

import json
import ollama
from flask import Flask, render_template, request, Response, stream_with_context

import config
from ollama_client import stream_response

app = Flask(__name__)

# In-memory conversation history shared across requests.
# Each entry: {"role": "user"|"assistant"|"system", "content": str}
history: list[dict] = []


def _parse_optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _parse_optional_int(value) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _build_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    """Prepend system prompt to history and append the new user message."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    return messages


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

    messages = _build_messages(system_prompt, user_prompt)

    # Accumulate the full assistant reply so we can save it to history.
    assistant_reply: list[str] = []

    def generate():
        try:
            for chunk in stream_response(messages, model, num_predict, temperature, top_p, top_k):
                assistant_reply.append(chunk)
                yield f"data: {json.dumps({'text': chunk})}\n\n"

            # Persist the completed exchange to history.
            history.append({"role": "user",      "content": user_prompt})
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


if __name__ == "__main__":
    app.run(debug=True, host=config.HOST, port=config.PORT)
