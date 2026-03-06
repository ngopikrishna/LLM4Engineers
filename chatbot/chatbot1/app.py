"""Flask web application for Chatbot1.

Routes:
    GET  /       — Render the chat UI.
    POST /chat   — Accept JSON params, stream Claude's response as SSE.

Run:
    python app.py
    # or: flask --app app run --port 5001
"""

import json
import anthropic
from flask import Flask, render_template, request, Response, stream_with_context

import config
from claude_client import stream_response

app = Flask(__name__)


def _parse_optional_float(value) -> float | None:
    """Return float if value is a non-empty string/number, else None."""
    if value in (None, ""):
        return None
    return float(value)


def _parse_optional_int(value) -> int | None:
    """Return int if value is a non-empty string/number, else None."""
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
    max_tokens    = int(data.get("max_tokens", config.MAX_TOKENS))
    temperature   = float(data.get("temperature", config.TEMPERATURE))
    top_p         = _parse_optional_float(data.get("top_p"))
    top_k         = _parse_optional_int(data.get("top_k"))

    def generate():
        try:
            for chunk in stream_response(
                system_prompt, user_prompt, model, max_tokens, temperature, top_p, top_k
            ):
                yield f"data: {json.dumps({'text': chunk})}\n\n"

        except anthropic.BadRequestError as e:
            yield f"data: {json.dumps({'error': f'Bad request: {e.message}'})}\n\n"
        except anthropic.AuthenticationError:
            yield f"data: {json.dumps({'error': 'Authentication failed. Check ANTHROPIC_API_KEY.'})}\n\n"
        except anthropic.RateLimitError:
            yield f"data: {json.dumps({'error': 'Rate limit reached. Please wait and retry.'})}\n\n"
        except anthropic.APIStatusError as e:
            yield f"data: {json.dumps({'error': f'API error {e.status_code}: {e.message}'})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(debug=True, host=config.HOST, port=config.PORT)
