"""Flask data-agent chatbot.

Loads an NSE stock Excel workbook at startup, builds a LlamaIndex ReActAgent
with pandas-backed FunctionTools, and serves a chat UI.  For every question
the agent selects the right tool(s), queries the data, and synthesises an answer.

Run:
    python app.py --data "path/to/NSE Data YTD.xlsx"
    python app.py --data "path/to/NSE Data YTD.xlsx" --port 5012
"""

import argparse
import json
import sys

from flask import Flask, Response, render_template, request, stream_with_context

import config
import data_agent_engine

app = Flask(__name__)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", cfg=config, info=data_agent_engine.data_info())


@app.route("/chat", methods=["POST"])
def chat():
    data          = request.get_json()
    user_question = (data.get("user_prompt") or "").strip()

    if not user_question:
        return {"error": "Empty question"}, 400

    def generate():
        try:
            for event_type, payload in data_agent_engine.ask_stream(user_question):
                if event_type == "status":
                    yield f"data: {json.dumps({'status': payload})}\n\n"
                elif event_type == "tool_results":
                    yield f"data: {json.dumps({'tool_results': payload})}\n\n"
                elif event_type == "text":
                    yield f"data: {json.dumps({'text': payload})}\n\n"
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
    data_agent_engine.reset()
    return "", 204


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="NSE data agent chatbot")
    parser.add_argument(
        "--data", required=True,
        help="Path to the NSE Excel file (e.g. 'Source Material/data/NSE Data YTD.xlsx')",
    )
    parser.add_argument("--port", type=int, default=config.PORT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        data_agent_engine.load_data(args.data)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    app.run(debug=False, host=config.HOST, port=args.port)
