# LLMs for Engineers — Companion Code

Companion code repository for the book **"LLMs for Engineers"**. Each chapter builds on the previous one, progressing from foundational ML concepts through chatbots, RAG systems, LlamaIndex agents, and MCP (Model Context Protocol) integrations.

---

## Folder Structure

```
LLM4Engineers/
├── basics/                     # Pure NumPy ML implementations
├── chatbot/                    # LangChain + Ollama / Claude chatbots
│   ├── sample_ollama_chat.py
│   ├── sample_ollama_chat2.py
│   ├── chatbot1/               # Flask chatbot — Claude API backend
│   ├── chatbot2/               # Flask chatbot — Ollama backend (stateless)
│   └── chatbot3/               # Flask chatbot — Ollama backend (persistent history)
├── rag/                        # Retrieval-Augmented Generation
│   ├── create_index/           # FAISS index builder from PDF
│   ├── rag_chatbot/            # RAG chatbot (manual FAISS retrieval)
│   ├── ragagent_chatbot/       # RAG chatbot backed by LlamaIndex ReActAgent
│   └── dataagent_chatbot/      # LlamaIndex agent over NSE stock Excel data
├── mcp/                        # Model Context Protocol examples
│   ├── llm_check.py            # AWS Bedrock connectivity check
│   ├── agent/                  # LlamaIndex FunctionAgent consuming MCP tools
│   ├── mcpclient/              # Standalone MCP client
│   ├── nseserver/              # FastMCP server exposing NSE stock data
│   └── email_server/           # FastMCP server exposing email sending
├── Source Material/            # Source PDFs and datasets
│   └── data/
│       └── NSE Data YTD.xlsx   # NSE daily trading data (YTD)
├── pyproject.toml              # Project dependencies (uv)
└── uv.lock                     # Locked dependency versions
```

---

## Folder Summaries

### `basics/`
Pure Python / NumPy implementations of foundational ML concepts — no ML frameworks.

| File | What it demonstrates |
|---|---|
| `perceptron_tips.py` | Single perceptron binary classifier on the Tips dataset; gradient descent from scratch; logs training to `training_log.csv` |
| `cost_surface.py` | Visualises the MSE cost surface and gradient-descent trajectory over (weight, bias) space |
| `encoder_decoder.py` | Character-level autoencoder using only NumPy; backpropagation and Adam optimiser implemented manually |

---

### `chatbot/`
Minimal chatbot examples using LangChain, Ollama, and the Anthropic Claude API.

| Path | What it demonstrates |
|---|---|
| `sample_ollama_chat.py` | One-shot Ollama chat via LangChain `ChatPromptTemplate` |
| `sample_ollama_chat2.py` | Variant with a different prompt structure |
| `chatbot1/` | Flask web chatbot streaming responses from the **Claude API** (Anthropic) |
| `chatbot2/` | Flask web chatbot streaming responses from a local **Ollama** model (stateless) |
| `chatbot3/` | Same as chatbot2 but with **persistent conversation history** across turns |

All three web chatbots share the same SSE streaming pattern and are started via `start.sh`.

---

### `rag/`
Retrieval-Augmented Generation pipeline. **Build order matters** — create the FAISS index before running any chatbot.

#### `create_index/`
Reads a PDF, chunks each page, embeds with `all-MiniLM-L6-v2`, and writes a FAISS flat-L2 index plus a metadata pickle.

```
Output: <name>.faiss  +  <name>.faiss.meta
```

#### `rag_chatbot/`
Flask chatbot with manual RAG: query → SentenceTransformer embedding → FAISS top-k search → inject chunks into system prompt → stream Ollama answer with inline citations. Retrieved source chunks are shown in a side panel.

#### `ragagent_chatbot/`
Same user-facing interface as `rag_chatbot` but the retrieval is handled by a **LlamaIndex `ReActAgent`** with a `QueryEngineTool`. A custom `BaseRetriever` bridges the existing FAISS index format into LlamaIndex's pipeline.

#### `dataagent_chatbot/`
LlamaIndex `ReActAgent` over **NSE stock Excel data** (`NSE Data YTD.xlsx`) — no vector store. Eight pandas-backed `FunctionTool`s cover price look-up, range history, summary stats, cross-stock comparison, top gainers/losers, return calculation, and volume ranking. The UI shows live step-by-step progress (tool calls, synthesis) as the agent works.

---

### `mcp/`
Model Context Protocol (MCP) examples using FastMCP and LlamaIndex.

| Path | What it demonstrates |
|---|---|
| `llm_check.py` | Verifies AWS Bedrock connectivity with Amazon Nova Pro |
| `nseserver/` | FastMCP server that exposes NSE stock data as MCP tools |
| `email_server/` | FastMCP server that wraps SMTP email sending as an MCP tool |
| `mcpclient/` | Standalone FastMCP client that connects to an MCP server |
| `agent/` | LlamaIndex `FunctionAgent` backed by AWS Bedrock (Claude/Nova) that discovers and calls tools from a running MCP server |

---

## Environment Setup

This project uses [**uv**](https://docs.astral.sh/uv/) for environment and dependency management.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create the virtual environment and install all dependencies

```bash
cd LLM4Engineers
uv sync
```

This reads `pyproject.toml` and `uv.lock`, creates `.venv/`, and installs everything.

### 3. Activate the environment (optional)

```bash
source .venv/bin/activate
```

Or prefix any command with `uv run` to use the project environment without activating:

```bash
uv run python basics/perceptron_tips.py
```

---

## Running the Code

### Prerequisites

**Ollama** (required for all chatbot and RAG scripts):

```bash
ollama serve
ollama pull llama3    # used by chatbots and RAG agents
ollama pull phi4      # used by some scripts
```

**AWS credentials** (required for `mcp/` Bedrock scripts only):

```bash
aws configure                          # static IAM credentials
# or
aws sso login --profile <your-profile>
export AWS_PROFILE=<your-profile>
```

---

### Basics

```bash
python basics/perceptron_tips.py
python basics/cost_surface.py
python basics/encoder_decoder.py
```

---

### Chatbots

```bash
# Quick terminal chatbot (no server)
python chatbot/sample_ollama_chat.py

# Web chatbots — open the printed URL in a browser after starting
cd chatbot/chatbot1 && ./start.sh    # Claude API (set ANTHROPIC_API_KEY first)
cd chatbot/chatbot2 && ./start.sh    # Ollama, stateless
cd chatbot/chatbot3 && ./start.sh    # Ollama, persistent history
```

---

### RAG Pipeline

**Step 1 — Build the FAISS index** (run once per PDF, output sits next to the PDF):

```bash
python rag/create_index/create_index.py "Source Material/TCS - annual-report-2023-2024.pdf"
# Produces:
#   Source Material/TCS - annual-report-2023-2024.faiss
#   Source Material/TCS - annual-report-2023-2024.faiss.meta
```

**Step 2 — Start a chatbot** (open the browser at the printed URL):

```bash
# Manual RAG chatbot (port 5010)
cd rag/rag_chatbot
python app.py --index "../../Source Material/TCS - annual-report-2023-2024.faiss"

# LlamaIndex ReActAgent RAG chatbot (port 5011)
cd rag/ragagent_chatbot
python app.py --index "../../Source Material/TCS - annual-report-2023-2024.faiss"

# NSE stock data agent — no index needed (port 5012)
cd rag/dataagent_chatbot
python app.py --data "../../Source Material/data/NSE Data YTD.xlsx"
```

Or use the convenience scripts from the respective folders:

```bash
cd rag/rag_chatbot       && ./start.sh "../../Source Material/TCS - annual-report-2023-2024.faiss"
cd rag/ragagent_chatbot  && ./start.sh "../../Source Material/TCS - annual-report-2023-2024.faiss"
cd rag/dataagent_chatbot && ./start.sh "../../Source Material/data/NSE Data YTD.xlsx"
```

---

### MCP

```bash
# Verify AWS Bedrock access
python mcp/llm_check.py

# Start an MCP server (terminal 1)
python mcp/nseserver/nse_server1.py

# Run the client or agentic chatbot (terminal 2)
python mcp/mcpclient/mcpclient.py
python mcp/agent/agentic_chatbot.py
```

---

## Port Reference

| App | Default port |
|---|---|
| `chatbot1` | 5001 |
| `chatbot2` | 5002 |
| `chatbot3` | 5003 |
| `rag_chatbot` | 5010 |
| `ragagent_chatbot` | 5011 |
| `dataagent_chatbot` | 5012 |

All web apps accept `--port <number>` to override the default.
