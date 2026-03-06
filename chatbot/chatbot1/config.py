# Default configuration for Chatbot1.
# Claude Sonnet 3.5 is retired; claude-sonnet-4-6 is the current equivalent.

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
TEMPERATURE = 1.0
TOP_P = None   # None = use API default
TOP_K = None   # None = use API default
SYSTEM_PROMPT = "You are a helpful assistant."

HOST = "127.0.0.1"
PORT = 5001
