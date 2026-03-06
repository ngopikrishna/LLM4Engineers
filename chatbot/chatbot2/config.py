# Default configuration for Chatbot2 (Ollama backend).

MODEL = "llama3"
NUM_PREDICT = 1024     # max tokens to generate (Ollama option)
TEMPERATURE = 0.8
TOP_P = None           # None = use Ollama default
TOP_K = None           # None = use Ollama default
SYSTEM_PROMPT = "You are a helpful assistant."

OLLAMA_HOST = "http://localhost:11434"
HOST = "127.0.0.1"
PORT = 5002
