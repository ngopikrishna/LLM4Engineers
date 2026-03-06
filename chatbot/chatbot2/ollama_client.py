"""Ollama client module for Chatbot2.

Each call is a standalone Q&A — no conversation history is maintained.
Yields text chunks via streaming so the UI can render them incrementally.
"""

import ollama

import config


def stream_response(
    system_prompt: str,
    user_prompt: str,
    model: str,
    num_predict: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
):
    """Stream a single-turn response from Ollama.

    Args:
        system_prompt: Instructions for the model's behaviour.
        user_prompt:   The user's question.
        model:         Ollama model name (e.g. 'llama3').
        num_predict:   Maximum tokens to generate.
        temperature:   Sampling temperature.
        top_p:         Nucleus sampling threshold, or None to use Ollama default.
        top_k:         Top-K sampling, or None to use Ollama default.

    Yields:
        str: Text chunks as they arrive from Ollama.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    options: dict = {"temperature": temperature, "num_predict": num_predict}
    if top_p is not None:
        options["top_p"] = top_p
    if top_k is not None:
        options["top_k"] = top_k

    client = ollama.Client(host=config.OLLAMA_HOST)
    for chunk in client.chat(model=model, messages=messages, stream=True, options=options):
        yield chunk["message"]["content"]
