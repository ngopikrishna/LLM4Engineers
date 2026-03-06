"""Claude API client module.

Each call is a standalone Q&A — no conversation history is maintained.
Yields text chunks via streaming so the UI can render them incrementally.
"""

import anthropic


def stream_response(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
):
    """Stream a single-turn response from Claude.

    Args:
        system_prompt: Instructions for Claude's behaviour.
        user_prompt:   The user's question.
        model:         Claude model ID.
        max_tokens:    Maximum tokens to generate.
        temperature:   Sampling temperature (0–1).
        top_p:         Nucleus sampling threshold, or None to use API default.
        top_k:         Top-K sampling, or None to use API default.

    Yields:
        str: Text chunks as they arrive from the API.
    """
    client = anthropic.Anthropic()

    params: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    if system_prompt:
        params["system"] = system_prompt
    if top_p is not None:
        params["top_p"] = top_p
    if top_k is not None:
        params["top_k"] = top_k

    with client.messages.stream(**params) as stream:
        for text in stream.text_stream:
            yield text
