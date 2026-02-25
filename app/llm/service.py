"""Prompt-to-payload adapter for LLM invocation.

Architectural role:
    Provides the canonical text-generation entrypoint used by orchestration and NLP
    layers. This module bridges prompt construction (core/prompting) to transport
    (`app.llm.client`).

Model call flow:
    prompt -> payload construction -> `client.send_request(...)`.

Token behavior:
    No explicit token-budget enforcement is implemented here. Token limits are managed
    upstream (if any) or by provider defaults.

Determinism:
    Payload construction is deterministic for fixed inputs and configuration.
    Generated output remains non-deterministic because inference runs remotely.
"""

from app.llm.provider_config import SYSTEM_MESSAGE, MODEL_NAME
from app.llm.client import send_request


def generate_answer(prompt: str, stream=False):
    """Invoke configured model with shared generation defaults.

    Args:
        prompt: Fully constructed user prompt from orchestration/prompting layers.
        stream: Whether to request token streaming from compatible providers.

    Returns:
        Provider response object:
        - generator in streaming paths,
        - final string in non-stream paths,
        - or sanitized error string on failures (delegated by `client`).

    Parameter semantics:
        - `temperature=0.45`: moderate randomness.
        - `top_p=0.9`: nucleus sampling cap.
        - `presence_penalty=0.4`: encourages topic spread.
        - `frequency_penalty=0.5`: discourages repetition.

    Interaction with prompting layer:
        The function does not build prompts. It only wraps the supplied prompt with
        `SYSTEM_MESSAGE` and forwards it to transport.

    Failure scenarios:
        Transport/provider failures are normalized in `client.send_request` and
        propagated as return values rather than raised by default.
    """

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.45,
        "top_p": 0.9,
        "presence_penalty": 0.4,
        "frequency_penalty": 0.5,
        "stream": stream
    }

    return send_request(payload, stream)
