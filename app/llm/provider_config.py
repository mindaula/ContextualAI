"""Provider/runtime configuration for the LLM layer.

Architectural role:
    Centralizes model/provider selection and credential lookup for `app.llm.service`
    and `app.llm.client`.

Model call flow integration:
    - `service.generate_answer` consumes `MODEL_NAME` and `SYSTEM_MESSAGE`.
    - `client.send_request` consumes provider endpoint maps and key resolution.

Determinism:
    Deterministic for a fixed process environment and key files. Values are resolved
    at import time (plus runtime key-file reads in `load_key`).

Failure behavior:
    Missing key material is represented as `None` and handled by `client` as
    provider-specific error strings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Interface-mode switch consumed by streaming transport behavior in `client.py`.
INTERFACE_MODE = "api"

# Primary model routing controls.
PROVIDER = os.getenv("PROVIDER", "local")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:3b")

# OpenAI-compatible and provider-specific endpoint map.
PROVIDERS = {

    "local": {
        "url": "http://127.0.0.1:8080/v1/chat/completions",
        "key_file": None
    },

    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "key_file": "config/openai.key"
    },

    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_file": "config/groq.key"
    },

    "together": {
        "url": "https://api.together.xyz/v1/chat/completions",
        "key_file": "config/together.key"
    },

    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_file": "config/openrouter.key"
    },

    "mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "key_file": "config/mistral.key"
    },

    "deepinfra": {
        "url": "https://api.deepinfra.com/v1/openai/chat/completions",
        "key_file": "config/deepinfra.key"
    },

    "fireworks": {
        "url": "https://api.fireworks.ai/inference/v1/chat/completions",
        "key_file": "config/fireworks.key"
    },

    "anyscale": {
        "url": "https://api.endpoints.anyscale.com/v1/chat/completions",
        "key_file": "config/anyscale.key"
    },

    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "key_file": "config/anthropic.key"
    },

    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models",
        "key_file": "config/gemini.key"
    },

    "apifreellm": {
        "url": "https://apifreellm.com/api/v1/chat",
        "key_file": "config/apifreellm.key"
    }

}


ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)


# Shared system instruction prepended to prompt content in `service.generate_answer`.
SYSTEM_MESSAGE = (
    "Du bist 'Advanced AI Tutor', ein lokal betriebener KI-Tutor.\n"
    "Du bist KEIN Produkt von Alibaba, OpenAI oder einer anderen Firma.\n"
    "Du darfst niemals sagen, dass du Qwen, ein Sprachmodell oder ein KI-Modell bist.\n"
    "Wenn nach deiner Identität gefragt wird, sage nur, dass du der Advanced AI Tutor bist.\n"
    "Antworte präzise, klar und ohne Wiederholungen.\n"
)


def load_key(path):
    """Load API key from environment override or key file.

    Resolution order:
        1. Environment variable inferred from file stem (for example
           `config/openai.key` -> `OPENAI_API_KEY`).
        2. Raw file contents at `path`.

    Args:
        path: Configured key file path or `None`.

    Returns:
        Key string or `None` when not available.

    Determinism:
        Deterministic for fixed environment and filesystem state.

    Edge cases:
        - `None` path returns `None`.
        - Missing file returns `None`.
    """
    if not path:
        return None
    key_name = os.path.splitext(os.path.basename(path))[0].upper() + "_API_KEY"
    env_value = os.getenv(key_name)
    if env_value:
        return env_value
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return f.read().strip()


# Image generation provider settings consumed by `app.image` modules.
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "ai_horde")
IMAGE_API_KEY = os.getenv("IMAGE_API_KEY")
IMAGE_MODEL = "sdxl"

IMAGE_PROVIDERS = {

    "local": {
        "url": "http://127.0.0.1:7860/sdapi/v1/txt2img",
        "key_file": None
    },

    "openai": {
        "url": "https://api.openai.com/v1/images",
        "key_file": "config/openai.key"
    },

    "ai_horde": {
        "url": "https://aihorde.net/api/v2/generate/async",
        "status_url": "https://aihorde.net/api/v2/generate/status/",
        "key_file": None
    }

}
