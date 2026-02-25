"""Provider-specific transport client for LLM requests.

Architectural role:
    Executes HTTP requests against configured model providers and normalizes response
    materialization for streaming and non-streaming paths.

Model invocation flow:
    `service.generate_answer` -> `send_request(payload, stream)` -> provider branch
    (OpenAI-compatible / Anthropic / Gemini) -> parsed text or streamed deltas.

Retry behavior:
    No retry loop is implemented. Each HTTP call is attempted once with timeout=120s.

Determinism:
    Provider routing and payload transformation are deterministic for fixed config and
    payload. Output text remains non-deterministic due to remote model inference.

Failure handling model:
    Exceptions are converted into sanitized error strings (or streamed error chunks)
    to keep caller-side control flow stable.
"""

import requests
import json

from app.llm.provider_config import (
    INTERFACE_MODE,
    PROVIDER,
    MODEL_NAME,
    PROVIDERS,
    ANTHROPIC_URL,
    GEMINI_URL_TEMPLATE,
    load_key,
)


def _build_sanitized_http_error(provider_name: str, err: requests.exceptions.RequestException) -> str:
    """Build provider-labeled HTTP error text without exposing raw internals.

    Args:
        provider_name: Active provider label.
        err: Request exception instance.

    Returns:
        Sanitized error string with optional status code.
    """
    status_code = None
    if getattr(err, "response", None) is not None:
        status_code = getattr(err.response, "status_code", None)

    label = str(provider_name or "provider").upper()
    if status_code:
        return f"\n{label} HTTP ERROR ({status_code})\n"
    return f"\n{label} HTTP ERROR\n"


def _sanitize_runtime_error(provider_name: str) -> str:
    """Build generic provider-labeled runtime failure text."""
    label = str(provider_name or "provider").upper()
    return f"\n{label} REQUEST FAILED\n"


def send_request(payload: dict, stream: bool):
    """Send one request to the configured provider and parse response content.

    Args:
        payload: Provider-agnostic request payload produced by caller layers.
        stream: Streaming preference for providers that support SSE-like output.

    Returns:
        - Streaming generator for OpenAI-compatible stream mode.
        - Final response string for non-stream mode.
        - Sanitized error string on failure.

    Provider handling:
        - OpenAI-compatible providers: direct pass-through payload.
        - Anthropic: message remap + optional `system` + default `max_tokens=1024`.
        - Gemini: message remap to `contents` and `generationConfig` mapping.

    Parameter handling:
        - OpenAI-compatible: payload forwarded unchanged.
        - Anthropic/Gemini: only supported fields are forwarded/mapped.

    Failure scenarios:
        - Missing keys return provider-specific `KEY FILE NOT FOUND` messages.
        - Unsupported provider returns `INVALID PROVIDER`.
        - Request/runtime errors return sanitized provider-labeled text.
    """
    try:

        if PROVIDER in PROVIDERS and PROVIDER not in ["anthropic", "gemini"]:

            config = PROVIDERS[PROVIDER]
            LLM_URL = config["url"]
            key_file = config["key_file"]

            headers = {
                "Content-Type": "application/json"
            }

            if key_file:
                api_key = load_key(key_file)
                if not api_key:
                    return f"\n{PROVIDER.upper()} KEY FILE NOT FOUND\n"

                headers["Authorization"] = f"Bearer {api_key}"

            if stream:

                def stream_generator():
                    """Yield incremental text deltas from OpenAI-compatible streams.

                    Behavior:
                        - Parses line-delimited JSON chunks.
                        - Extracts delta text from common response shapes.
                        - In CLI mode, prints chunks directly; otherwise yields.

                    Error handling:
                        Request exceptions are converted to sanitized error output.
                    """
                    try:
                        with requests.post(
                            LLM_URL,
                            headers=headers,
                            json=payload,
                            stream=True,
                            timeout=120,
                        ) as response:

                            response.raise_for_status()
                            response.encoding = "utf-8"

                            full_text = ""

                            for line in response.iter_lines(decode_unicode=True):

                                if not line:
                                    continue

                                if line.startswith("data: "):
                                    line = line[6:]

                                if line.strip() == "[DONE]":
                                    break

                                try:
                                    data = json.loads(line)
                                except Exception:
                                    continue

                                delta = None

                                if "choices" in data:
                                    choice = data["choices"][0]

                                    if (
                                        "delta" in choice
                                        and "content" in choice["delta"]
                                    ):
                                        delta = choice["delta"]["content"]

                                    elif (
                                        "message" in choice
                                        and "content" in choice["message"]
                                    ):
                                        delta = choice["message"]["content"]

                                    elif "text" in choice:
                                        delta = choice["text"]

                                elif (
                                    "message" in data
                                    and "content" in data["message"]
                                ):
                                    delta = data["message"]["content"]

                                if delta:
                                    full_text += delta

                                    if INTERFACE_MODE == "cli":
                                        print(delta, end="", flush=True)
                                    else:
                                        yield delta

                            if INTERFACE_MODE == "cli":
                                return full_text.strip()
                    except requests.exceptions.RequestException as err:
                        safe_error = _build_sanitized_http_error(PROVIDER, err)
                        if INTERFACE_MODE == "cli":
                            print(safe_error, end="", flush=True)
                        else:
                            yield safe_error

                return stream_generator()

            else:
                response = requests.post(
                    LLM_URL,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )

                response.raise_for_status()
                data = response.json()

                return data["choices"][0]["message"]["content"].strip()

        elif PROVIDER == "anthropic":

            api_key = load_key("config/anthropic.key")
            if not api_key:
                return "\nANTHROPIC KEY FILE NOT FOUND\n"

            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }

            messages = payload.get("messages", [])
            system_prompt = None
            anthropic_messages = []

            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                role = msg.get("role")
                content = msg.get("content", "")

                if role == "system":
                    if isinstance(content, str) and content.strip():
                        system_prompt = content.strip()
                elif role in ["user", "assistant"]:
                    anthropic_messages.append({
                        "role": role,
                        "content": content,
                    })

            anthropic_payload = {
                "model": payload.get("model", MODEL_NAME),
                "max_tokens": payload.get("max_tokens", 1024),
                "messages": anthropic_messages,
            }

            if system_prompt:
                anthropic_payload["system"] = system_prompt

            if "temperature" in payload:
                anthropic_payload["temperature"] = payload["temperature"]
            if "top_p" in payload:
                anthropic_payload["top_p"] = payload["top_p"]

            response = requests.post(
                ANTHROPIC_URL,
                headers=headers,
                json=anthropic_payload,
                timeout=120,
            )

            response.raise_for_status()
            data = response.json()

            return data["content"][0]["text"].strip()

        elif PROVIDER == "gemini":

            api_key = load_key("config/gemini.key")
            if not api_key:
                return "\nGEMINI KEY FILE NOT FOUND\n"

            url = GEMINI_URL_TEMPLATE.format(model=MODEL_NAME)

            headers = {
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            }

            messages = payload.get("messages", [])
            gemini_contents = []

            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                role = msg.get("role")
                content = msg.get("content", "")

                if not content:
                    continue

                if role == "assistant":
                    gemini_role = "model"
                elif role in ["user", "system"]:
                    gemini_role = "user"
                else:
                    continue

                gemini_contents.append({
                    "role": gemini_role,
                    "parts": [{"text": str(content)}],
                })

            gemini_payload = {
                "contents": gemini_contents,
            }

            generation_config = {}
            if "temperature" in payload:
                generation_config["temperature"] = payload["temperature"]
            if "top_p" in payload:
                generation_config["topP"] = payload["top_p"]
            if generation_config:
                gemini_payload["generationConfig"] = generation_config

            response = requests.post(
                url,
                headers=headers,
                json=gemini_payload,
                timeout=120,
            )

            response.raise_for_status()
            data = response.json()

            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        else:
            return "\nINVALID PROVIDER\n"

    except KeyboardInterrupt:
        return ""

    except requests.exceptions.RequestException as err:
        return _build_sanitized_http_error(PROVIDER, err)

    except Exception:
        return _sanitize_runtime_error(PROVIDER)
