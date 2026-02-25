"""Generic image-provider HTTP client.

Processing flow:
    1. Resolve active provider config from `app.llm.provider_config`.
    2. Optionally load API key from configured key file.
    3. Submit JSON payload to provider endpoint.
    4. Return parsed JSON response or raise on non-200 status.

Base64 and temporary files:
    - This module does not decode Base64 content.
    - This module does not create or manage temporary files.

Size validation:
    - No local payload-size validation is performed here.

Error handling strategy:
    - Misconfiguration and HTTP failures raise exceptions for upstream handling.

Determinism:
    - Request assembly is deterministic for fixed inputs/configuration.
    - Final output remains provider/network dependent.

Security considerations:
    - Exceptions may include upstream provider response bodies.
"""

import requests
import base64

from app.llm.provider_config import IMAGE_PROVIDER, IMAGE_PROVIDERS, load_key


def send_image_request(payload: dict) -> dict:
    """Send an image-generation request to the currently selected provider.

    Args:
        payload: Provider JSON payload (prompt/size/step parameters).

    Returns:
        Parsed JSON response from provider.

    Interaction with API/core:
        Called by `app.image.service.generate_image`, which is invoked by core
        `/image` command handling and surfaced by API adapters.

    Error handling:
        - Unknown provider -> `ValueError`
        - Missing/empty configured API key -> `RuntimeError`
        - Non-200 HTTP response -> `RuntimeError`
    """
    provider_config = IMAGE_PROVIDERS.get(IMAGE_PROVIDER)
    if not provider_config:
        raise ValueError(f"Unknown image provider: {IMAGE_PROVIDER}")

    url = provider_config["url"]
    key_file = provider_config.get("key_file")
    headers = {}

    if key_file is not None:
        api_key = load_key(key_file)
        if not api_key:
            raise RuntimeError(f"Image API key file missing or empty: {key_file}")
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise RuntimeError(
            f"Image request failed with status {response.status_code}: {response.text}"
        )

    return response.json()
