"""Image service dispatcher used by core `/image` command handling.

Role in pipeline:
    - Receives image-generation parameters from orchestration.
    - Selects provider path (`ai_horde` vs generic provider client).
    - Returns provider response payload unchanged to upstream callers.

API integration:
    Core returns this payload to API adapters, which may format image URLs for
    client-facing responses.

Multimodal/Base64/temp-file scope:
    - No file ingestion or multimodal extraction.
    - No Base64 decoding logic.
    - No temporary-file lifecycle management.

Size validation:
    - No hard size limits are validated here.
    - For AI Horde with non-high-quality mode, parameters are normalized to
      512x512 and 25 steps when an API key is present.

Error handling strategy:
    - Exceptions from provider clients are intentionally propagated.

Determinism:
    - Provider branch selection is deterministic for fixed env/config/inputs.
    - Output content remains externally non-deterministic.

Performance characteristics:
    - Thin dispatch layer with negligible local overhead.
    - End-to-end latency depends on downstream provider client behavior.
"""

import os

from app.image.client import send_image_request
from app.image.horde_client import send_ai_horde_request
from app.llm.provider_config import IMAGE_MODEL, IMAGE_PROVIDER


def generate_image(
    prompt: str,
    width: int = 512,
    height: int = 512,
    steps: int = 25,
    high_quality: bool = False,
) -> dict:
    """Generate an image via configured provider adapter.

    Args:
        prompt: Text prompt for generation.
        width: Requested width (provider-dependent semantics).
        height: Requested height.
        steps: Requested inference steps.
        high_quality: Whether to keep caller-specified params on AI Horde.

    Returns:
        Provider response dictionary (typically containing image reference data).
    """
    payload = {
        "prompt": prompt,
        "steps": steps,
        "width": width,
        "height": height,
    }

    if IMAGE_PROVIDER == "ai_horde":
        if os.getenv("AI_HORDE_API_KEY") and not high_quality:
            width = 512
            height = 512
            steps = 25
        return send_ai_horde_request(prompt, width, height, steps)

    return send_image_request(payload)
