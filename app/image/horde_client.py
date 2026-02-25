"""AI Horde-specific image-generation client.

Processing flow:
    1. Read provider endpoint configuration.
    2. Build submission payload from prompt + generation params.
    3. Submit async generation job.
    4. Poll status endpoint until completion/fault.
    5. Return first generated image URL.

Multimodal scope:
    This module only handles text-to-image generation requests. It does not
    process uploaded files or perform multimodal content extraction.

Base64 and temporary files:
    - No Base64 decoding is performed.
    - No temporary files are created or cleaned up.

Size validation:
    - Width/height/steps are forwarded as provided.
    - No explicit upper-bound validation is enforced in this client.

Error handling strategy:
    - Configuration and provider-state issues raise `RuntimeError`.
    - HTTP-layer failures propagate via `requests.raise_for_status()`.

Determinism:
    - Client-side request construction and polling logic are deterministic.
    - Completion timing and generated output are non-deterministic externally.

Security considerations:
    - Current debug prints include sensitive/request data and provider responses.

Performance characteristics:
    - Uses synchronous HTTP and blocking sleep-based polling.
    - Poll loop has no max-attempt/time budget in this module.
"""

import os
import json
import requests
import time
from dotenv import load_dotenv

from app.llm.provider_config import IMAGE_PROVIDERS, IMAGE_PROVIDER, load_key

load_dotenv()


def send_ai_horde_request(
    prompt: str,
    width: int,
    height: int,
    steps: int,
) -> dict:
    """Submit and poll an AI Horde async image generation job.

    Args:
        prompt: User prompt forwarded to AI Horde.
        width: Requested output width.
        height: Requested output height.
        steps: Sampling/inference steps.

    Returns:
        Dict containing `{"image_url": <url>}` when generation completes.

    Interaction with API/core:
        Called by `app.image.service.generate_image`; its output is later adapted
        by API response wrappers (for example Markdown image link conversion).

    Failure handling:
        - Missing API key/provider config/status URL -> `RuntimeError`
        - Faulted job -> `RuntimeError`
        - Missing image URL after completion -> `RuntimeError`
        - HTTP transport/status failures -> propagated request exceptions
    """
    provider_config = IMAGE_PROVIDERS.get("ai_horde")
    if not provider_config:
        raise RuntimeError(
            f"AI Horde provider config missing (active IMAGE_PROVIDER={IMAGE_PROVIDER})"
        )

    AI_HORDE_API_KEY = os.getenv("AI_HORDE_API_KEY")
    model_name = os.getenv("AI_HORDE_MODEL", "Anything v5")

    if not AI_HORDE_API_KEY:
        raise RuntimeError("AI_HORDE_API_KEY is not set in .env")
    print("AI HORDE API KEY RAW:", repr(AI_HORDE_API_KEY))
    headers = {
        "apikey": AI_HORDE_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": prompt,
        "params": {
            "width": width,
            "height": height,
            "steps": steps
        }
    }

    print("FINAL PAYLOAD:", json.dumps(payload, indent=2))
    submit_response = requests.post(provider_config["url"], json=payload, headers=headers)
    print("AI HORDE STATUS:", submit_response.status_code)
    print("AI HORDE RESPONSE:", submit_response.text)
    submit_response.raise_for_status()
    submit_data = submit_response.json()

    job_id = submit_data.get("id")
    if not job_id:
        raise RuntimeError("AI Horde did not return a job id.")

    status_url = provider_config.get("status_url")
    if not status_url:
        raise RuntimeError("AI Horde status_url missing in provider config.")

    while True:
        status_response = requests.get(f"{status_url}{job_id}", headers=headers)
        if status_response.status_code == 429:
            time.sleep(3)
            continue
        status_response.raise_for_status()
        status_data = status_response.json()

        if status_data.get("faulted"):
            raise RuntimeError("AI Horde job faulted.")

        finished = bool(status_data.get("done") or status_data.get("finished"))
        generations = status_data.get("generations") or []

        if finished and generations:
            image_url = generations[0].get("img") or generations[0].get("image_url")
            if not image_url:
                raise RuntimeError("AI Horde finished but no image URL returned.")
            return {"image_url": image_url}

        time.sleep(2)
