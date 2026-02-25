"""
HTTP API adapter for the ContextualAI engine.

Architectural role:
- Expose OpenAI-compatible HTTP interfaces.
- Enforce adapter-level input validation and model/domain selection.
- Delegate generation/routing work to `app.core.engine.process_message`.
- Normalize engine output to response transport contracts (JSON or SSE).

Endpoint responsibilities:
- `GET /v1/models`: discover and expose local knowledge domains.
- `POST /v1/chat/completions`: validate input, apply hard-trigger guard,
  invoke core, and format completion output.

API request lifecycle (`POST /v1/chat/completions`):
1. Parse request JSON (`messages`, `model`, optional `stream`).
2. Validate required fields and requested domain.
3. Apply hard trigger guard for tool-style prompts (`### Task:`).
4. Extract the latest user message and forward it to `app.core.engine.process_message`.
5. Format the engine output for non-stream or streaming response contracts.

Input validation behavior:
- Missing `messages` -> HTTP 400.
- Missing `model` -> HTTP 400.
- Empty domain catalog -> HTTP 400.
- Unknown requested model/domain -> HTTP 400.

Hard trigger handling:
- Only a tool-prompt guard is implemented (`### Task:` -> empty assistant response).
- Route triggers like `/memory`, `/academic`, `/general` are not handled here.

Error handling strategy:
- Explicit validation failures return structured HTTP 400 JSON responses.
- Streaming cancels/disconnects are handled inside the SSE generator.
- Unexpected parsing/runtime exceptions are not globally wrapped in this module.

Response formatting:
- Non-stream mode returns an OpenAI-compatible completion envelope.
- Stream mode returns SSE `chat.completion.chunk` frames ending with `[DONE]`.

Side effects:
- Reads local filesystem to discover domains under `knowledge/`.
- Emits debug logs only when `DEBUG == "true"`.
- Loads environment variables at import time via `load_dotenv()`.

Determinism considerations:
- Response content depends on engine behavior and filesystem domain state.
- IDs and timestamps are generated per request/chunk (`uuid`, `time.time()`).
"""

from dotenv import load_dotenv

load_dotenv()

import uuid
import os
import time
import json
import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from app.core.engine import process_message

app = FastAPI()
# Sensitive request/response debug logging is opt-in.
DEBUG = os.getenv("DEBUG") == "true"

# ============================================================
# Dynamic Domain Discovery
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KNOWLEDGE_PATH = os.path.join(BASE_DIR, "knowledge")


def get_available_domains():
    """Return discovered knowledge domains from `knowledge/` directory names."""
    if not os.path.exists(KNOWLEDGE_PATH):
        return []

    return [
        name
        for name in os.listdir(KNOWLEDGE_PATH)
        if os.path.isdir(os.path.join(KNOWLEDGE_PATH, name))
    ]


# ============================================================
# Generator Flatten Helper
# ============================================================

def flatten_generator(gen):
    """
    Recursively flatten nested iterables produced by engine streaming output.

    Strings and bytes are treated as terminal values to avoid character-level splitting.
    """
    for item in gen:
        if hasattr(item, "__iter__") and not isinstance(item, (str, bytes)):
            yield from flatten_generator(item)
        else:
            yield item


# ============================================================
# Helper: Detect OpenWebUI Tool Prompts
# ============================================================

def is_tool_prompt(text: str) -> bool:
    """
    Detect OpenWebUI tool-execution prompts.

    This serves as a local hard trigger guard to prevent assistant text output
    for tool-task control messages.
    """
    if not text:
        return False
    return text.strip().startswith("### Task:")


# ============================================================
# Request Schema
# ============================================================

class ChatRequest(BaseModel):
    """
    Reference schema for chat payload shape.

    Note:
    - The active endpoint currently parses JSON directly from `Request`.
    """
    message: str
    domain: str


# ============================================================
# Model Listing
# ============================================================

@app.get("/v1/models")
def list_models():
    """
    Return discoverable local domains as OpenAI-style model metadata.

    Response formatting:
    - `object: "list"`
    - `data[]` entries with `id`, `object`, `created`, `owned_by`
    """
    available_domains = get_available_domains()

    return {
        "object": "list",
        "data": [
            {
                "id": domain,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local"
            }
            for domain in available_domains
        ]
    }


# ============================================================
# OpenAI-Compatible Chat Completions
# ============================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.

    Endpoint responsibilities:
    - Input parsing/validation for required payload fields.
    - Domain selection by validating `model` against local knowledge directories.
    - Hard trigger short-circuit for tool prompts.
    - Delegation to `process_message` in the core layer.
    - Output normalization to non-stream JSON or SSE chunk protocol.

    Input validation behavior:
    - Returns HTTP 400 for missing `messages`.
    - Returns HTTP 400 for missing `model`.
    - Returns HTTP 400 when no domains are available.
    - Returns HTTP 400 when requested model/domain is unknown.

    Error handling strategy:
    - Explicit validation failures return structured 400 JSON errors.
    - Streaming generator catches client-disconnect exceptions and exits quietly.
    - Unexpected exceptions from JSON parsing or engine execution are not wrapped
      here and follow FastAPI default exception handling.

    Hard trigger handling:
    - Tool-style control prompts (`### Task:`) short-circuit to an empty
      assistant response.

    Determinism considerations:
    - Uses wall-clock timestamps and random UUIDs in response envelopes.
    - Stream chunk boundaries reflect iterable behavior of the engine result.
    """

    body = await request.json()

    messages = body.get("messages", [])
    stream = body.get("stream", False)
    model_name = body.get("model")

    if DEBUG:
        print("\n==== API DEBUG START ====")
        print("Incoming messages:", messages)
        print("Stream:", stream)
        print("Model:", model_name)

    if not messages:
        return JSONResponse(status_code=400, content={"error": "No messages provided"})

    if not model_name:
        return JSONResponse(status_code=400, content={"error": "No model provided"})

    available_domains = get_available_domains()

    if not available_domains:
        return JSONResponse(status_code=400, content={"error": "No domains available."})
    if model_name not in available_domains:
        return JSONResponse(status_code=400, content={"error": "Unknown model requested"})
    domain = model_name

    if DEBUG:
        print("Selected domain:", domain)

    # ============================================================
    # Tool Prompt Block (hard trigger guard for tool-task prompts)
    # ============================================================

    last_message = messages[-1]

    if (
        last_message.get("role") == "user"
        and isinstance(last_message.get("content"), str)
        and is_tool_prompt(last_message.get("content"))
    ):
        if DEBUG:
            print("Tool prompt detected.")
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop"
                }
            ]
        }

    # ============================================================
    # Conversation sync
    # Adapter behavior: forward only the latest user turn to core.
    # ============================================================
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = str(msg.get("content", ""))
            break

    if DEBUG:
        print("User message extracted:", user_message)

    if stream:

        result = await process_message(user_message, domain=domain)

        if DEBUG:
            print("Engine result type:", type(result))
            print("Engine result repr:", repr(result))

        if result is None:
            if DEBUG:
                print("Engine returned None -> converted to empty string")
            result = ""
        elif isinstance(result, dict) and "image_url" in result:
            result = f"![image]({result['image_url']})"

        if DEBUG:
            print("Streaming mode activated")

        if isinstance(result, str):
            if DEBUG:
                print("Result is string -> wrapping into generator")
            result_text = result

            def single_chunk():
                yield result_text
            result = single_chunk()

        async def event_generator():
            """
            Yield SSE frames matching OpenAI chunk semantics.

            Response formatting:
            - Content chunks use `chat.completion.chunk` with `delta.content`.
            - Terminal chunk sets `finish_reason: "stop"`.
            - Final sentinel frame is `[DONE]`.

            Side effects:
            - Checks client connection state to stop work on disconnect.
            - Attempts to close engine iterables supporting `aclose()` / `close()`.

            Error handling:
            - Handles cancellation/broken connection exceptions and stops stream.
            """
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            try:
                for chunk in flatten_generator(result):
                    if await request.is_disconnected():
                        if DEBUG:
                            print("Client disconnected during stream.")
                        return

                    if DEBUG:
                        print("Streaming chunk:", repr(chunk))
                    data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": str(chunk)},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(data)}\n\n"

                if DEBUG:
                    print("Streaming finished")

                end_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }

                yield f"data: {json.dumps(end_data)}\n\n"
                yield "data: [DONE]\n\n"
            except (asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
                if DEBUG:
                    print("Streaming cancelled by client.")
                return
            finally:
                if hasattr(result, "aclose"):
                    try:
                        await result.aclose()
                    except Exception:
                        pass
                elif hasattr(result, "close"):
                    try:
                        result.close()
                    except Exception:
                        pass

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ============================================================
    # Engine Call (non-stream mode)
    # ============================================================

    result = await process_message(user_message, domain=domain)

    if DEBUG:
        print("Engine result type:", type(result))
        print("Engine result repr:", repr(result))

    # 🔥 ULTRA-MINIMAL NONE FIX
    if result is None:
        if DEBUG:
            print("Engine returned None -> converted to empty string")
        result = ""
    elif isinstance(result, dict) and "image_url" in result:
        result = f"![image]({result['image_url']})"

    # ============================================================
    # NON-STREAM
    # Result normalization guarantees assistant message content is a string.
    # ============================================================

    if DEBUG:
        print("Non-stream mode")

    if hasattr(result, "__iter__") and not isinstance(result, str):
        result = "".join(str(x) for x in flatten_generator(result))
    else:
        result = str(result)

    if DEBUG:
        print("Final result:", repr(result))
        print("==== API DEBUG END ====\n")

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop"
            }
        ]
    }
