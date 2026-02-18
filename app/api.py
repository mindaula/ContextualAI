from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import json
import time
import uuid

from app.engine import process_message

app = FastAPI()


class ChatRequest(BaseModel):
    """
    Request schema for the basic chat endpoint.
    
    Attributes:
        message (str): The user input message to be processed by the engine.
    """
    message: str


# ============================================================
# Standard Chat Endpoint
# ============================================================
# This endpoint accepts a plain user message and forwards it
# to the internal processing engine.
#
# If the engine returns a generator, the response is streamed.
# Otherwise, a standard JSON response is returned.
# ============================================================
@app.post("/chat")
def chat(req: ChatRequest):

    # Forward the user message to the core processing engine
    result = process_message(req.message)

    # If the result is an iterable (e.g. token stream), return streaming response
    if hasattr(result, "__iter__") and not isinstance(result, str):
        return StreamingResponse(result, media_type="text/plain")

    # Fallback: non-streaming response
    return {"response": result}


# ============================================================
# Industry-Standard Chat Completions Endpoint
# ============================================================
# This endpoint implements a widely used chat completion schema
# compatible with common LLM API clients.
#
# It supports:
# - Standard JSON responses
# - Server-Sent Events (SSE) streaming
#
# The schema structure mirrors commonly adopted LLM APIs to
# ensure easy integration with existing tooling.
# ============================================================
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):

    body = await request.json()

    messages = body.get("messages", [])
    stream = body.get("stream", False)

    if not messages:
        return JSONResponse(
            status_code=400,
            content={"error": "No messages provided"}
        )

    # Extract the latest user message from the conversation history
    user_message = messages[-1]["content"]

    # Process the message using the internal engine
    result = process_message(user_message)

    # ============================================================
    # STREAMING MODE (Server-Sent Events)
    # ============================================================
    # Returns incremental token chunks using the SSE protocol.
    # The format follows a commonly adopted streaming schema
    # used by modern LLM APIs.
    # ============================================================
    if stream:

        def event_generator():
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            for chunk in result:
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "local-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk
                            },
                            "finish_reason": None
                        }
                    ]
                }

                # Emit chunk in SSE format
                yield f"data: {json.dumps(data)}\n\n"

            # Final chunk indicating completion
            end_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "local-model",
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

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )

    # ============================================================
    # NON-STREAMING MODE
    # ============================================================
    # Returns a complete chat completion response in a single
    # JSON payload using a standardized schema.
    # ============================================================
    else:

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": "local-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result
                    },
                    "finish_reason": "stop"
                }
            ]
        }
