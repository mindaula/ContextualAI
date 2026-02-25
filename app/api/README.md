# `app/api` Module

## Module Purpose
`app/api` is the system boundary layer for client-facing interaction.

It provides:
- HTTP endpoints (`FastAPI`) with OpenAI-compatible response envelopes.
- Local terminal interfaces (`main.py`, `cli.py`) for interactive use.
- Multimodal preprocessing helpers for file-to-text prompt augmentation.

This layer does not perform model reasoning directly; it delegates to `app.core.engine.process_message(...)`.

## Architectural Role
In the overall architecture, `app/api` is an adapter layer between external callers and internal orchestration.

Primary responsibilities:
- Parse and validate adapter-level request fields.
- Resolve requested model/domain names from local `knowledge/` directories.
- Normalize engine outputs into transport-specific formats (JSON or SSE).
- Expose local session controls in CLI entrypoints.

## Endpoint Overview
HTTP endpoints are defined in `app/api/http_api.py`.

### `GET /v1/models`
- Discovers available domains by directory listing under `knowledge/`.
- Returns:
  - `object: "list"`
  - `data[]` entries shaped as OpenAI-style model records.

### `POST /v1/chat/completions`
Endpoint responsibilities:
- Parse payload fields from request JSON.
- Validate `messages` and `model`.
- Validate model/domain availability.
- Apply local hard-trigger guard for tool prompts.
- Forward the latest user turn to core.
- Return either non-stream completion JSON or SSE chunks.

## API Request Lifecycle
`POST /v1/chat/completions` processing sequence:

1. Parse JSON body.
2. Read `messages`, `stream`, and `model`.
3. Validate:
   - `messages` is present/non-empty.
   - `model` is present.
   - at least one local domain exists.
   - requested model exists in discovered domains.
4. Check hard trigger for tool prompt (`### Task:`).
5. Extract the latest `role == "user"` message.
6. Invoke `await process_message(user_message, domain=model_name)`.
7. Normalize output:
   - `None` -> `""`
   - `{"image_url": ...}` -> Markdown image text
   - nested iterables -> flattened chunk/string output
8. Respond:
   - non-stream completion object, or
   - stream of `chat.completion.chunk` SSE frames ending with `[DONE]`.

## Hard Trigger Handling
Implemented in this module:
- HTTP tool-prompt guard:
  - if last user message begins with `### Task:`, return an empty assistant completion without engine invocation.
- CLI command trigger:
  - `/mode` in `app/api/cli.py` for domain inspection/switching.

Not implemented in this module:
- No explicit `/memory`, `/academic`, or `/general` route/command routing in HTTP handlers.
- Those triggers, if supported, must be interpreted downstream.

## Session Management
HTTP:
- No explicit HTTP session object or session ID is managed in `http_api.py`.
- Only the latest user turn is forwarded from `messages`.

CLI (`main.py`, `cli.py`):
- `exit` / `quit`: archive session.
- `empty chat` / `clear chat`: discard current session.
- EOF handling discards current session.

## Error Handling Strategy
HTTP explicit `400` responses:
- missing messages -> `{"error": "No messages provided"}`
- missing model -> `{"error": "No model provided"}`
- missing domains -> `{"error": "No domains available."}`
- unknown model -> `{"error": "Unknown model requested"}`

Streaming behavior:
- disconnect/cancel exceptions are handled and stream emission stops cleanly.
- client disconnect is checked per chunk.

General runtime behavior:
- JSON parsing and engine exceptions are not globally wrapped in `http_api.py`; framework defaults apply.

Multimodal file preprocessing:
- per-file exceptions are handled per item and converted to a generic marker (`[File processing error]`).
- processing continues for subsequent files.

## Response Formatting
`GET /v1/models`:
- OpenAI-style model list envelope.

`POST /v1/chat/completions` non-stream:
- OpenAI-style completion object with one assistant message in `choices[0].message.content`.

`POST /v1/chat/completions` stream:
- `text/event-stream` with chunk objects containing `delta.content`.
- terminal chunk sets `finish_reason: "stop"`.
- final SSE event is `[DONE]`.

## Side Effects
- Environment loading at import time via `load_dotenv()` in HTTP/CLI modules.
- Filesystem scans of `knowledge/` for domain discovery.
- Console logging under HTTP DEBUG gate.
- CLI prints status and interaction output to stdout.
- Multimodal module:
  - ensures upload base directory exists at import time.
  - writes decoded base64 payloads to temporary files.
  - deletes those temporary files in `finally`, including failure paths.

## Determinism Considerations
- Response envelopes include runtime-generated UUIDs and timestamps.
- Stream chunk boundaries depend on iterable shape from `process_message`.
- Domain discovery depends on current filesystem state.
- OCR/PDF/CSV parser outputs can vary across dependency/runtime environments.

## Security Considerations
- Domain/model selection is constrained to local `knowledge/` subdirectories.
- Multimodal validation enforces:
  - allowed base directory (`FILE_INPUT_BASE_DIR`, default `uploads/`),
  - extension allowlist,
  - maximum file size,
  - `file://` host restriction (empty or `localhost` only).
- Base64 payload size is pre-validated before decoding/writing.
- Prompt-injected file error text is sanitized (generic marker only).
- Sensitive HTTP debug logs are gated behind `DEBUG == "true"`.

## Example Request and Response Flow
### Request
```http
POST /v1/chat/completions
Content-Type: application/json
```

```json
{
  "model": "biology",
  "stream": false,
  "messages": [
    {"role": "system", "content": "You are a tutor."},
    {"role": "user", "content": "Explain photosynthesis briefly."}
  ]
}
```

### Processing Notes
- `model` must match an available domain directory.
- Adapter extracts the latest user turn and forwards that text to `process_message`.

### Non-stream response shape
```json
{
  "id": "chatcmpl-<uuid>",
  "object": "chat.completion",
  "created": 1730000000,
  "model": "biology",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "<engine output>"},
      "finish_reason": "stop"
    }
  ]
}
```

### Stream frame shape
```text
data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","created":1730000000,"model":"biology","choices":[{"index":0,"delta":{"content":"<chunk>"},"finish_reason":null}]}

data: {"id":"chatcmpl-<uuid>","object":"chat.completion.chunk","created":1730000000,"model":"biology","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Known Limitations
- HTTP path forwards only the latest user message to core from the provided `messages` array.
- No explicit HTTP handling for `/memory`, `/academic`, or `/general` triggers in this module.
- `ChatRequest` exists for schema reference but is not used as the endpoint binding model.
