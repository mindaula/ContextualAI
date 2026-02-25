# app/llm

## 1) Module Purpose
`app/llm` provides the model-access layer for text generation.

It is responsible for:
- Loading runtime provider/model configuration.
- Building provider-agnostic generation payloads from prompts.
- Dispatching requests to provider-specific HTTP endpoints.
- Parsing streaming and non-streaming responses into application-consumable output.

## 2) Architectural Role in the System
`app/llm` sits between orchestration/prompt construction and external model providers.

- Upstream callers:
  - `app/core/engine.py` via `app.llm.service.generate_answer(...)`.
  - `app/nlp/query_rewriter.py` via `app.llm.service.generate_answer(...)`.
  - `app/memory/conversation_memory.py` directly via `app.llm.client.send_request(...)` for summary generation.
- Downstream targets:
  - Local OpenAI-compatible endpoint (`local` provider).
  - Remote providers (OpenAI-compatible APIs, Anthropic, Gemini, etc.).

Module responsibilities are split as follows:
- `provider_config.py`: env and endpoint/key configuration.
- `service.py`: canonical prompt -> payload adapter.
- `client.py`: provider routing, HTTP transport, response parsing, and error normalization.

## 3) Model Invocation Lifecycle
Standard lifecycle for `generate_answer(prompt, stream=...)`:

1. `service.generate_answer` receives fully constructed prompt text.
2. Service wraps prompt with:
   - `SYSTEM_MESSAGE` (system role),
   - user prompt (user role),
   - generation parameters (`temperature`, `top_p`, penalties, stream flag),
   - `MODEL_NAME`.
3. Service calls `client.send_request(payload, stream)`.
4. Client selects provider branch by `PROVIDER`:
   - OpenAI-compatible branch (`local`, `openai`, `groq`, etc.),
   - Anthropic branch,
   - Gemini branch,
   - fallback invalid-provider path.
5. Client returns either:
   - streamed chunks (generator, OpenAI-compatible stream mode),
   - final response string,
   - or sanitized error text.

## 4) Parameter Handling
### Parameters set in `service.generate_answer`
- `temperature = 0.45`
- `top_p = 0.9`
- `presence_penalty = 0.4`
- `frequency_penalty = 0.5`
- `stream = <caller value>`
- `model = MODEL_NAME`

### Provider-specific mapping behavior in `client.send_request`
- OpenAI-compatible providers:
  - Payload forwarded as-is.
- Anthropic:
  - Messages remapped to Anthropic message format.
  - System message extracted to top-level `system` field.
  - `max_tokens` defaults to `1024` if not supplied in payload.
  - `temperature` and `top_p` forwarded when present.
- Gemini:
  - Messages remapped to `contents` with provider-specific role mapping.
  - `temperature` forwarded when present.
  - `top_p` mapped to `topP`.

### Configuration inputs
- `PROVIDER` and `MODEL_NAME` from environment (`provider_config.py`).
- API keys via env override or key file resolution (`load_key`).

## 5) Determinism Analysis
Deterministic components:
- Provider selection logic for fixed `PROVIDER`.
- Payload construction in `service.py`.
- Message remapping logic for Anthropic/Gemini.
- Sanitized error string formatting.

Non-deterministic components:
- Remote model inference outputs.
- Network behavior and provider-side variability.
- Runtime environment values loaded at process startup.

Import-time determinism note:
- `provider_config.py` calls `load_dotenv()` and reads env values at import time; behavior depends on process environment and `.env` contents at startup.

## 6) Error Handling Strategy
### Credential/config errors
- Missing key for selected provider returns provider-specific error text:
  - `"<PROVIDER> KEY FILE NOT FOUND"`

### Provider selection errors
- Unknown provider returns:
  - `"INVALID PROVIDER"`

### Transport/runtime errors
- HTTP/Request failures are normalized by `_build_sanitized_http_error(...)`.
- Other runtime exceptions are normalized by `_sanitize_runtime_error(...)`.
- `KeyboardInterrupt` returns empty string.

### Streaming path behavior
- Streaming parser ignores malformed JSON lines and continues.
- Request exceptions in streaming mode yield/print sanitized error chunk.

## 7) Token Management
`app/llm` does not implement token budgeting or prompt truncation.

Token-related behavior within this module:
- Anthropic branch sets `max_tokens` to `payload.get("max_tokens", 1024)`.
- Other branches rely on provider defaults unless caller explicitly includes token fields.
- Prompt length control is expected upstream (for example, core-layer prompt budgeting).

## 8) External Dependencies
Python packages:
- `requests` (HTTP transport)
- `python-dotenv` (environment loading)

Standard library:
- `os`, `json`

External services/endpoints:
- Local OpenAI-compatible server (`http://127.0.0.1:8080/v1/chat/completions`)
- OpenAI-compatible remote providers (OpenAI, Groq, Together, OpenRouter, Mistral, DeepInfra, Fireworks, Anyscale, ApiFreeLLM)
- Anthropic API
- Gemini API

Credential sources:
- Environment variables (`*_API_KEY` pattern inferred from key filename)
- Local key files (`config/*.key`)

## 9) Known Limitations
- No retry/backoff logic in HTTP transport; each request is single-attempt.
- Streaming mode is only implemented in the OpenAI-compatible branch.
- Stream/non-stream handling is provider-uneven:
  - Anthropic and Gemini paths return non-stream strings in current implementation.
- Error handling is return-value based (strings), not exception-based API contracts.
- Response parsing assumes provider response schema; schema drift can produce runtime-failure strings.

## 10) Example Invocation Flow
Example: core asks for a non-streamed answer.

1. Core constructs prompt and calls:
   `generate_answer(prompt, stream=False)`.
2. Service creates payload with model, system/user messages, and generation parameters.
3. Service calls `send_request(payload, stream=False)`.
4. Client selects active provider via `PROVIDER`.
5. Client resolves key using `load_key(...)` (env override first, then file fallback).
6. Client sends HTTP request to provider endpoint.
7. Client parses provider response and returns final text.
8. On failure, client returns sanitized provider-labeled error text instead of raising.
