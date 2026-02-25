# app/core

## 1) Module Purpose
`app/core` is the orchestration layer that turns an incoming user message into a routed response.

Its primary responsibilities are:
- Command-first handling (`/image`, `/search`).
- Intent-aware routing for standard text requests.
- Retrieval/context assembly integration.
- Prompt construction and LLM invocation.
- Session persistence coordination (user/assistant messages, route continuity).

## 2) Architectural Role in the Overall System
`app/core` sits between API/CLI entrypoints and lower-level subsystems.

- Upstream callers: API and CLI layers invoke `process_message(...)`.
- Downstream dependencies: memory, retrieval, NLP routing/rewriting, prompting, and LLM adapter modules.
- Main entrypoint: `engine.process_message`.
- Routing contract type: `routing_types.RoutingDecision`.

## 3) Request Processing Pipeline
The request pipeline in `engine.process_message` is:

1. Input guard:
- Empty input returns `""`.

2. Command-first routing:
- `/image ...` returns `generate_image(...)` directly.
- `/search ...` is parsed and handled before standard intent routing.

3. Multimodal branch (optional):
- If `file_inputs` are present and request is not `/search`, file content is extracted via `handle_files(...)` and sent to LLM with a file-only prompt.

4. Standard text branch:
- Safety check via `is_allowed(...)`.
- Previous route read from `conversation_manager.get_last_route()`.
- Query rewrite via `rewrite_query(...)`.
- Intent decision via `decide_route(...)`.
- Intent flags mapped to concrete route string.

5. Retrieval context assembly:
- `build_memory_context(rewritten_question, intent, confidence, domain=...)`.
- Effective route may be overridden by context builder output.

6. Route-specific execution:
- `followup_transform`, `academic`, `manual_web_search`, `personal_store`, `personal_query`, `conversation_query`, or fallback `general`.

7. Output + persistence:
- User message and final route are persisted.
- Assistant response is persisted immediately (non-stream) or on stream completion (streaming wrappers).

## 4) Routing System Explanation
Routing is two-stage:

1. Intent stage:
- `decide_route(...)` returns a `RoutingDecision` with boolean flags and fallback.
- `engine` maps flags to route names in fixed priority order:
  - `store_personal_fact` -> `personal_store`
  - `use_long_term_memory` -> `personal_query`
  - `use_academic_chunks` -> `academic`
  - `use_short_term_memory` -> `conversation_query`
  - `followup_transform` -> `followup_transform`
  - otherwise `fallback` or `general`

2. Context stage:
- `build_memory_context(...)` returns route + scoped context payload.
- This stage can override route (for example `/memory ...` forcing `conversation_query`).

Additional behavior:
- Manual web search continuity is preserved on referential follow-up requests.
- Route strings are shared contracts across modules (`academic`, `personal_query`, etc.).

## 5) Interaction with Other Modules
### memory
- Uses `app.memory.conversation_manager` for:
  - `add_message(...)`
  - `set_last_route(...)`
  - `get_last_route(...)`
  - `get_last_assistant_message(...)`
- Uses `app.memory.memory_system` for personal-memory writes (`add_personal_fact(...)`).
- In `personal_store`, engine also accesses `memory_system` internals (`personal_meta`, `personal_index`, FAISS rebuild/write utilities).

### retrieval
- Uses `app.retrieval.context_builder.build_memory_context(...)` as the retrieval-context gateway.
- Uses optional web retrieval module (`WebSearchModule`) for `/search` path through `_handle_search_command(...)`.

### llm
- Uses `app.llm.service.generate_answer(...)` via `safe_generate(...)`.
- Applies prompt-token budget guard before invoking provider call.
- Handles both stream and non-stream responses.

### api
- API/CLI modules call `engine.process_message(...)` as the orchestration boundary.
- `app/core` does not expose transport-layer logic; it returns response objects/streams for API/CLI adapters to render.

## 6) Control Flow Diagram (Text-Based)
```text
API/CLI request
  -> core.engine.process_message(question, domain, file_inputs, web_module)

    -> empty? return ""
    -> /image ? generate_image(...) -> return
    -> /search ? _handle_search_command(...) -> search payload

    -> file_inputs and not /search ?
         handle_files(...) -> safe_generate(multimodal prompt)
         persist user+route, then assistant (immediate or stream wrapper)
         -> return

    -> safety check (is_allowed)
    -> last_route from conversation_manager
    -> rewrite_query(...)
    -> decide_route(...)
    -> intent -> route mapping
    -> build_memory_context(...)
    -> effective route = memory_context.route

    -> switch(route):
         followup_transform -> transform prompt -> safe_generate
         academic -> citation prompt/general prompt -> safe_generate
         manual_web_search -> web context prompt -> safe_generate
         personal_store -> slot replacement + add_personal_fact
         personal_query -> build_personal_prompt -> safe_generate
         conversation_query -> conversation-history prompt -> safe_generate
         general -> build_general_prompt -> safe_generate

    -> persist user message + last route
    -> if streaming: wrap stream to persist assistant on completion
    -> else persist assistant text and return response
```

## 7) Determinism Analysis
Deterministic for fixed input/state:
- Command parsing and branch order.
- Intent-flag to route mapping in `engine`.
- Prompt template assembly.
- Academic citation suffix formatting and deduplication.
- Stream wrapping behavior (given same stream consumption behavior).

Non-deterministic or state/time-dependent:
- Query rewriting and generated responses (LLM/provider behavior).
- Web retrieval content.
- Multimodal extraction variability (OCR and parsing outcomes).
- Route decisions influenced by mutable prior session state (`last_route`, recent messages).
- Any downstream retrieval components with time-based ranking.

## 8) Failure Handling
Primary failure patterns in `engine`:

- Empty input:
  - Returns `""`.

- Safety failure:
  - Returns policy message (`"This request violates safety policies."`).

- Multimodal processing failure:
  - Catches exception, logs `logger.exception(...)`, returns `"File processing error."`.

- LLM failure:
  - `safe_generate` catches provider exceptions and returns `"LLM error."`.

- Search command errors:
  - Returns explicit payload errors (`usage`, `policy`, `web module not configured`).

- Streaming persistence caveat:
  - Assistant message persistence in wrapped streams occurs only after stream is fully consumed.

## 9) Known Architectural Constraints
- String-based route contracts:
  - Route names are shared across modules as plain strings.

- Context-stage route override:
  - `build_memory_context(...)` can replace route chosen by intent stage.

- Internal coupling to memory implementation:
  - `personal_store` directly manipulates `memory_system` globals and FAISS write paths.

- Dynamic routing decision fields:
  - `engine` checks `hasattr(decision, "cleaned_question")`, but this is not defined in `RoutingDecision` dataclass.

- Streaming completion dependency:
  - Persistence of assistant text for streaming responses depends on caller draining stream.

- Mixed prompt identity layering:
  - Prompt builder text and LLM service system message are composed in separate layers; consistency depends on both modules.

## 10) Example Request Lifecycle
Example: user asks an academic question from CLI/API.

1. API/CLI calls `process_message("Explain process scheduling", domain="LPIC")`.
2. Request is not `/image` or `/search`; no multimodal branch.
3. Safety filter allows input.
4. Query may be rewritten for context clarity.
5. Intent router marks academic path.
6. `build_memory_context(...)` fetches academic hits and returns route/context.
7. Engine builds citation-constrained academic prompt.
8. `safe_generate(...)` invokes LLM provider.
9. Engine writes user message and selected route to `conversation_manager`.
10. Engine returns streaming or non-stream response; assistant text is persisted when complete.
