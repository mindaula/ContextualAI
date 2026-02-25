# ContextualAI

## 1. System Overview

### Purpose
ContextualAI is a modular, retrieval-grounded conversational system that combines:

1. Route-aware orchestration.
2. Memory-backed retrieval (personal, academic, conversation).
3. Prompt assembly.
4. Pluggable LLM provider calls.
5. API and CLI interfaces.

### Architectural philosophy
The system is built as layered boundaries:

1. Transport adapters (`app/api`).
2. Orchestration (`app/core/engine.py`).
3. NLP routing and rewrite (`app/nlp`).
4. Retrieval and memory (`app/retrieval`, `app/memory`).
5. Prompt construction (`app/prompting`).
6. Model invocation (`app/llm`).
7. Safety and specialized paths (`app/safety`, `app/image`).

### Modular design overview
- Each module has a focused responsibility.
- Routing decisions and prompt assembly are explicit.
- Persistence is file-backed (FAISS + JSON).
- External provider calls are isolated to dedicated adapters.

---

## 2. Architecture Overview

### Module summaries

| Module | Responsibility |
|---|---|
| `engine` | Implemented in `app/core/engine.py`; central request orchestrator and route executor. |
| `core` | Routing control flow, prompt budget enforcement, response persistence wiring. |
| `memory` | Session memory manager, long-term conversation memory, personal/academic FAISS storage. |
| `retrieval` | Route-scoped context builder, retrieval adapters, web retrieval, ingestion tooling. |
| `llm` | Provider config, prompt-to-payload adapter, provider-specific HTTP transport. |
| `api` | FastAPI OpenAI-compatible endpoints and CLI interfaces. |
| `nlp` | Intent router, structure classifier, semantic-role classifier, query rewrite. |
| `prompting` | Deterministic prompt builders (personal/academic/general templates). |
| `safety` | Rule-based lexical safety gate. |
| `image` | Text-to-image provider dispatch (`/image` command path). |

### Request lifecycle diagram (textual)

```text
Client (HTTP or CLI)
  -> app/api (validation + transport formatting)
  -> app/core/engine.process_message(...)
     -> command-first check (/image, /search)
     -> safety gate (normal text paths)
     -> query rewrite + intent decision (app/nlp)
     -> retrieval context build (app/retrieval + app/memory)
     -> prompt assembly (app/prompting + core inline prompts)
     -> LLM call (app/llm)
     -> session persistence (app/memory/conversation_manager)
  -> app/api formats final JSON or SSE response
```

### State management
State is mixed in-memory + file-backed:

1. In-memory short-term session buffer in `conversation_manager`.
2. Persistent FAISS + JSON artifacts for personal, academic, and conversation memory.
3. Domain-based academic indexes under `knowledge/<domain>/`.
4. Session backup/recovery via `current_session.json` and `conversation_backups/`.

---

## 3. Installation

### Python requirement
- Python `3.10+` is required (type-hint syntax and runtime usage match this baseline).

### Virtual environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### System dependency note (OCR)
- `pytesseract` requires the `tesseract` binary to be installed on the host OS.

### `.env` configuration

Create `.env` in repository root.

```env
# LLM provider selection
PROVIDER=local
MODEL_NAME=qwen2.5:3b

# Provider keys
OPENAI_API_KEY=
GROQ_API_KEY=
GEMINI_API_KEY=
TOGETHER_API_KEY=
OPENROUTER_API_KEY=
MISTRAL_API_KEY=
DEEPINFRA_API_KEY=
FIREWORKS_API_KEY=
ANYSCALE_API_KEY=
ANTHROPIC_API_KEY=
APIFREELLM_API_KEY=

# API debug logging gate (HTTP adapter)
DEBUG=false

# Web retrieval configuration (/search)
WEB_SEARCH_PROVIDER=brave
SEARCH_API_KEY=
WEB_TIMEOUT_SECONDS=12
WEB_MAX_RESULTS=5
WEB_MAX_CHARS=3000
WEB_USER_AGENT=contextualai/1.0
WEB_RETRY_ATTEMPTS=3
WEB_BACKOFF_SECONDS=0.5

# Image generation configuration (/image)
IMAGE_PROVIDER=ai_horde
AI_HORDE_API_KEY=
AI_HORDE_MODEL=Anything v5

# Multimodal file base directory
FILE_INPUT_BASE_DIR=uploads

# Compatibility / reserved placeholders (not consumed directly in current code)
EMBEDDING_MODEL=intfloat/multilingual-e5-small
MAX_FILE_SIZE_BYTES=10485760
MEMORY_PATH=.
VECTOR_DB_PATH=.
...
```

### Variable explanation

| Variable | Purpose | Used directly in current code |
|---|---|---|
| `PROVIDER` | Selects LLM backend branch in `app/llm/client.py`. | Yes |
| `MODEL_NAME` | Default model name sent to LLM provider payloads. | Yes |
| `OPENAI_API_KEY` etc. | Provider credentials resolved by `load_key(...)`. | Yes (provider-dependent) |
| `DEBUG` | Enables HTTP-layer sensitive debug logging when `true`. | Yes (`app/api/http_api.py`) |
| `WEB_SEARCH_PROVIDER` | Selects web provider (`brave`, `serpapi`, `tavily`). | Yes |
| `SEARCH_API_KEY` | Web search provider API key. | Yes |
| `WEB_TIMEOUT_SECONDS` | HTTP timeout for web retrieval requests. | Yes |
| `WEB_MAX_RESULTS` | URL result cap before fetch. | Yes |
| `WEB_MAX_CHARS` | Max chars in assembled untrusted web context. | Yes |
| `WEB_USER_AGENT` | User-Agent header for web requests. | Yes |
| `WEB_RETRY_ATTEMPTS` | Retry count for transient web request failures. | Yes |
| `WEB_BACKOFF_SECONDS` | Base delay for retry backoff. | Yes |
| `IMAGE_PROVIDER` | Selects image provider path in `app/image/service.py`. | Yes |
| `AI_HORDE_API_KEY` | Credential for AI Horde path. | Yes |
| `AI_HORDE_MODEL` | AI Horde model label (read in client). | Yes |
| `FILE_INPUT_BASE_DIR` | Allowed local base directory for multimodal file access. | Yes |
| `EMBEDDING_MODEL` | Placeholder for embedding model selection. | No (current code uses constant in `embedding_model.py`) |
| `MAX_FILE_SIZE_BYTES` | Placeholder for file size limit override. | No (current code uses in-module constants) |
| `MEMORY_PATH` | Placeholder for memory storage path override. | No |
| `VECTOR_DB_PATH` | Placeholder for vector DB path override. | No |

---

## 4. Running the System

### 4.1 Run via API

Start server:

```bash
uvicorn app.api.http_api:app --host 0.0.0.0 --port 8000
```

Default port:
- `8000` (uvicorn default if not overridden).

List models (domain names):
```bash
curl http://localhost:8000/v1/models
```

Chat request example:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LPIC",
    "stream": false,
    "messages": [
      {"role":"system","content":"You are a tutor."},
      {"role":"user","content":"Explain process scheduling."}
    ]
  }'
```

Expected JSON payload schema:
```json
{
  "model": "string (must match a local domain directory name)",
  "stream": "boolean (optional, default false)",
  "messages": [
    {
      "role": "user|assistant|system",
      "content": "string"
    }
  ]
}
```

Chat endpoint behavior:
1. Validates `messages` and `model`.
2. Validates model against discovered `knowledge/*` domains.
3. Applies `### Task:` tool-prompt guard.
4. Forwards latest user message to core.
5. Returns OpenAI-compatible JSON or SSE chunks.

---

### 4.2 Run via CLI (Debug Mode)

Primary interactive CLI:
```bash
python -m app.api.cli
```

Minimal CLI:
```bash
python -m app.api.main
```

Interactive usage:
1. Enter normal questions to route through core.
2. Use `exit` or `quit` to archive session.
3. Use `empty chat` or `clear chat` to discard current session.
4. In `app.api.cli`, use `/mode <domain>` to switch active domain.

Memory stats display:
- CLI prints academic chunk counts, personal fact counts, and conversation index counts at startup.

Debug logging behavior:
1. `DEBUG=true` enables HTTP adapter debug logs.
2. Core route debug prints are controlled by `DEBUG_ROUTING=True` constant in `app/core/engine.py`.
3. Semantic/router debug prints are emitted by NLP classifiers.
4. Some image-path logs include provider diagnostics.

---

## 5. Book Ingestion (Chunk Loading)

### Ingestion flow (`app/retrieval/ingestion/ingest_books.py`)

1. Resolve input mode:
   - Single file mode (`.pdf` or `.epub`).
   - Batch mode (directory or glob).
2. Extract raw text.
3. Clean text.
4. Chunk content.
5. Generate embeddings through memory layer.
6. Persist chunks + metadata into domain academic index.

### Supported formats
- Single-file document ingestion: `.pdf`, `.epub`.
- Directory/repo ingestion: source/text extensions in `SUPPORTED_CODE_EXTENSIONS` (`.py`, `.js`, `.ts`, `.java`, `.cpp`, `.c`, `.go`, `.rs`, `.md`, `.txt`, `.json`, `.yaml`, `.yml`).

### Chunk size logic
- Prose chunking: paragraph grouping with `max_words=300` (`semantic_chunk_text`).
- Code chunking: fixed windows of `max_lines=80` (`semantic_chunk_code`).

### Embedding generation
- `memory_system.embed_batch(...)` creates normalized vectors (`passage:` prefix + FAISS L2 normalization).

### Storage location
- `knowledge/<domain>/academic.index`
- `knowledge/<domain>/academic_meta.json`

### Retrieval access path
- `app/retrieval/retriever.py -> memory_system.search_academic(...)`
- Used by `context_builder` for academic route context.

### Ingestion command examples

Single file:
```bash
python -m app.retrieval.ingestion.ingest_books ./books/os.pdf \
  --type book \
  --source "Operating Systems Notes" \
  --year 2024 \
  --domain LPIC
```

Batch directory:
```bash
python -m app.retrieval.ingestion.ingest_books ./books --domain LPIC
```

Batch mode requirement:
- `metadata.txt` must exist in target directory.
- Format per line: `filename | key=value | key=value`.

### Positional argument behavior (`filepath`)
The first positional argument is the ingestion input spec. Internally it is read as `path_spec` and interpreted in exactly one of three ways:

1. File path:
   - Example: `./books/os.pdf`
   - Treated as single-file mode.
   - `--type` is required.
   - Only `.pdf` and `.epub` are accepted in this path.

2. Directory path:
   - Example: `./books`
   - Treated as batch mode.
   - Files are enumerated from that directory level.
   - `metadata.txt` in that directory is mandatory.
   - `--type`, `--source`, and `--year` must not be passed in batch mode.

3. Glob/wildcard pattern:
   - Example: `./books/*.pdf`
   - Treated as batch mode.
   - File list is resolved with `glob.glob(...)` in sorted order.
   - `metadata.txt` is required in the parent directory of the glob.

Argument cardinality behavior:
- The CLI accepts exactly one positional path spec. If more than one is passed, it fails with `argparse` validation error.

### CLI parameter details

#### `--type`
- Required in single-file mode.
- Disallowed in batch mode.
- Stored as metadata field `type` and forwarded into `memory_system.add_academic_chunks(...)`.
- Typical values are `book`, `repo`, `notes`, but the current code does not enforce a closed enum at this layer.

#### `--source`
- Optional in single-file mode.
- Disallowed in batch mode.
- If omitted, single-file ingestion defaults to the input filename.
- Stored as metadata field `source` and propagated to retrieval output and citation prompt assembly.

#### `--year`
- Optional integer in single-file mode.
- Disallowed in batch mode.
- Normalized by `normalize_year(...)`; invalid/empty values fall back to current year.
- Stored as metadata field `year` and used later by academic scoring (`search_academic` recency bonus).

#### `--domain`
- Required in all modes.
- Maps directly to `knowledge/<domain>/`.
- If `knowledge/<domain>/` does not exist, ingestion creates it before writing indexes.
- Controls the storage and retrieval namespace for academic chunks.

### Domain mapping and non-existing domains
- Domain value is used as a folder key under `knowledge/`.
- Index artifacts are domain-scoped:
  - `knowledge/<domain>/academic.index`
  - `knowledge/<domain>/academic_meta.json`
- Missing domain directories are created automatically in ingestion.
- Retrieval uses the same domain key when `process_message(..., domain=...)` calls academic retrieval.

### Wildcard (`*`) usage in batch mode
Wildcard input allows file-subset ingestion without manual listing.

Example:
```bash
python -m app.retrieval.ingestion.ingest_books ./books/*.pdf --domain LPIC
```

Behavior:
1. Pattern is expanded by `glob.glob(...)`.
2. Only files are kept.
3. Parent directory is used as batch base directory.
4. `metadata.txt` is loaded from that base directory.
5. Each matched file must have a metadata entry by filename.

If a matched file has no metadata entry:
- The CLI prints `No metadata entry found for <filename>`.
- That file is skipped.
- Processing continues with remaining files.

### `metadata.txt` format and parsing behavior
In batch mode, `metadata.txt` is required and parsed by `load_metadata_file_from_path(...)`.

Required line shape:
```text
<filename> | key=value | key=value | ...
```

Parsing rules:
1. Blank lines are ignored.
2. Lines without `|` are ignored.
3. First segment becomes the filename key.
4. Remaining segments are parsed only if they contain `=`.
5. Duplicate filename entries overwrite earlier entries.

Example `metadata.txt`:
```text
os_book.pdf | source=Operating Systems Vol.1 | type=book | year=2022
networks.pdf | source=Networking Notes | type=notes | year=2024
kernel.epub | source=Kernel Handbook | type=book | year=2021
```

Missing metadata behavior:
- If `metadata.txt` is missing in batch mode, ingestion aborts with exit code `1`.
- If a specific file has no entry, that file is skipped (non-fatal for whole batch).

Metadata influence on retrieval context:
- `source` is surfaced in retrieval hits and prompt citations.
- `year` participates in academic score boosting (`(year - 2000) * 0.002`).
- `type` is persisted and available in metadata, though not directly used in current ranking formula.
- `ingested_at` is written during ingestion and can be used as date fallback in citation normalization.

### Embedding, chunk indexing, and FAISS update behavior
Internal write path (`memory_system.add_academic_chunks`) is append-oriented:

1. Load existing domain index/metadata if present; otherwise initialize empty index/list.
2. Embed all chunks via `embed_batch(...)`.
3. Add vectors to FAISS `IndexFlatIP`.
4. Append one metadata record per chunk (`text` + merged metadata fields).
5. Write updated FAISS index to `academic.index`.
6. Write updated metadata JSON to `academic_meta.json`.

Operational implications:
- Chunk order is preserved by append sequence.
- New ingestion runs extend existing domain memory rather than replacing it.
- There is no deduplication pass in this ingestion path.

---

## 6. Chat Slash Options

### `/memory <query>`
- Purpose: force conversation-memory retrieval path.
- Pipeline effect: `retrieval.context_builder` overrides route to `conversation_query`.
- NLP/core interaction: override occurs at retrieval-context stage, after intent computation.

Detailed pipeline behavior:
1. Core still executes normal safety check first (`is_allowed(question)`).
2. NLP rewrite + intent routing still run.
3. Route override happens inside `build_memory_context(...)` when question starts with `/memory`.
4. Retrieval is forced to conversation retrieval (`retrieve_conversation(cleaned_question, top_k=3)`).
5. Core executes `conversation_query` branch and builds conversation-history prompt.
6. User/assistant messages are persisted as usual.

Bypass/force semantics:
- NLP routing is influenced but not bypassed.
- Retrieval is forced to conversation memory for this turn.
- Memory persistence remains enabled.

Example flow:
- Input: `/memory what did we discuss yesterday?`
- Effective internal route: `conversation_query`
- Retrieval source: long-term conversation memory (plus normal conversation path behavior).


### `/academic <question>`
- Purpose: explicit academic route hint.
- Pipeline effect: `nlp.intent_router` hard-prefix handling sets fallback route to `academic` and strips prefix.
- NLP/core interaction: core consumes router decision and proceeds with academic retrieval/prompt logic.

Detailed pipeline behavior:
1. Core runs safety gate on raw input first.
2. Query rewrite runs before `decide_route(...)`.
3. `decide_route(...)` detects `/academic` prefix and sets `decision.fallback = "academic"` with `cleaned_question`.
4. Core maps intent to `academic` and calls `build_memory_context(...)`.
5. Academic retrieval is attempted (`retrieve_academic(...)`), capped to top hits.
6. If hits exist, citation-constrained academic prompt is used; otherwise general prompt fallback is used.
7. User/assistant persistence still occurs.

Bypass/force semantics:
- NLP routing is influenced by hard prefix inside NLP module.
- Academic retrieval is requested for this route (forced attempt), but may return no hits.
- Safety gate runs before route execution.
- Memory persistence remains enabled.

Example flow:
- Input: `/academic Explain process scheduling.`
- Effective internal route: `academic`
- Retrieval source: domain-scoped academic index.

### `/general <question>`
- Purpose: explicit general route hint.
- Pipeline effect: `nlp.intent_router` hard-prefix handling sets fallback route to `general` and strips prefix.
- NLP/core interaction: core follows general prompt path.

Detailed pipeline behavior:
1. Core safety check runs first.
2. Query rewrite runs.
3. `decide_route(...)` detects `/general` prefix and sets `fallback = "general"` with `cleaned_question`.
4. `build_memory_context(...)` executes with `general` intent (no forced retrieval branch).
5. Core falls through to general prompt generation.
6. User/assistant persistence still occurs.

Bypass/force semantics:
- NLP routing is influenced via explicit hard prefix.
- Retrieval is optional/non-forced for this route in current context builder logic.
- Safety gate runs before route execution.
- Memory persistence remains enabled.

Example flow:
- Input: `/general What is TCP?`
- Effective internal route: `general`
- Retrieval source: none required for route execution.

### `/search <query>`
- Purpose: manual web search route.
- Pipeline effect: command-first handling in core, web retrieval module invoked, untrusted context prompt built.
- NLP/core interaction: bypasses normal intent route selection for that turn.

Detailed pipeline behavior:
1. Core checks `/search` prefix before normal text routing.
2. `_handle_search_command(...)` runs immediately:
   - validates non-empty query,
   - runs safety check on query,
   - initializes/uses web module,
   - retrieves web context.
3. Because search payload exists, core skips standard `decide_route(...)` path.
4. Effective intent is set to `manual_web_search`.
5. Core builds untrusted web-context prompt and calls LLM.
6. User/assistant persistence still occurs after route handling.

Bypass/force semantics:
- NLP intent routing is bypassed in initial command path.
- Web retrieval is forced attempt for this command.
- Safety gate runs inside search handler before retrieval/provider call.
- Memory persistence remains enabled.

Example flow:
- Input: `/search Linux cgroups overview`
- Effective internal route: `manual_web_search`
- Retrieval source: web module (`app/retrieval/web/web_module.py`).

### `/image <prompt>`
- Purpose: text-to-image generation.
- Pipeline effect: command-first handling in core, direct call to image service, returns image URL payload.
- NLP/core interaction: bypasses NLP routing/retrieval/prompting flow for text generation.

Detailed pipeline behavior:
1. Core checks `/image` before search, safety, rewrite, and intent routing.
2. Prompt suffix is extracted and forwarded to `generate_image(...)`.
3. Provider-specific image client executes and returns image payload (typically `{"image_url": ...}`).
4. Core returns immediately from `/image` branch.
5. HTTP adapter converts image payload to markdown image content for chat response format.

Bypass/force semantics:
- NLP routing is bypassed.
- Text retrieval is bypassed.
- Normal text safety gate in `process_message` is not reached for this path.
- Standard conversation message persistence in core is not executed in this early-return branch.

Example flow:
- Input: `/image draw a kernel scheduler diagram`
- Effective internal route: command-first image path (no text route assignment)
- Retrieval source: none; image provider only.

### `/mode <domain>` (CLI only)
- Purpose: switch active domain in full CLI adapter.
- Pipeline effect: local CLI state update; affects `domain` argument passed into core.
- NLP/core interaction: no NLP route change directly; domain influences academic retrieval scope.

---

## 7. OpenWebUI Compatibility

### Why schema is compatible
`app/api/http_api.py` exposes:
1. `GET /v1/models`
2. `POST /v1/chat/completions`

Response envelopes and streaming chunks follow OpenAI-style shapes (`chat.completion`, `chat.completion.chunk`, `[DONE]`).

### Connection settings (OpenWebUI)
- Base URL: `http://<host>:8000/v1`
- Endpoint path: use OpenAI-compatible chat completion path.
- API key: server does not enforce built-in HTTP auth in current implementation.

### Model naming expectation
- `model` must match a local domain directory name under `knowledge/`.
- Example: if `knowledge/LPIC/` exists, use `"model": "LPIC"`.

### Important behavior
- HTTP adapter forwards only the latest user message from `messages` to core.
- Session continuity is primarily server-side via conversation memory, not full client history replay.

---

## 8. Multimodal Usage (if enabled)

### File-processing flow
`app/api/multimodal/file_input_manager.py` supports:
1. `data:` URLs (Base64 payloads).
2. `file://` URLs (only empty host or `localhost`).
3. Local paths inside allowed base directory.

### Base64 handling
1. Approx decoded size is checked before decode/write.
2. Payload is decoded into a temporary file under allowed base dir.
3. File is validated and extracted by extension.

### Size limits
- Current code uses:
  - `MAX_FILE_SIZE_MB = 10`
  - `MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024`
- Extension allowlist includes image, PDF, TXT, CSV, DOCX.

### Temporary file lifecycle
- Base64 temp files are removed in `finally`, including error paths.

### Image flow
- `/image` command triggers `app/image` provider adapters.
- API layer converts `{"image_url": ...}` to markdown image content for chat response formatting.

### Note on API exposure
- Core supports `file_inputs`, but current `/v1/chat/completions` handler does not expose a dedicated file field; multimodal integration is available to callers that pass `file_inputs` into core directly.

---

## 9. Debugging

### `DEBUG=true` behavior
- Enables HTTP adapter debug prints of request/response flow and streaming details.

### Logging implications
- Debug output can include message content and response chunks.
- Additional always-on diagnostic prints exist in core/NLP routing and some provider paths.

### CLI vs API debugging
1. API debug is env-gated by `DEBUG`.
2. CLI prints operational status by default.
3. Core route debug currently runs with constant-enabled prints.
4. NLP semantic/router debug prints are emitted during classification.

### Common failure cases
1. `{"error":"Unknown model requested"}` when `model` does not match a knowledge domain.
2. `{"error":"No domains available."}` when `knowledge/` has no domain folders.
3. `"This request violates safety policies."` from lexical safety gate.
4. `"LLM error."` from core safe generation wrapper.
5. `"File processing error."` when multimodal preprocessing fails.
6. `/search` errors for missing/invalid web module config or key.
7. Image provider failures for missing keys, HTTP errors, or provider-side faults.

---

## 10. Security Considerations

### Prompt injection
1. Web retrieval content is marked untrusted and sanitized for known patterns.
2. Memory/retrieved text is still injected as plain text into prompts, so instruction contamination remains a risk boundary.

### Memory poisoning
1. Ingested academic documents and stored personal facts become retrieval context.
2. Low-quality or malicious content in stored memory can influence future answers.

### File upload / file input risks
1. Path traversal is restricted by base-directory checks.
2. Allowed extension and size checks reduce attack surface.
3. Extraction libraries still process untrusted content; parser-level risks remain.

### Debug exposure
1. Debug logs can expose user content and intermediate outputs.
2. Some image-path logs print sensitive data (provider diagnostics).
3. Run production deployments with conservative logging and strict environment handling.

### HTTP access control
- Current HTTP layer does not implement built-in authentication/authorization.

---

## 11. Known Limitations

1. Prompt budget enforcement is heuristic (`chars -> token estimate`) and approximate.
2. Retrieval quality depends on chunking quality, metadata consistency, and embedding behavior.
3. Determinism is bounded; LLM outputs, web retrieval, OCR, and time-based ranking are non-deterministic.
4. Concurrency is limited by global mutable module state and some synchronous network calls in async-adjacent paths.
5. HTTP adapter forwards only latest user turn from `messages`.
6. `/image` command path executes before normal text safety gate.
7. Web and image provider paths rely on external service availability.

---

## 12. Minimal Quick Start (TL;DR)

1. Clone repository.
   ```bash
   git clone github.com/mindaula/ContextualAI
   cd ContextualAI
   ```

2. Create and activate virtual environment.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```

4. Configure `.env` (provider, model, keys, debug flags).

5. Start API server.
   ```bash
   uvicorn app.api.http_api:app --host 0.0.0.0 --port 8000
   ```

6. Ingest at least one domain dataset.
   ```bash
   python -m app.retrieval.ingestion.ingest_books ./books/os.pdf --type book --domain LPIC
   ```

7. Chat through API.
   ```bash
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"LPIC","stream":false,"messages":[{"role":"user","content":"What is process scheduling?"}]}'
   ```

## 13. Using This System with OpenWebUI

OpenWebUI is an OpenAI-compatible web interface for chat-based model interaction.
Official repository: https://github.com/open-webui/open-webui

In this deployment, Docker runs OpenWebUI only as the frontend UI. ContextualAI remains the backend system that executes routing, retrieval, prompting, and model calls.

Architecture mapping:

```text
ContextualAI (Backend API, port 8000)
       v
OpenWebUI (Docker container, port 3000)
       v
Browser / Mobile App
```

This system is compatible with OpenWebUI because:
- It exposes OpenAI-style endpoints.
- It provides `GET /v1/models`.
- It provides `POST /v1/chat/completions`.
- The `model` value is expected to match a local domain name under `knowledge/<domain>/`.
- The API layer does not enforce built-in authentication.

OpenWebUI sends requests to this backend endpoint:
- `http://<server-host>:8000/v1/chat/completions`

OpenWebUI receives responses from this system at that endpoint. It is not directly connecting to OpenAI servers and it is not using OpenAI-hosted models by default in this setup.

Configuration implications:
- `OPENAI_API_BASE_URL` must point to this ContextualAI backend (`http://<server-host>:8000/v1`).
- `OPENAI_API_KEY` is accepted by OpenWebUI but ignored by this backend because no built-in auth enforcement is implemented.
- No traffic goes to OpenAI unless the backend is explicitly configured to do so.

Provider behavior note:
If `PROVIDER=openai` is configured in `.env`, this backend may call OpenAI internally after receiving requests from OpenWebUI. Otherwise, inference stays local or is sent only to whichever provider is configured in ContextualAI.

### Step 1 - Start ContextualAI API

Start the API server:

```bash
uvicorn app.api.http_api:app --host 0.0.0.0 --port 8000
```

Notes:
- Default port is `8000`.
- Base URL for OpenAI-compatible access is `http://<server-host>:8000/v1`.

### Step 2 - Run OpenWebUI via Docker

Example:

```bash
docker run -d \
  -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=dummy \
  --name openwebui \
  ghcr.io/open-webui/open-webui:main
```

Notes:
- `OPENAI_API_KEY` can be a dummy value because this server does not enforce API-key auth by default.
- `OPENAI_API_BASE_URL` points OpenWebUI to this backend, not to OpenAI.
- `host.docker.internal` allows the container to reach the host machine on Docker Desktop environments.
- On Linux, `host.docker.internal` may not resolve; use the host IP address instead.

### Model Configuration in OpenWebUI

Set the following in OpenWebUI:
- Base URL: `http://<server-host>:8000/v1`
- API key: any placeholder value (for example `dummy`)
- Model: must match a directory name in `knowledge/<domain>/`

Model naming rule:
- The model name is the domain folder name, not an OpenAI model ID.
- Example: if the folder is `knowledge/LPIC/`, set model to `LPIC`.

Example model value:
- `LPIC`

### Mobile Usage

OpenWebUI supports mobile browser access.

Typical access URL from phone on the same network:
- `http://<server-ip>:3000`

Deployment notes:
- For public deployment, use HTTPS.
- Place OpenWebUI behind a reverse proxy.
- The interface behaves as a ChatGPT-style chat UI on mobile browsers.

### Optional Production Note

If exposed outside a trusted local network:
- Use a reverse proxy such as `nginx` or `traefik`.
- Enforce HTTPS.
- Add authentication and access control at the proxy or gateway layer.
