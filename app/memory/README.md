# app/memory

## Module Purpose
The `app/memory` package provides all memory-layer behavior for the application:

- Embedding model initialization and reuse.
- Short-term conversation session buffering.
- Long-term conversation persistence and retrieval.
- Personal and academic vector memory storage/retrieval.

It is the boundary where in-memory state is synchronized with on-disk FAISS indexes and metadata files.

## Architectural Role in the System
`app/memory` is the persistence-backed memory subsystem used by orchestration and retrieval layers.

- `conversation_manager.py`: short-term session state and archive/recovery control.
- `conversation_memory.py`: long-term conversation memory indexing + temporal/semantic retrieval.
- `memory_system.py`: personal facts + academic domain memory indexing/search.
- `embedding_model.py`: shared embedding model loader (singleton).

## Interactions with Other System Layers

### core
- `app/core/engine.py` calls:
  - `conversation_manager` for message persistence and route continuity.
  - `memory_system` for personal fact writes and personal index maintenance.

### retrieval
- `app/retrieval/context_builder.py` calls:
  - `conversation_memory.retrieve_conversation(...)`
  - `conversation_manager.get_recent_messages(...)`
  - `memory_system` (fallback access to personal metadata)
- `app/retrieval/retriever.py` calls:
  - `memory_system.search_personal(...)`
  - `memory_system.search_academic(...)`
- `app/retrieval/ingestion/ingest_books.py` calls:
  - `memory_system.add_academic_chunks(...)`

### llm
- `conversation_memory.py` uses:
  - `app.llm.client.send_request(...)`
  - `app.llm.provider_config.MODEL_NAME`
- Scope: summary generation only (`add_session_summary_to_memory` pipeline).

### api
- `app/api/cli.py`, `app/api/main.py`, `app/api/http_api.py` import memory components for:
  - startup status visibility,
  - session persistence behavior,
  - request handling continuity.

## Data Flow (Text Diagram)
```text
User request
  -> core.engine.process_message(...)
    -> conversation_manager.add_message("user", ...)
      -> current_session.json (short-term mirror)

Route-specific processing
  -> retrieval/context_builder
    -> memory_system.search_personal / search_academic
    -> conversation_memory.retrieve_conversation

Assistant response
  -> conversation_manager.add_message("assistant", ...)
    -> current_session.json update

Archive trigger (token limit or explicit archive)
  -> conversation_manager.archive_session()
    -> conversation_backups/session_*.json
    -> conversation_memory.add_session_to_memory(...)
      -> conversation_index.faiss + conversation_meta.json
    -> conversation_memory.add_session_summary_to_memory(...)
      -> summary_index.faiss + summary_meta.json
    -> remove current_session.json

Recovery path (on module import)
  -> conversation_manager.recover_unsaved_session()
    -> conversation_backups/recovered_*.json
    -> conversation_memory.add_session_to_memory(...)
```

## Index Lifecycle

### Personal memory
- Files:
  - `personal.index`
  - `personal_meta.json`
- Managed by: `memory_system.py`
- Lifecycle:
  - Loaded at import.
  - Updated in `add_personal_fact`.
  - Rebuilt when retention cap (`MAX_PERSONAL_FACTS`) is exceeded.

### Academic memory
- Files:
  - `knowledge/<domain>/academic.index`
  - `knowledge/<domain>/academic_meta.json`
- Managed by: `memory_system.py`
- Lifecycle:
  - Loaded on demand per domain in search/store paths.
  - Created if missing during `add_academic_chunks`.

### Conversation memory
- Files:
  - `conversation_index.faiss`
  - `conversation_meta.json`
- Managed by: `conversation_memory.py`
- Lifecycle:
  - Loaded at import.
  - Rebuilt if `index.ntotal != len(metadata)`.
  - Appended during `add_session_to_memory`.

### Conversation summary memory
- Files:
  - `summary_index.faiss`
  - `summary_meta.json`
- Managed by: `conversation_memory.py`
- Lifecycle:
  - Loaded at import (empty if missing).
  - Reinitialized on embedding dimension mismatch.
  - Rebuilt when index/meta counts diverge.
  - Appended during `add_session_summary_to_memory`.

## Configuration Dependencies (.env Variables)
`app/memory` does not directly call `load_dotenv()` or `os.getenv(...)` for runtime provider values.

It has **transitive** `.env` dependency via `conversation_memory -> llm.client/provider_config` for summary generation:

- `PROVIDER`
- `MODEL_NAME`
- Provider-specific API key variables resolved by `app.llm.provider_config.load_key(...)`:
  - `OPENAI_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`, `OPENROUTER_API_KEY`,
    `MISTRAL_API_KEY`, `DEEPINFRA_API_KEY`, `FIREWORKS_API_KEY`,
    `ANYSCALE_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `APIFREELLM_API_KEY`

Additional runtime environment behavior:
- `embedding_model.get_model()` may set `CUDA_VISIBLE_DEVICES=""` to force CPU mode.

## Known Constraints
- Heavy import-time initialization:
  - Embedding model and index loading/reconciliation occur during import.
- Mixed locking model:
  - `conversation_manager` uses a lock for session state.
  - `conversation_memory` and `memory_system` mutate module globals without equivalent lock coverage.
- Partial atomicity:
  - Several write paths use temp files + `os.replace`, but index and metadata writes are still separate operations.
- Recovery and logging behavior:
  - Some session recovery/write failures are intentionally suppressed in short-term manager paths.
- Personal duplicate control:
  - `PERSONAL_DUPLICATE_THRESHOLD` is defined but not enforced in write logic.
- Temporal parsing scope:
  - Temporal retrieval only recognizes a fixed set of German phrases.

## Deterministic vs Non-Deterministic Behavior

### Deterministic (for fixed inputs/state)
- Text normalization and token counting.
- Session hash generation (`sha256` over normalized JSON serialization).
- FAISS retrieval and index rebuild logic (excluding time-dependent scoring terms).
- Session reconstruction ordering by `chunk_index`.

### Non-deterministic / time-dependent
- Summary generation (LLM output).
- Fused retrieval recency boost (`datetime.utcnow()` dependent).
- Timestamps (`time.time()`) and generated UUIDs (`uuid4()`).
- GPU/CPU selection can vary by runtime VRAM availability.

## Temporal Retrieval Behavior
Implemented in `conversation_memory._parse_time_window(...)`.

Supported phrases:
- `vorgestern`
- `gestern`
- `heute morgen`
- `heute`
- `letzte woche`

Behavior:
- Temporal match is evaluated before semantic retrieval.
- If matched, retrieval uses timestamp window reconstruction (`_retrieve_by_time_window`).
- For non-temporal queries with <= 3 tokens (`SHORT_QUERY_TOKEN_LIMIT`), retrieval returns no semantic results.

## Example Usage

### 1) Short-term message persistence and archive
```python
import app.memory.conversation_manager as cm

cm.add_message("user", "What did we discuss yesterday?")
cm.add_message("assistant", "You asked about memory retrieval.")
cm.archive_session()
```

### 2) Conversation retrieval
```python
from app.memory.conversation_memory import retrieve_conversation

results = retrieve_conversation("was haben wir gestern besprochen", top_k=3)
for session_text in results:
    print(session_text)
```

### 3) Personal memory write + query
```python
import app.memory.memory_system as ms

ms.add_personal_fact("my name is Alex")
hits = ms.search_personal("what is my name", top_k=3, return_scores=True)
print(hits)
```

### 4) Academic chunks ingestion (domain-specific)
```python
import app.memory.memory_system as ms

ms.add_academic_chunks(
    texts=["Kernel scheduling controls CPU time allocation."],
    metadata={"source": "notes.md", "type": "notes", "year": 2026},
    domain="LPIC",
)
```
