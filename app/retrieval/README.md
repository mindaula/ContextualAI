# app/retrieval

## 1) Module Purpose
`app/retrieval` provides retrieval-time context assembly and ingestion support for the application.

Primary responsibilities:
- Build route-scoped retrieval context for `core` request handling.
- Adapt memory-layer search outputs into normalized retrieval payloads.
- Retrieve and sanitize untrusted web context for manual web-search flows.
- Ingest external documents/repositories into academic memory for later retrieval.

## 2) Architectural Role in the Overall System
`app/retrieval` is the retrieval boundary between orchestration (`app/core`) and persistence-backed memory (`app/memory`).

- Upstream caller: `app.core.engine.process_message(...)`.
- Downstream dependencies: memory retrieval and write APIs (`memory_system`, `conversation_memory`, `conversation_manager`).
- Functional split:
  - `context_builder.py`: route-isolated retrieval context composition.
  - `retriever.py`: personal/academic retrieval adapters.
  - `web/web_module.py`: external web retrieval and sanitization.
  - `ingestion/ingest_books.py`: offline ingestion into academic memory.

## 3) Retrieval Strategies Explained
### Route-isolated context retrieval (`context_builder`)
- `academic` route:
  - Semantic retrieval through `retrieve_academic(...)`.
- `personal_query` route:
  - Semantic retrieval through `retrieve_personal(...)`.
  - Deterministic fallback: expose stored personal facts with synthetic `score=1.0` if semantic retrieval returns no hits and personal index is non-empty.
- `conversation_query` route:
  - Hybrid merge of:
    - long-term semantic conversation retrieval (`conversation_memory.retrieve_conversation(...)`), and
    - short-term recent session messages (`conversation_manager.get_recent_messages(...)`).
- `/memory ...` command:
  - Hard override to `conversation_query` path.

### Web retrieval strategy (`web_module`)
- Provider search (Brave / SerpAPI / Tavily) -> URL extraction -> HTTP fetch -> text extraction (`trafilatura`) -> sanitization -> bounded context assembly.
- Content is explicitly marked untrusted before return.

### Ingestion strategy (`ingest_books`)
- Document/repository parsing -> cleaning -> chunking -> `memory_system.add_academic_chunks(...)`.
- Ingestion affects future retrieval quality but does not perform retrieval itself.

## 4) Ranking and Scoring Logic
`app/retrieval` itself does not compute embedding similarity scores. Ranking is delegated to memory-layer search functions.

### In-module ordering/capping
- `academic`: hard cap `[:5]`.
- `personal`: hard cap `[:5]`.
- `conversation`: merge order preserved (`semantic_hits + formatted_recent`), deduplicated by first occurrence, hard cap `[:6]`.
- Web URLs: provider order preserved, deduplicated, capped by `max_results`.

### Delegated scoring formulas (memory layer)
- Personal memory (`memory_system.search_personal`):
  - Similarity threshold: `PERSONAL_MIN_SIMILARITY = 0.60`.
  - Relative filtering: keep hits `score >= top_score * RELATIVE_SCORE_RATIO` with `RELATIVE_SCORE_RATIO = 0.90`.
- Academic memory (`memory_system.search_academic`):
  - Minimum tokens guard: `MIN_QUERY_TOKENS_FOR_ACADEMIC = 2`.
  - Boosted score: `boosted = similarity + (year - 2000) * 0.002`.
  - Threshold: `ACADEMIC_MIN_SIMILARITY = 0.72`.
  - Relative filtering with same ratio `0.90`.
- Conversation memory (`conversation_memory.retrieve_conversation`, fused mode):
  - Fused score: `0.6 * raw + 0.3 * summary + 0.1 * recency`.
  - Absolute minimum filter: `MIN_SCORE_ABSOLUTE = 0.48`.

## 5) Data Flow Diagram (Text-Based)
```text
core.engine.process_message(...)
  -> retrieval.context_builder.build_memory_context(question, intent, confidence, domain)

    if intent == academic:
      -> retrieval.retriever.retrieve_academic(...)
        -> memory.memory_system.search_academic(...)
        -> academic hits (scored/filtered in memory layer)

    if intent == personal_query:
      -> retrieval.retriever.retrieve_personal(...)
        -> memory.memory_system.search_personal(...)
        -> personal hits (scored/filtered in memory layer)
      -> if empty: fallback to memory_system.personal_meta exposure

    if intent == conversation_query or /memory ...:
      -> memory.conversation_memory.retrieve_conversation(...)
      -> memory.conversation_manager.get_recent_messages(...)
      -> merge + dedup + cap

  -> context payload returned to core
  -> core builds prompt and calls llm service

Manual web-search path (core):
  -> retrieval.web.web_module.WebSearchModule.aretrieve_context(query)
  -> provider search -> URL parse -> async fetch -> extract/clean -> bounded context
  -> context returned to core prompt builder
```

## 6) Interaction with Memory Indexes
`app/retrieval` interacts with memory indexes indirectly through memory APIs.

Read paths:
- Personal memory:
  - `personal.index`, `personal_meta.json` via `memory_system.search_personal(...)`.
- Academic memory:
  - `knowledge/<domain>/academic.index`, `knowledge/<domain>/academic_meta.json` via `memory_system.search_academic(...)`.
- Conversation memory:
  - `conversation_index.faiss`, `conversation_meta.json`, plus optional summary index via `conversation_memory.retrieve_conversation(...)`.
- Short-term session context:
  - `conversation_manager.get_recent_messages(...)` (in-memory/session-log-backed manager).

Write paths:
- Ingestion writes academic chunks through `memory_system.add_academic_chunks(...)`.
- Retrieval runtime modules (`context_builder`, `retriever`, `web_module`) do not write FAISS indexes directly.

## 7) Determinism Analysis
Deterministic for fixed state/input:
- Route-isolated branching in `build_memory_context`.
- Context caps and dedup rules.
- Adapter formatting in `retriever`.
- Cleaning/chunking logic in ingestion.
- URL filtering and truncation rules in web module.

Non-deterministic / time-varying:
- Provider-ranked web results and network response variability.
- Remote page content changes.
- Underlying memory retrieval outcomes when index contents change over time.
- Conversation retrieval recency-sensitive fused scoring (delegated to memory layer).

## 8) Known Limitations
- No cross-route blending by design:
  - Context builder intentionally does not combine academic/personal/conversation contexts outside route rules.
- Personal fallback may be broad:
  - On semantic miss, personal route can expose all stored personal facts (capped to first five after ordering in metadata list).
- Web retrieval failure sensitivity:
  - In async context retrieval, any raised task exception is re-raised; partial successful chunks are not returned in that error path.
- No semantic reranking in web module:
  - Provider order is used; retrieved page text is not vector-scored inside `app/retrieval`.
- Batch ingestion memory usage:
  - `ingest_directory` accumulates all chunks before writing, which can be heavy on large repositories.
- Sync wrappers in web module:
  - `search(...)`/`retrieve_context(...)` call `asyncio.run`, which is unsuitable from an already-running event loop context.

## 9) Performance Notes
- Retrieval-time caps bound payload size:
  - academic <= 5, personal <= 5, conversation <= 6.
- Conversation short-term fetch is small (`limit=4`) to keep prompt context bounded.
- Web retrieval uses concurrent fetch (`asyncio.gather`) for throughput.
- Web text assembly enforces `max_chars` budget to avoid unbounded prompt growth.
- Ingestion performance drivers:
  - Parser cost (`fitz`, `ebooklib`), file count, and chunk volume.
  - Directory ingestion scans recursively and reads files into memory.

## 10) Example Retrieval Flow
Example: academic question handled by core.

1. `core.engine` calls:
   `build_memory_context("What is process scheduling?", "academic", confidence, domain="LPIC")`.
2. `context_builder` routes to `retrieve_academic(...)`.
3. `retriever` calls `memory_system.search_academic(..., return_scores=True, domain="LPIC")`.
4. Memory layer applies semantic thresholding and score filtering, returns scored entries.
5. `retriever` normalizes entries (`text`, `source`, `year`, `score`).
6. `context_builder` caps to top 5 and returns payload to `core`.
7. `core` builds academic prompt from retrieved context and generates final answer via LLM.
