"""Long-term conversation memory indexing and retrieval.

Purpose:
- Persist archived chat sessions as vectorized conversation memory.
- Maintain a parallel summary index for fused retrieval ranking.
- Serve semantic and temporal lookups for conversation recall requests.

Data flow:
1. On import, load embedding model, FAISS indexes, and JSON metadata.
2. Validate index/metadata consistency and rebuild indexes when mismatched.
3. `add_session_to_memory` writes raw conversation chunks (FAISS + metadata).
4. `add_session_summary_to_memory` writes one summary vector per session hash.
5. `retrieve_conversation` resolves temporal expressions first, otherwise performs
   fused semantic retrieval over raw and summary indexes.

Index lifecycle:
- `conversation_index.faiss` and `conversation_meta.json` are loaded at import and
  reconciled by count (`index.ntotal == len(metadata)`), with full index rebuild when
  inconsistent.
- `summary_index.faiss` and `summary_meta.json` are loaded at import, rebuilt on
  dimension mismatch or count mismatch, and filtered to valid summary text entries.
- Writes use temporary files and `os.replace` for atomic replacement of individual
  artifacts.

Temporal retrieval logic:
- Queries matching supported German temporal phrases are converted into UTC timestamp
  windows and answered via timestamp-filtered session reconstruction.
- Without a temporal match, retrieval uses FAISS semantic search with thresholding and
  optional fused scoring (raw score + summary score + recency boost).

External dependencies:
- FAISS (`faiss`) for vector index storage and nearest-neighbor search.
- JSON (`json`) for metadata persistence.
- Datetime (`datetime`, `timedelta`) for temporal window parsing and recency scoring.
- Embeddings via `app.memory.embedding_model.get_model`.
- Summary generation via `app.llm.client.send_request`.

Side effects:
- Reads/writes local index and metadata files.
- Rebuilds indexes when file state is inconsistent.
- Logs rebuild and persistence failures.
- Uses in-memory module globals (`index`, `summary_index`, `metadata`,
  `summary_metadata`) that are mutated by write paths.
"""

import os
import json
import time
import hashlib
import logging
import faiss
import numpy as np
import uuid
from datetime import datetime, timedelta

from app.memory.embedding_model import get_model
from app.llm.client import send_request
from app.llm.provider_config import MODEL_NAME


logger = logging.getLogger(__name__)


INDEX_PATH = "conversation_index.faiss"
META_PATH = "conversation_meta.json"
SUMMARY_INDEX_PATH = "summary_index.faiss"
SUMMARY_META_PATH = "summary_meta.json"
SESSION_LOG_PATH = "current_session.json"

MIN_SCORE_ABSOLUTE = 0.48
SHORT_QUERY_TOKEN_LIMIT = 3
CHUNK_SIZE = 4
SUMMARY_MIN_TOKENS = 500
SUMMARY_MAX_TOKENS = 800
SUMMARY_PROMPT_TOKEN_BUDGET = 3500
SUMMARY_CHARS_PER_TOKEN_ESTIMATE = 4
USE_FUSED_RETRIEVAL = True
FUSION_WEIGHT_RAW = 0.6
FUSION_WEIGHT_SUMMARY = 0.3
FUSION_WEIGHT_RECENCY = 0.1
SUMMARY_SYSTEM_PROMPT = (
    "You are a summarization component. Produce a concise, factual, neutral "
    "compression of the conversation. Keep only key facts, decisions, constraints, "
    "and open items. Do not speculate or add new information."
)

model = get_model()
dimension = model.get_sentence_embedding_dimension()


def _load_faiss_index(path):
    """Load a FAISS index from disk or create an empty compatible index.

    Args:
        path: Filesystem path to a FAISS index file.

    Returns:
        Loaded `faiss.Index` instance or an empty `IndexFlatIP` with module dimension.

    Determinism:
        Deterministic for a fixed file state and embedding dimension.

    Edge cases:
        - Missing index file returns an empty in-memory index.
        - Corrupt index file falls back to an empty in-memory index.

    Failure modes:
        - Read failures are logged with traceback and converted to empty index fallback.
    """
    if os.path.exists(path):
        try:
            return faiss.read_index(path)
        except Exception:
            logger.exception("Failed to load conversation FAISS index from %s", path)
    return faiss.IndexFlatIP(dimension)


index = _load_faiss_index(INDEX_PATH)
summary_index = _load_faiss_index(SUMMARY_INDEX_PATH)


if os.path.exists(META_PATH):
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        logger.exception("Failed to load conversation metadata from %s", META_PATH)
        metadata = []
else:
    metadata = []

if os.path.exists(SUMMARY_META_PATH):
    try:
        with open(SUMMARY_META_PATH, "r", encoding="utf-8") as f:
            summary_metadata = json.load(f)
    except Exception:
        logger.exception("Failed to load summary metadata from %s", SUMMARY_META_PATH)
        summary_metadata = []
else:
    summary_metadata = []


if index.ntotal != len(metadata):

    index = faiss.IndexFlatIP(dimension)

    for entry in metadata:
        vec = model.encode(["passage: " + entry["text"]])
        vec = np.array(vec).astype("float32")
        faiss.normalize_L2(vec)
        index.add(vec)

    faiss.write_index(index, INDEX_PATH)

if getattr(summary_index, "d", dimension) != dimension:
    logger.warning(
        "Summary FAISS dimension mismatch (index=%s, expected=%s). Reinitializing summary index.",
        getattr(summary_index, "d", "unknown"),
        dimension,
    )
    summary_index = faiss.IndexFlatIP(dimension)

if summary_index.ntotal != len(summary_metadata):

    summary_index = faiss.IndexFlatIP(dimension)
    rebuilt_summary_metadata = []

    for entry in summary_metadata:
        summary_text = entry.get("summary_text", "")
        if not summary_text or not str(summary_text).strip():
            continue

        vec = model.encode(["passage: " + str(summary_text)])
        vec = np.array(vec).astype("float32")
        faiss.normalize_L2(vec)
        summary_index.add(vec)
        rebuilt_summary_metadata.append(entry)

    summary_metadata = rebuilt_summary_metadata

    summary_index_tmp = SUMMARY_INDEX_PATH + ".tmp"
    summary_meta_tmp = SUMMARY_META_PATH + ".tmp"

    try:
        faiss.write_index(summary_index, summary_index_tmp)

        with open(summary_meta_tmp, "w", encoding="utf-8") as f:
            json.dump(summary_metadata, f, indent=2, ensure_ascii=False)

        os.replace(summary_index_tmp, SUMMARY_INDEX_PATH)
        os.replace(summary_meta_tmp, SUMMARY_META_PATH)
    except Exception:
        logger.exception("Failed to persist rebuilt summary index/metadata")


def _embed(text: str, is_query: bool = False):
    """Embed text into normalized FAISS-compatible float32 vectors.

    Args:
        text: Input text to embed.
        is_query: When `True`, uses query prefix; otherwise passage prefix.

    Returns:
        A normalized 2D float32 numpy array or `None` for empty input.

    Determinism:
        Deterministic for fixed model weights/runtime and identical input.

    Edge cases:
        - Empty/whitespace input returns `None`.

    Failure modes:
        - Model/runtime errors propagate to caller.
    """
    if not text or not str(text).strip():
        return None

    prefix = "query: " if is_query else "passage: "
    vec = model.encode([prefix + str(text)])

    vec = np.array(vec).astype("float32")
    faiss.normalize_L2(vec)
    return vec


def chunk_summary_text(summary_text, chunk_size=CHUNK_SIZE):
    """Split summary text/list into fixed-size line-group chunks.

    Args:
        summary_text: Summary content as `str` or list-like entries.
        chunk_size: Maximum number of lines/items per chunk.

    Returns:
        List of chunk strings.

    Determinism:
        Deterministic and side-effect free for identical inputs.

    Edge cases:
        - Invalid/empty input returns `[]`.
        - Non-positive or non-integer `chunk_size` falls back to `CHUNK_SIZE`.

    Failure modes:
        - No explicit exceptions are raised for malformed content types; returns `[]`.
    """
    if not summary_text:
        return []

    try:
        size = int(chunk_size)
    except Exception:
        size = CHUNK_SIZE

    if size <= 0:
        size = CHUNK_SIZE

    if isinstance(summary_text, str):
        units = [line.strip() for line in summary_text.splitlines() if line.strip()]
    elif isinstance(summary_text, list):
        units = [str(item).strip() for item in summary_text if str(item).strip()]
    else:
        return []

    if not units:
        return []

    chunks = []
    for i in range(0, len(units), size):
        chunks.append("\n".join(units[i:i + size]))

    return chunks


def summarize_messages_pure(messages):
    """Build a normalized transcript string from message dictionaries.

    Args:
        messages: List of chat message dictionaries with `role` and `content`.

    Returns:
        Newline-joined transcript (`ROLE: content`) or empty string.

    Determinism:
        Deterministic and side-effect free for identical inputs.

    Edge cases:
        - Non-list or empty input returns `""`.
        - Non-dict items and empty content are skipped.

    Failure modes:
        - No internal exception handling; type errors are minimized via casting.
    """
    if not messages or not isinstance(messages, list):
        return ""

    lines = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "")).strip() or "unknown"
        content = str(message.get("content", "")).strip()
        if not content:
            continue

        lines.append(f"{role.upper()}: {content}")

    return "\n".join(lines).strip()


def _estimate_tokens(text):
    """Estimate token count with a character heuristic.

    Args:
        text: Prompt text.

    Returns:
        Approximate token count.

    Determinism:
        Deterministic and side-effect free.

    Edge cases:
        - Empty input returns `0`.
        - Non-empty input returns at least `1`.

    Failure modes:
        - No explicit failure handling; relies on `str(text)`.
    """
    if not text:
        return 0
    return max(1, len(str(text)) // SUMMARY_CHARS_PER_TOKEN_ESTIMATE)


def _enforce_summary_prompt_budget(prompt):
    """Trim summary prompt text to configured token budget estimate.

    Args:
        prompt: Summary prompt text.

    Returns:
        Original prompt when within budget; truncated variant otherwise.

    Determinism:
        Deterministic for identical input and constant budget values.

    Edge cases:
        - Empty prompt returns `""`.
        - Very small computed max length falls back to head-only truncation.

    Failure modes:
        - No external I/O; failures are unlikely and would propagate.
    """
    if not prompt:
        return ""

    if _estimate_tokens(prompt) <= SUMMARY_PROMPT_TOKEN_BUDGET:
        return prompt

    max_chars = SUMMARY_PROMPT_TOKEN_BUDGET * SUMMARY_CHARS_PER_TOKEN_ESTIMATE
    marker = "\n\n[TRUNCATED SUMMARY INPUT]\n\n"

    if max_chars <= len(marker) + 32:
        return prompt[:max_chars]

    head_budget = int(max_chars * 0.55)
    tail_budget = max_chars - head_budget - len(marker)
    if tail_budget <= 0:
        return prompt[:max_chars]

    head = prompt[:head_budget].rstrip()
    tail = prompt[-tail_budget:].lstrip()
    trimmed = f"{head}{marker}{tail}"
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars]
    return trimmed


def _materialize_response_text(response):
    """Convert model response objects into a single normalized string.

    Args:
        response: `None`, string, or iterable of chunks.

    Returns:
        Stripped response text.

    Determinism:
        Deterministic for a fixed response object/iteration order.

    Edge cases:
        - `None` returns `""`.
        - Iterable chunks are concatenated without separators.

    Failure modes:
        - Exceptions during iteration/string-cast propagate.
    """
    if response is None:
        return ""

    if hasattr(response, "__iter__") and not isinstance(response, (str, bytes)):
        return "".join(str(chunk) for chunk in response).strip()

    return str(response).strip()


def _safe_generate_summary_only(prompt):
    """Generate session summary text via isolated LLM call.

    Args:
        prompt: User-content summary prompt.

    Returns:
        Materialized summary string or `""` on failure.

    Determinism:
        Not deterministic; underlying LLM generation is probabilistic.

    Edge cases:
        - Empty prompt returns `""` without calling the provider.
        - Prompt is budget-limited before dispatch.

    Failure modes:
        - Provider/request exceptions are logged and converted to `""`.
    """
    if not prompt or not str(prompt).strip():
        return ""

    bounded_prompt = _enforce_summary_prompt_budget(prompt)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": bounded_prompt}
        ],
        "temperature": 0.45,
        "top_p": 0.9,
        "presence_penalty": 0.4,
        "frequency_penalty": 0.5,
        "stream": False
    }

    try:
        response = send_request(payload, stream=False)
    except Exception:
        logger.exception("Summary generation failed")
        return ""

    return _materialize_response_text(response)


def _trim_summary_to_max_tokens(summary_text):
    """Enforce hard maximum summary length using char-based token estimate.

    Args:
        summary_text: Generated summary text.

    Returns:
        Original/trimmed summary text.

    Determinism:
        Deterministic for identical input and max-token constants.

    Edge cases:
        - Empty input returns `""`.

    Failure modes:
        - No external dependencies; failures are unlikely.
    """
    if not summary_text:
        return ""

    max_chars = SUMMARY_MAX_TOKENS * SUMMARY_CHARS_PER_TOKEN_ESTIMATE
    if len(summary_text) <= max_chars:
        return summary_text.strip()

    return summary_text[:max_chars].rstrip()


def _generate_session_summary(messages):
    """Create bounded summary text from archived session messages.

    Args:
        messages: Session message list.

    Returns:
        Final summary text or `""` when generation should be skipped.

    Determinism:
        Partially deterministic: transcript building is deterministic, LLM output is not.

    Edge cases:
        - Empty transcript returns `""`.
        - Provider error strings (for example key missing) are detected and rejected.

    Failure modes:
        - Downstream LLM failures are handled in `_safe_generate_summary_only`.
    """
    transcript = summarize_messages_pure(messages)
    if not transcript:
        return ""

    summary_prompt = (
        "Erstelle eine kompakte Session-Zusammenfassung nur aus dem folgenden Chat-Transcript.\n"
        "Nutze ausschließlich den bereitgestellten Transcript-Inhalt.\n"
        f"Ziel-Länge: {SUMMARY_MIN_TOKENS}-{SUMMARY_MAX_TOKENS} Tokens.\n"
        f"Obergrenze: maximal {SUMMARY_MAX_TOKENS} Tokens.\n"
        "Fokussiere auf Ziele, Fakten, Entscheidungen und offene Punkte.\n"
        "Nur Klartext, keine Markdown-Überschrift.\n\n"
        "=== SESSION TRANSCRIPT START ===\n"
        f"{transcript}\n"
        "=== SESSION TRANSCRIPT END ===\n\n"
        "Session Summary:"
    )

    summary_text = _safe_generate_summary_only(summary_prompt)
    if not summary_text:
        return ""

    error_signals = [
        "HTTP ERROR",
        "REQUEST FAILED",
        "KEY FILE NOT FOUND",
        "INVALID PROVIDER",
    ]
    if any(signal in summary_text.upper() for signal in error_signals):
        logger.warning("Skipping summary persistence due to LLM provider error response")
        return ""

    return _trim_summary_to_max_tokens(summary_text)


def add_session_summary_to_memory(messages, session_hash):
    """Persist one summary vector/metadata entry per unique session hash.

    Args:
        messages: Archived session messages.
        session_hash: Stable content hash produced by `add_session_to_memory`.

    Returns:
        None.

    Determinism:
        Persistence decisions are deterministic except LLM summary generation.

    Edge cases:
        - Empty messages or missing hash are ignored.
        - Existing summary for hash is not duplicated.
        - Missing raw metadata session id triggers generated UUID fallback.

    Failure modes:
        - Any generation or persistence failure is logged and exits early.
        - Function is best-effort and intentionally suppresses raised exceptions.
    """
    global summary_metadata

    if not messages or not isinstance(messages, list):
        return

    if not session_hash:
        return

    try:
        if any(entry.get("session_hash") == session_hash for entry in summary_metadata):
            return

        summary_text = _generate_session_summary(messages)
        if not summary_text:
            return

        vec = _embed(summary_text, is_query=False)
        if vec is None:
            return

        summary_session_id = None
        for entry in metadata:
            if entry.get("session_hash") == session_hash:
                summary_session_id = entry.get("session_id")
                break

        if not summary_session_id:
            summary_session_id = str(uuid.uuid4())

        summary_index.add(vec)

        summary_metadata.append({
            "session_id": summary_session_id,
            "session_hash": session_hash,
            "created_at": time.time(),
            "summary_text": summary_text
        })

        summary_index_tmp = SUMMARY_INDEX_PATH + ".tmp"
        summary_meta_tmp = SUMMARY_META_PATH + ".tmp"

        try:
            faiss.write_index(summary_index, summary_index_tmp)

            with open(summary_meta_tmp, "w", encoding="utf-8") as f:
                json.dump(summary_metadata, f, indent=2, ensure_ascii=False)

            os.replace(summary_index_tmp, SUMMARY_INDEX_PATH)
            os.replace(summary_meta_tmp, SUMMARY_META_PATH)
        except Exception:
            logger.exception("Failed to persist summary index/metadata")
            return
    except Exception:
        logger.exception("Failed to add session summary to memory")
        return


def add_session_to_memory(messages):
    """Index a full session into raw conversation memory.

    Args:
        messages: Session message dictionaries.

    Returns:
        Session hash string on success/deduplicated existing session, or `None`
        when input is invalid.

    Determinism:
        - Session hash generation is deterministic for identical message payload.
        - Embedding vectors depend on model/runtime determinism.

    Edge cases:
        - Duplicate session hashes are not re-indexed.
        - Empty/non-dict message content is skipped.

    Failure modes:
        - Index/metadata persistence failures are logged and re-raised.
    """

    if not messages or not isinstance(messages, list):
        return None

    session_hash_source = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    session_hash = hashlib.sha256(session_hash_source.encode("utf-8")).hexdigest()

    if any(entry.get("session_hash") == session_hash for entry in metadata):
        return session_hash

    session_id = str(uuid.uuid4())
    chunk_index = 0

    for m in messages:

        role = m.get("role", "")
        content = m.get("content", "")

        if not content or not str(content).strip():
            continue

        entry_type = "user_query" if role == "user" else "assistant_response"

        vec = _embed(content, is_query=False)
        if vec is None:
            continue

        index.add(vec)

        metadata.append({
            "session_id": session_id,
            "session_hash": session_hash,
            "chunk_index": chunk_index,
            "type": entry_type,
            "text": content,
            "timestamp": time.time()
        })

        chunk_index += 1

    index_tmp = INDEX_PATH + ".tmp"
    meta_tmp = META_PATH + ".tmp"

    try:
        faiss.write_index(index, index_tmp)

        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        os.replace(index_tmp, INDEX_PATH)
        os.replace(meta_tmp, META_PATH)

    except Exception:
        logger.exception("Failed to persist conversation index/metadata")
        raise

    return session_hash


def _parse_time_window(query: str):
    """Parse supported temporal phrases into UTC timestamp windows.

    Args:
        query: User query text.

    Returns:
        `(start_ts, end_ts)` tuple when a supported phrase is found, else `None`.

    Determinism:
        Deterministic for a fixed `query` and evaluation time.

    Edge cases:
        - Supports specific German expressions only.
        - If no phrase matches, returns `None`.

    Failure modes:
        - No explicit failure handling; unexpected values propagate.
    """
    q = query.lower()
    now = datetime.utcnow()

    if "vorgestern" in q:
        target = now - timedelta(days=2)
        start = datetime(target.year, target.month, target.day)
        end = start + timedelta(days=1)
        return start.timestamp(), end.timestamp()

    if "gestern" in q:
        start = now - timedelta(days=1)
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return start.timestamp(), end.timestamp()

    if "heute morgen" in q:
        start = now.replace(hour=5, minute=0, second=0, microsecond=0)
        end = now.replace(hour=12, minute=0, second=0, microsecond=0)
        return start.timestamp(), end.timestamp()

    if "heute" in q:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
        return start.timestamp(), end.timestamp()

    if "letzte woche" in q:
        start = now - timedelta(days=7)
        return start.timestamp(), now.timestamp()

    return None


def _retrieve_by_time_window(start_ts, end_ts):
    """Reconstruct sessions whose chunk timestamps fall inside a time window.

    Args:
        start_ts: Inclusive start timestamp.
        end_ts: Inclusive end timestamp.

    Returns:
        List of full-session text blocks.

    Determinism:
        Deterministic for fixed metadata ordering/content.

    Edge cases:
        - Entries without timestamps are ignored.
        - Empty matches return `[]`.

    Failure modes:
        - Missing expected metadata keys may raise and propagate.
    """

    sessions = {}

    for entry in metadata:
        ts = entry.get("timestamp")
        if not ts:
            continue

        if start_ts <= ts <= end_ts:
            sid = entry["session_id"]
            sessions.setdefault(sid, []).append(entry)

    results = []

    for sid, entries in sessions.items():
        entries.sort(key=lambda x: x.get("chunk_index", 0))
        text = "\n\n".join(e["text"] for e in entries)
        results.append(text)

    return results


def search_raw_index(query_embedding, top_k=10):
    """Search raw conversation index and return scored session-hash hits.

    Args:
        query_embedding: FAISS-compatible query vector.
        top_k: Number of nearest neighbors requested.

    Returns:
        List of dicts with `score`, `session_hash`, and `timestamp`.

    Determinism:
        Deterministic for fixed index state and query vector.

    Edge cases:
        - Out-of-range FAISS indices are skipped.
        - Empty/undersized index may yield `[]`.

    Failure modes:
        - FAISS search exceptions propagate to caller.
    """
    scores, indices = index.search(query_embedding, top_k)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        results.append({
            "score": float(score),
            "session_hash": metadata[idx]["session_hash"],
            "timestamp": metadata[idx]["timestamp"]
        })

    return results


def search_summary_index(query_embedding, top_k=10):
    """Search summary index and return scored session-hash hits.

    Args:
        query_embedding: FAISS-compatible query vector.
        top_k: Number of nearest neighbors requested.

    Returns:
        List of dicts with `score`, `session_hash`, and summary `created_at`.

    Determinism:
        Deterministic for fixed summary index state and query vector.

    Edge cases:
        - Out-of-range FAISS indices are skipped.

    Failure modes:
        - FAISS search exceptions propagate to caller.
    """
    scores, indices = summary_index.search(query_embedding, top_k)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(summary_metadata):
            continue

        results.append({
            "score": float(score),
            "session_hash": summary_metadata[idx]["session_hash"],
            "timestamp": summary_metadata[idx].get("created_at")
        })

    return results


def search_memory_fused(query_embedding, top_k=10):
    """Fuse raw and summary retrieval signals into ranked session candidates.

    Args:
        query_embedding: FAISS-compatible query vector.
        top_k: Maximum number of fused results to return.

    Returns:
        Ranked list of fused hit dicts with `final_score`.

    Determinism:
        Partially deterministic: raw/summary scores are deterministic for fixed
        indexes, while recency boost depends on current UTC time.

    Edge cases:
        - Missing/invalid result fields are normalized with safe fallbacks.
        - Sessions are deduplicated to best raw hit per hash.
        - Results below `MIN_SCORE_ABSOLUTE` are discarded.

    Failure modes:
        - Depends on callable search helpers; non-callable fallbacks yield empty sets.
        - Unexpected types are skipped rather than raising.
    """
    summary_search_fn = globals().get("search_summary_index")
    raw_search_fn = globals().get("search_raw_index")

    summary_results = summary_search_fn(query_embedding, top_k=10) if callable(summary_search_fn) else []
    raw_results = raw_search_fn(query_embedding, top_k=50) if callable(raw_search_fn) else []

    best_raw_by_session = {}
    filtered_raw_results = []

    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            continue

        session_hash = raw_result.get("session_hash")
        if not session_hash:
            filtered_raw_results.append(raw_result)
            continue

        try:
            raw_score = float(raw_result.get("score", 0.0))
        except Exception:
            raw_score = 0.0

        previous = best_raw_by_session.get(session_hash)
        if previous is None or raw_score > previous[0]:
            best_raw_by_session[session_hash] = (raw_score, raw_result)

    for _, best_result in best_raw_by_session.values():
        filtered_raw_results.append(best_result)

    raw_results = filtered_raw_results

    summary_scores = {}
    for summary_result in summary_results:
        if not isinstance(summary_result, dict):
            continue

        session_hash = summary_result.get("session_hash")
        if not session_hash:
            continue

        try:
            summary_score = float(summary_result.get("score", 0.0))
        except Exception:
            summary_score = 0.0

        previous = summary_scores.get(session_hash)
        if previous is None or summary_score > previous:
            summary_scores[session_hash] = summary_score

    if summary_scores:
        max_summary_score = max(summary_scores.values())
    else:
        max_summary_score = 0.0

    now = datetime.utcnow()
    fused_results = []

    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            continue

        try:
            raw_score = float(raw_result.get("score", 0.0))
        except Exception:
            raw_score = 0.0

        session_hash = raw_result.get("session_hash")
        if session_hash:
            raw_summary_score = float(summary_scores.get(session_hash, 0.0))
            if max_summary_score > 0:
                summary_score = raw_summary_score / max_summary_score
            else:
                summary_score = 0.0
        else:
            summary_score = 0.0

        timestamp_value = raw_result.get("timestamp")
        timestamp_dt = now

        if isinstance(timestamp_value, (int, float)):
            try:
                timestamp_dt = datetime.utcfromtimestamp(float(timestamp_value))
            except Exception:
                timestamp_dt = now

        age_hours = (now - timestamp_dt).total_seconds() / 3600
        recency_boost = 1 / (1 + age_hours / 24)

        final_score = (
            FUSION_WEIGHT_RAW * raw_score +
            FUSION_WEIGHT_SUMMARY * summary_score +
            FUSION_WEIGHT_RECENCY * recency_boost
        )

        if final_score < MIN_SCORE_ABSOLUTE:
            continue

        fused_result = dict(raw_result)
        fused_result["final_score"] = final_score
        fused_results.append(fused_result)

    fused_results.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
    return fused_results[:top_k]


def retrieve_conversation(query, top_k=5, intent_type=None):
    """Retrieve conversation history using temporal-first and semantic strategies.

    Args:
        query: User query text.
        top_k: Maximum number of sessions to return.
        intent_type: Optional route guard; only `"conversation_query"` is accepted.

    Returns:
        List of reconstructed session text blocks.

    Determinism:
        Mixed:
        - Temporal path is deterministic for fixed metadata and evaluation time.
        - Semantic ranking is deterministic for fixed index/query state.
        - Fused ranking includes time-dependent recency weighting.

    Edge cases:
        - Hard route behavior: `/memory ...` in `app.retrieval.context_builder`
          forces `conversation_query`, which resolves through this function.
        - Non-conversation `intent_type` returns `[]`.
        - Very short non-temporal queries are blocked by token guard.
        - Temporal matches bypass semantic retrieval and return window results.

    Failure modes:
        - Embedding generation failures return `[]`.
        - FAISS search failures propagate if raised by index/search helpers.
    """

    if intent_type and intent_type != "conversation_query":
        return []

    time_window = _parse_time_window(query)
    if time_window:
        start_ts, end_ts = time_window
        if start_ts is not None and end_ts is not None:
            return _retrieve_by_time_window(start_ts, end_ts)

    if not time_window:
        if len(str(query).split()) <= SHORT_QUERY_TOKEN_LIMIT:
            return []

    if not metadata or index.ntotal == 0:
        return []

    qvec = _embed(query, is_query=True)
    if qvec is None:
        return []

    query_embedding = qvec
    if USE_FUSED_RETRIEVAL:
        fused_results = search_memory_fused(query_embedding, top_k=top_k)
        matched_sessions = []
        seen_session_hashes = set()

        for result in fused_results:
            if not isinstance(result, dict):
                continue

            session_hash = result.get("session_hash")
            if not session_hash or session_hash in seen_session_hashes:
                continue

            seen_session_hashes.add(session_hash)

            session_chunks = [
                m for m in metadata
                if m.get("session_hash") == session_hash
            ]

            session_chunks.sort(key=lambda x: x.get("chunk_index", 0))

            full_session_text = "\n\n".join(
                m["text"] for m in session_chunks
            )

            if full_session_text:
                matched_sessions.append(full_session_text)

            if len(matched_sessions) >= top_k:
                break

        return matched_sessions

    search_k = min(top_k * 4, index.ntotal)
    scores, indices = index.search(qvec, search_k)

    if len(scores[0]) == 0:
        return []

    top_score = float(scores[0][0])
    if top_score < MIN_SCORE_ABSOLUTE:
        return []

    matched_sessions = {}
    seen_sessions = set()

    for rank, idx in enumerate(indices[0]):

        if idx < 0 or idx >= len(metadata):
            continue

        entry = metadata[idx]

        if entry.get("type") != "user_query":
            continue

        raw_score = float(scores[0][rank])

        if raw_score < MIN_SCORE_ABSOLUTE:
            continue

        if raw_score < top_score * 0.8:
            continue

        if time_window:
            start_ts, end_ts = time_window
            ts = entry.get("timestamp")
            if not ts or not (start_ts <= ts <= end_ts):
                continue

        session_id = entry.get("session_id")
        if not session_id or session_id in seen_sessions:
            continue

        seen_sessions.add(session_id)

        session_chunks = [
            m for m in metadata
            if m.get("session_id") == session_id
        ]

        session_chunks.sort(key=lambda x: x.get("chunk_index", 0))

        full_session_text = "\n\n".join(
            m["text"] for m in session_chunks
        )

        matched_sessions[session_id] = full_session_text

        if len(matched_sessions) >= top_k:
            break

    if time_window and not matched_sessions:
        start_ts, end_ts = time_window
        return _retrieve_by_time_window(start_ts, end_ts)

    return list(matched_sessions.values())


def get_last_session():
    """Return the most recently indexed full session transcript.

    Args:
        None.

    Returns:
        Single-item list containing the latest reconstructed session, or `[]`.

    Determinism:
        Deterministic for fixed metadata ordering/content.

    Edge cases:
        - Empty metadata returns `[]`.

    Failure modes:
        - Missing expected metadata keys may raise and propagate.
    """

    if not metadata:
        return []

    last_session_id = metadata[-1].get("session_id")

    session_chunks = [
        m for m in metadata
        if m.get("session_id") == last_session_id
    ]

    session_chunks.sort(key=lambda x: x.get("chunk_index", 0))

    full_session_text = "\n\n".join(
        m["text"] for m in session_chunks
    )

    return [full_session_text]
