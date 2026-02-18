import os
import json
import time
import faiss
import numpy as np
import uuid
from datetime import datetime, timedelta

from app.embedding_model import get_model


# =========================================================
# Configuration
# =========================================================
INDEX_PATH = "conversation_index.faiss"
META_PATH = "conversation_meta.json"
SESSION_LOG_PATH = "current_session.json"

MIN_SCORE_ABSOLUTE = 0.48
SHORT_QUERY_TOKEN_LIMIT = 3
CHUNK_SIZE = 4

model = get_model()
dimension = model.get_sentence_embedding_dimension()


# =========================================================
# Load or Create FAISS Index
# =========================================================
def _load_faiss_index(path):
    if os.path.exists(path):
        try:
            return faiss.read_index(path)
        except Exception:
            pass
    return faiss.IndexFlatIP(dimension)


index = _load_faiss_index(INDEX_PATH)


# =========================================================
# Load Metadata
# =========================================================
if os.path.exists(META_PATH):
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        metadata = []
else:
    metadata = []


# =========================================================
# Index / Metadata Consistency Check
# =========================================================
if index.ntotal != len(metadata):

    index = faiss.IndexFlatIP(dimension)

    for entry in metadata:
        vec = model.encode(["passage: " + entry["text"]])
        vec = np.array(vec).astype("float32")
        faiss.normalize_L2(vec)
        index.add(vec)

    faiss.write_index(index, INDEX_PATH)


# =========================================================
# Embedding Utilities
# =========================================================
def _embed(text: str, is_query: bool = False):
    """
    Generates a normalized embedding vector for a given text.

    Uses different prefixes for query and passage encoding
    to align with embedding model best practices.
    """
    if not text or not str(text).strip():
        return None

    prefix = "query: " if is_query else "passage: "
    vec = model.encode([prefix + str(text)])

    vec = np.array(vec).astype("float32")
    faiss.normalize_L2(vec)
    return vec


# =========================================================
# Add Session to Memory
# =========================================================
def add_session_to_memory(messages):
    """
    Indexes a full conversation session into long-term memory.

    Each message is embedded and stored with metadata, including
    session ID, chunk order, type, and timestamp.
    """

    if not messages or not isinstance(messages, list):
        return

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
        pass


# =========================================================
# Temporal Query Detection
# =========================================================
def _parse_time_window(query: str):
    """
    Detects simple temporal expressions and converts them
    into timestamp ranges for time-based filtering.
    """
    q = query.lower()
    now = datetime.utcnow()

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


# =========================================================
# Temporal Retrieval
# =========================================================
def _retrieve_by_time_window(start_ts, end_ts):
    """
    Retrieves full sessions that fall within a specified
    timestamp window.
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


# =========================================================
# Hybrid Semantic + Temporal Retrieval
# =========================================================
def retrieve_conversation(query, top_k=5, intent_type=None):
    """
    Retrieves relevant past conversations using a hybrid strategy:
    - Semantic similarity search (FAISS)
    - Optional temporal filtering
    - Short-query safeguards
    """

    if intent_type and intent_type != "conversation_query":
        return []

    time_window = _parse_time_window(query)

    # If the query is purely temporal and very short,
    # perform direct time-based retrieval without embeddings.
    if time_window and len(str(query).split()) <= SHORT_QUERY_TOKEN_LIMIT:
        start_ts, end_ts = time_window
        return _retrieve_by_time_window(start_ts, end_ts)

    # Short query guard (when not temporal)
    if not time_window:
        if len(str(query).split()) <= SHORT_QUERY_TOKEN_LIMIT:
            return []

    if not metadata or index.ntotal == 0:
        return []

    qvec = _embed(query, is_query=True)
    if qvec is None:
        return []

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

        # Temporal filter
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

    # Fallback: purely temporal query without semantic match
    if time_window and not matched_sessions:
        start_ts, end_ts = time_window
        return _retrieve_by_time_window(start_ts, end_ts)

    return list(matched_sessions.values())


# =========================================================
# Last Archived Session
# =========================================================
def get_last_session():
    """
    Returns the most recently indexed conversation session.
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
