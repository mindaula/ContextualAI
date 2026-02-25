"""Personal and academic memory storage/retrieval engine.

Architectural role:
    Implements vector-backed persistence for two memory domains used by the
    application:
    - Personal memory (`personal.index`, `personal_meta.json`) for user facts.
    - Academic memory (`knowledge/<domain>/academic.index`, metadata) for ingested
      technical/reference content.

Responsibilities:
    - Normalize and embed text for FAISS-compatible search/indexing.
    - Load and persist index/metadata artifacts.
    - Execute domain-specific retrieval with thresholding and score filtering.
    - Append new personal facts and academic chunks.
"""

import os
import json
import re
import logging
import numpy as np
import faiss
from datetime import datetime

from app.memory.embedding_model import get_model


logger = logging.getLogger(__name__)


MAX_PERSONAL_FACTS = 200

PERSONAL_MIN_SIMILARITY = 0.60
PERSONAL_DUPLICATE_THRESHOLD = 0.85

ACADEMIC_MIN_SIMILARITY = 0.72
RELATIVE_SCORE_RATIO = 0.90
MIN_QUERY_TOKENS_FOR_ACADEMIC = 2


def normalize_text(text: str):
    """Normalize text for embedding and token-level heuristics.

    Args:
        text: Raw user or document text.

    Returns:
        Lowercased, de-accented, punctuation-reduced text with normalized spacing.
    """
    if not text:
        return ""

    text = str(text).lower()
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_count(text: str):
    """Return token count after normalization.

    Args:
        text: Input text.

    Returns:
        Number of whitespace-delimited tokens in normalized text.
    """
    return len(normalize_text(text).split())


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PERSONAL_INDEX_FILE = os.path.join(BASE_DIR, "personal.index")
PERSONAL_META_FILE = os.path.join(BASE_DIR, "personal_meta.json")


model = get_model()
dimension = model.get_sentence_embedding_dimension()


def atomic_json_save(path, data):
    """Persist JSON data atomically via temporary file replacement.

    Args:
        path: Destination JSON path.
        data: JSON-serializable payload.

    Returns:
        None.

    Side effects:
        Writes `<path>.tmp` and atomically replaces `path`.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_index(path):
    """Load a FAISS index or return a new empty compatible index.

    Args:
        path: Index file path.

    Returns:
        Loaded FAISS index when available and readable, otherwise `IndexFlatIP`.
    """
    if os.path.exists(path):
        try:
            return faiss.read_index(path)
        except Exception:
            logger.exception("Failed to load FAISS index from %s", path)
    return faiss.IndexFlatIP(dimension)


def load_meta(path):
    """Load metadata JSON list from disk.

    Args:
        path: Metadata file path.

    Returns:
        Parsed metadata list, or an empty list when file is missing/unreadable.
    """
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.exception("Failed to load metadata JSON from %s", path)
    return []


personal_index = load_index(PERSONAL_INDEX_FILE)
personal_meta = load_meta(PERSONAL_META_FILE)


def embed(text: str, is_query=False):
    """Embed one text into a normalized float32 vector.

    Args:
        text: Text payload.
        is_query: Selects `query:` vs `passage:` prefix.

    Returns:
        2D normalized numpy vector or `None` when input is empty.
    """
    if not text or not str(text).strip():
        return None

    text = normalize_text(text)
    prefix = "query: " if is_query else "passage: "

    vec = model.encode([prefix + text])
    vec = np.array(vec).astype("float32")
    faiss.normalize_L2(vec)
    return vec


def embed_batch(texts):
    """Embed multiple texts into normalized float32 vectors.

    Args:
        texts: Iterable of text values.

    Returns:
        2D normalized vector matrix, or `None` if no valid text is provided.
    """
    clean = [normalize_text(t) for t in texts if t and str(t).strip()]
    if not clean:
        return None

    prefixed = ["passage: " + t for t in clean]
    vecs = model.encode(prefixed)
    vecs = np.array(vecs).astype("float32")
    faiss.normalize_L2(vecs)
    return vecs


def search_personal(query, top_k=3, return_scores=False):
    """Search personal memory with similarity and relative-score filtering.

    Args:
        query: User query.
        top_k: Maximum FAISS neighbors to inspect.
        return_scores: Whether to return `(text, score)` tuples.

    Returns:
        Filtered personal memory hits as text strings or scored tuples.
    """

    if personal_index.ntotal == 0:
        return []

    vec = embed(query, is_query=True)
    if vec is None:
        return []

    scores, indices = personal_index.search(vec, top_k)

    candidates = []

    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(personal_meta):
            continue

        similarity = float(scores[0][rank])
        if similarity < PERSONAL_MIN_SIMILARITY:
            continue

        candidates.append((personal_meta[idx]["text"], similarity))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_score = candidates[0][1]

    results = []

    for text, score in candidates:
        if score < top_score * RELATIVE_SCORE_RATIO:
            continue

        if return_scores:
            results.append((text, score))
        else:
            results.append(text)

    return results


def search_academic(query, domain, top_k=5, return_scores=False):
    """Search academic memory for a specific domain (or root fallback path).

    Args:
        query: Query text.
        domain: Domain folder name under `knowledge/`.
        top_k: Maximum FAISS neighbors to inspect.
        return_scores: Whether to return `(entry, score)` tuples.

    Returns:
        Filtered academic entries (or scored tuples), ordered by boosted score.
    """

    if token_count(query) < MIN_QUERY_TOKENS_FOR_ACADEMIC:
        return []

    if domain:
        domain_dir = os.path.join(BASE_DIR, "knowledge", domain)
        index_path = os.path.join(domain_dir, "academic.index")
        meta_path = os.path.join(domain_dir, "academic_meta.json")
    else:
        knowledge_dir = os.path.join(BASE_DIR, "knowledge")
        index_path = os.path.join(knowledge_dir, "academic.index")
        meta_path = os.path.join(knowledge_dir, "academic_meta.json")

        if not os.path.exists(index_path):
            return []

    academic_index = load_index(index_path)
    academic_meta = load_meta(meta_path)

    if academic_index.ntotal == 0:
        return []

    vec = embed(query, is_query=True)
    if vec is None:
        return []

    scores, indices = academic_index.search(vec, top_k)

    raw = []

    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(academic_meta):
            continue

        similarity = float(scores[0][rank])
        raw.append((academic_meta[idx], similarity))

    if not raw:
        return []

    current_year = datetime.now().year
    boosted = []

    for entry, score in raw:
        year = entry.get("year", current_year)
        recency_bonus = (year - 2000) * 0.002
        boosted.append((entry, score + recency_bonus))

    boosted.sort(key=lambda x: x[1], reverse=True)

    top_score = boosted[0][1]

    if top_score < ACADEMIC_MIN_SIMILARITY:
        return []

    results = []

    for entry, score in boosted:
        if score < top_score * RELATIVE_SCORE_RATIO:
            continue

        if return_scores:
            results.append((entry, score))
        else:
            results.append(entry)

    return results


def add_personal_fact(text: str):
    """Append a personal fact to persistent personal memory.

    Args:
        text: Fact statement to store.

    Returns:
        `True` when fact embedding/persistence succeeds, else `False`.

    Side effects:
        - Mutates module globals `personal_index` and `personal_meta`.
        - Rebuilds the personal index when retention limit is exceeded.
        - Writes `personal.index` and `personal_meta.json`.
    """

    global personal_index, personal_meta

    if not text or not str(text).strip():
        return False

    text = text.strip()

    vec = embed(text, is_query=False)
    if vec is None:
        return False

    personal_index.add(vec)

    personal_meta.append({
        "text": text
    })

    if len(personal_meta) > MAX_PERSONAL_FACTS:
        personal_meta.pop(0)

        texts = [entry["text"] for entry in personal_meta]
        new_index = faiss.IndexFlatIP(dimension)

        vecs = embed_batch(texts)
        if vecs is not None:
            new_index.add(vecs)

        personal_index = new_index

    faiss.write_index(personal_index, PERSONAL_INDEX_FILE)
    atomic_json_save(PERSONAL_META_FILE, personal_meta)

    return True


def add_academic_chunks(texts, metadata: dict, domain):
    """Append academic chunks and metadata into a domain-specific index.

    Args:
        texts: Chunk list to embed/store.
        metadata: Metadata overrides merged with defaults.
        domain: Target knowledge domain directory.

    Returns:
        `True` when chunks are embedded and persisted, else `False`.

    Side effects:
        - Creates `knowledge/<domain>` when missing.
        - Writes/updates `academic.index` and `academic_meta.json`.
    """

    if not texts:
        return False

    domain_dir = os.path.join(BASE_DIR, "knowledge", domain)
    os.makedirs(domain_dir, exist_ok=True)

    index_path = os.path.join(domain_dir, "academic.index")
    meta_path = os.path.join(domain_dir, "academic_meta.json")

    academic_index = load_index(index_path)
    academic_meta = load_meta(meta_path)

    now_year = datetime.now().year

    default_metadata = {
        "source": "unknown",
        "type": "notes",
        "year": now_year,
        "ingested_at": now_year
    }

    full_metadata = {**default_metadata, **metadata}

    required = ["type", "year"]
    for field in required:
        if not full_metadata.get(field):
            raise ValueError(f"Missing required metadata: {field}")

    vecs = embed_batch(texts)
    if vecs is None:
        return False

    academic_index.add(vecs)

    for text in texts:
        academic_meta.append({
            "text": text,
            **full_metadata
        })

    faiss.write_index(academic_index, index_path)
    atomic_json_save(meta_path, academic_meta)

    return True
