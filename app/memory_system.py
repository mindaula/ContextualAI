import os
import json
import re
import numpy as np
import faiss
from datetime import datetime

from app.embedding_model import get_model


# =========================================================
# CONFIGURATION (BALANCED CALIBRATION)
# =========================================================

MAX_PERSONAL_FACTS = 200

PERSONAL_MIN_SIMILARITY = 0.60
PERSONAL_DUPLICATE_THRESHOLD = 0.85

ACADEMIC_MIN_SIMILARITY = 0.72
RELATIVE_SCORE_RATIO = 0.90
MIN_QUERY_TOKENS_FOR_ACADEMIC = 2


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(text: str):
    if not text:
        return ""

    text = str(text).lower()
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_count(text: str):
    return len(normalize_text(text).split())


# =========================================================
# STORAGE PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PERSONAL_INDEX_FILE = os.path.join(BASE_DIR, "personal.index")
PERSONAL_META_FILE = os.path.join(BASE_DIR, "personal_meta.json")

ACADEMIC_INDEX_FILE = os.path.join(BASE_DIR, "academic.index")
ACADEMIC_META_FILE = os.path.join(BASE_DIR, "academic_meta.json")


# =========================================================
# MODEL INITIALIZATION
# =========================================================

model = get_model()
dimension = model.get_sentence_embedding_dimension()


# =========================================================
# FILE UTILITIES
# =========================================================

def atomic_json_save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_index(path):
    if os.path.exists(path):
        try:
            return faiss.read_index(path)
        except Exception:
            pass
    return faiss.IndexFlatIP(dimension)


def load_meta(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


personal_index = load_index(PERSONAL_INDEX_FILE)
personal_meta = load_meta(PERSONAL_META_FILE)

academic_index = load_index(ACADEMIC_INDEX_FILE)
academic_meta = load_meta(ACADEMIC_META_FILE)


# =========================================================
# EMBEDDING HELPERS
# =========================================================

def embed(text: str, is_query=False):
    if not text or not str(text).strip():
        return None

    text = normalize_text(text)
    prefix = "query: " if is_query else "passage: "

    vec = model.encode([prefix + text])
    vec = np.array(vec).astype("float32")
    faiss.normalize_L2(vec)
    return vec


def embed_batch(texts):
    clean = [normalize_text(t) for t in texts if t and str(t).strip()]
    if not clean:
        return None

    prefixed = ["passage: " + t for t in clean]
    vecs = model.encode(prefixed)
    vecs = np.array(vecs).astype("float32")
    faiss.normalize_L2(vecs)
    return vecs


# =========================================================
# PERSONAL SEARCH
# =========================================================

def search_personal(query, top_k=3, return_scores=False):

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


# =========================================================
# ACADEMIC SEARCH (RECENCY BOOSTED)
# =========================================================

def search_academic(query, top_k=5, return_scores=False):

    if token_count(query) < MIN_QUERY_TOKENS_FOR_ACADEMIC:
        return []

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


# =========================================================
# PERSONAL STORE
# =========================================================

def add_personal_fact(text: str):

    global personal_index, personal_meta

    if not text or not str(text).strip():
        return False

    text = text.strip()

    if personal_index.ntotal > 0:
        existing = search_personal(text, top_k=1, return_scores=True)

        if existing:
            existing_text, score = existing[0]

            if score >= PERSONAL_DUPLICATE_THRESHOLD:
                try:
                    idx_to_remove = next(
                        i for i, m in enumerate(personal_meta)
                        if m["text"] == existing_text
                    )

                    personal_meta.pop(idx_to_remove)
                    personal_index = faiss.IndexFlatIP(dimension)

                    if personal_meta:
                        vecs = embed_batch([m["text"] for m in personal_meta])
                        if vecs is not None:
                            personal_index.add(vecs)

                except StopIteration:
                    pass

    vec = embed(text, is_query=False)
    if vec is None:
        return False

    personal_index.add(vec)

    personal_meta.append({
        "text": text
    })

    if len(personal_meta) > MAX_PERSONAL_FACTS:
        personal_meta.pop(0)

    faiss.write_index(personal_index, PERSONAL_INDEX_FILE)
    atomic_json_save(PERSONAL_META_FILE, personal_meta)

    return True


# =========================================================
# ACADEMIC STORE
# =========================================================

def add_academic_chunks(texts, metadata: dict):

    global academic_index, academic_meta

    if not texts:
        return False

    now_year = datetime.now().year

    default_metadata = {
        "source": "unknown",
        "type": "notes",
        "year": now_year,
        "ingested_at": now_year
    }

    full_metadata = {**default_metadata, **metadata}

    required = ["source", "type", "year"]
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

    faiss.write_index(academic_index, ACADEMIC_INDEX_FILE)
    atomic_json_save(ACADEMIC_META_FILE, academic_meta)

    return True
