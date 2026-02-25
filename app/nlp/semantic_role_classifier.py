"""Semantic-role classifier using embedding similarity plus rule guards.

Intent classification logic:
- Encodes normalized user text and compares against role prototypes via FAISS.
- Aggregates per-class max similarity, then applies margin/threshold rules.
- Refines ambiguous cases using structure and first-person marker heuristics.

Normalization steps:
- Lowercasing, contraction expansion, accent folding, and whitespace collapse.
- Query prefixing (`"query: "`) before embedding to keep vector format stable.

Interaction with core and memory:
- Imported by `intent_router` and `query_rewriter`, which are consumed by core.
- Uses `app.memory.embedding_model.get_model()` for embedding inference.

Temporal handling:
- None. Classification uses only current text input.

Determinism:
- Rule layer is deterministic.
- Model similarity depends on embedding model/runtime determinism.

Failure handling:
- Empty/blank input returns (`"other"`, `0.0`).
"""

import numpy as np
import faiss
import re
from app.memory.embedding_model import get_model
from app.nlp.structure_classifier import classify_structure


# =========================================================
# CONFIG (Thresholds)
# =========================================================

SELF_DISCLOSURE_THRESHOLD = 0.78
SELF_QUERY_THRESHOLD = 0.75
CONVERSATION_THRESHOLD = 0.75

MARGIN = 0.03
MIN_SELF_QUERY_CONF = 0.80


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text: str) -> str:
    """
    Normalize multilingual user text before prototype similarity scoring.

    Edge cases:
    - `None`/empty input returns an empty string.
    """
    if not text:
        return ""

    t = text.strip().lower()

    contractions = {
        "what's": "what is",
        "whats": "what is",
        "who's": "who is",
        "whos": "who is",
        "where's": "where is",
        "wheres": "where is",
        "how's": "how is",
        "hows": "how is",
        "i'm": "i am",
        "im": "i am",
        "i've": "i have",
        "ive": "i have",
    }

    for k, v in contractions.items():
        t = t.replace(k, v)

    t = (
        t.replace("ä", "ae")
         .replace("ö", "oe")
         .replace("ü", "ue")
         .replace("é", "e")
         .replace("è", "e")
         .replace("ê", "e")
         .replace("á", "a")
         .replace("í", "i")
         .replace("ó", "o")
         .replace("ú", "u")
         .replace("¿", "")
         .replace("¡", "")
    )

    t = re.sub(r"\s+", " ", t).strip()
    return t


# =========================================================
# MODEL
# =========================================================

model = get_model()
dimension = model.get_sentence_embedding_dimension()


# =========================================================
# PROTOTYPES
# =========================================================

PROTOTYPES = {
    "self_disclosure": [
        "Ich heiße Max.",
        "Mein Name ist Anna.",
        "Ich bin 25 Jahre alt.",
        "Ich arbeite als Entwickler.",
        "Ich wohne in Berlin.",
        "My name is John.",
        "I am 30 years old.",
        "I work as a developer.",
        "I live in London.",
    ],

    "self_query": [

        # -------------------------
        # German
        # -------------------------
        "Wie heiße ich?",
        "Was ist mein Name?",
        "Wo wohne ich?",
        "Wie alt bin ich?",
        "Wo arbeite ich?",
        "Was habe ich dir gesagt?",
        "Was weißt du über mich?",
        "Was hast du über mich gespeichert?",
        "Erinnerst du dich an mich?",
        "Was sind meine Hobbys?",
        "Welche Informationen hast du über mich?",
        "Was ist meine Adresse?",
        "Wann habe ich Geburtstag?",

        # -------------------------
        # English
        # -------------------------
        "What is my name?",
        "Where do I live?",
        "How old am I?",
        "Where do I work?",
        "What did I tell you?",
        "What do you know about me?",
        "What have you stored about me?",
        "Do you remember me?",
        "What are my hobbies?",
        "What information do you have about me?",
        "What is my address?",
        "When is my birthday?",

        # -------------------------
        # Spanish
        # -------------------------
        "¿Cómo me llamo?",
        "¿Dónde vivo?",
        "¿Cuántos años tengo?",
        "¿Dónde trabajo?",
        "¿Qué sabes sobre mí?",
        "¿Qué te dije?",
        "¿Qué información tienes sobre mí?",

        # -------------------------
        # French
        # -------------------------
        "Comment je m'appelle?",
        "Où est-ce que j'habite?",
        "Quel âge ai-je?",
        "Où est-ce que je travaille?",
        "Que sais-tu sur moi?",
        "Qu'est-ce que je t'ai dit?",
    ],

    "conversation_reference": [
        "Worüber haben wir gesprochen?",
        "What did we talk about?",
        "Do you remember our conversation?",
    ],

    "knowledge_question": [
        "Was ist TCP?",
        "Wie funktioniert ein Motor?",
        "What is Linux?",
        "Explain quantum computing.",
    ],

    "smalltalk": [
        "Hallo",
        "Hello",
        "How are you?",
    ]
}


# =========================================================
# BUILD INDEX
# =========================================================

labels = []
texts = []

for label, examples in PROTOTYPES.items():
    for ex in examples:
        labels.append(label)
        texts.append("query: " + normalize_text(ex))

vectors = model.encode(texts)
vectors = np.array(vectors).astype("float32")
faiss.normalize_L2(vectors)

index = faiss.IndexFlatIP(dimension)
index.add(vectors)


# =========================================================
# CLASSIFIER
# =========================================================

def classify_semantic_role(text: str):
    """
    Classify semantic role for routing and rewrite guards.

    Returns:
    - Tuple of (`label`, `confidence_score`).

    Parsing and decision rules:
    - Embedding similarity determines initial candidate class.
    - Low-margin non-personal ties are coerced to `knowledge_question`.
    - `self_query` requires question structure and personal markers; otherwise
      it may be downgraded to `knowledge_question` or shifted to
      `self_disclosure` for first-person non-questions.

    Edge cases:
    - Ambiguous markers can redirect personal-looking text to knowledge route.
    """

    if not text or not text.strip():
        return "other", 0.0

    structure = classify_structure(text)
    q_norm = normalize_text(text)
    qvec = model.encode(["query: " + q_norm])
    qvec = np.array(qvec).astype("float32")
    faiss.normalize_L2(qvec)

    scores, indices = index.search(qvec, len(labels))

    class_scores = {k: 0.0 for k in PROTOTYPES.keys()}

    for score, idx in zip(scores[0], indices[0]):
        label = labels[idx]
        class_scores[label] = max(class_scores[label], float(score))

    sorted_classes = sorted(
        class_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_label, best_score = sorted_classes[0]
    second_score = sorted_classes[1][1] if len(sorted_classes) > 1 else 0.0
    margin = best_score - second_score

    def _debug_print(personal_markers_matched: bool) -> None:
        print(
            "[SEMANTIC DEBUG] "
            f"best_label={best_label}, "
            f"best_score={best_score:.4f}, "
            f"second_best_score={second_score:.4f}, "
            f"margin={margin:.4f}, "
            f"personal_markers_matched={personal_markers_matched}"
        )

    def _return_with_debug(label: str, score: float):
        print(
            "[SEMANTIC DEBUG] "
            f"structure={structure}, "
            f"best_label={best_label}, "
            f"best_score={best_score:.4f}, "
            f"final_label={label}"
        )
        return label, score

    if margin < MARGIN and best_label not in ["self_query", "self_disclosure"]:
        _debug_print(False)
        return _return_with_debug("knowledge_question", best_score)

    if best_label == "self_query":

        personal_markers = [
            " ich ", " mein ", " meine ", " mich ",
            "ueber mich", "über mich",
            " i ", " my ", " me ", " about me ",
            " yo ", " mi ", " mis ", " sobre mi ",
            " je ", "-je", " moi ", " mon ", " ma ", " mes ", " sur moi "
        ]
        # Self-query is only valid for questions
        if structure != "question":

            # Normalize and remove leading discourse markers
            cleaned = q_norm.strip()

            discourse_prefixes = (
                "nein ", "doch ", "ja ", "also ", "aber "
            )

            for prefix in discourse_prefixes:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):]
                    break

            first_person_markers = (
                "ich ", "mein ", "meine ", "meiner ", "meinem ",
                "i ", "my ",
                "yo ", "mi ",
                "je ", "mon ", "ma ", "mes "
            )

            if cleaned.startswith(first_person_markers):
                _debug_print(True)
                return _return_with_debug("self_disclosure", best_score)

            _debug_print(False)
            return _return_with_debug("knowledge_question", best_score)

        # Existing personal marker logic stays unchanged below
        text_check = " " + q_norm + " "
        personal_markers_matched = any(marker in text_check for marker in personal_markers)
        _debug_print(personal_markers_matched)

        if not personal_markers_matched:
            return _return_with_debug("knowledge_question", best_score)

        return _return_with_debug("self_query", best_score)

    if best_label == "self_disclosure":
        _debug_print(False)
        if best_score >= SELF_DISCLOSURE_THRESHOLD:
            return _return_with_debug("self_disclosure", best_score)
        return _return_with_debug("knowledge_question", best_score)

    if best_label == "conversation_reference":
        _debug_print(False)
        if best_score >= CONVERSATION_THRESHOLD:
            return _return_with_debug("conversation_reference", best_score)
        return _return_with_debug("knowledge_question", best_score)

    _debug_print(False)
    return _return_with_debug(best_label, best_score)
