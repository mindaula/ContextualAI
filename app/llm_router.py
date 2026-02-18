import re
import numpy as np
import faiss

from app.embedding_model import get_model


# =========================================================
# CONFIG
# =========================================================

INTENT_THRESHOLD = 0.50
RELATIVE_MARGIN = 0.08
LOW_INFO_TOKEN_LIMIT = 2


# =========================================================
# PATTERNS (Multilingual: EN / DE / ES / FR)
# =========================================================

PERSONAL_STORE_PATTERNS = [

    # German
    r"\bich\s+bin\b",
    r"\bmein\s+name\s+ist\b",
    r"\bich\s+studiere\b",
    r"\bich\s+wohne\b",
    r"\bich\s+arbeite\b",

    # English
    r"\bi\s+am\b",
    r"\bmy\s+name\s+is\b",
    r"\bi\s+study\b",
    r"\bi\s+live\b",
    r"\bi\s+work\b",

    # Spanish
    r"\bsoy\b",
    r"\bme\s+llamo\b",
    r"\bestudio\b",
    r"\bvivo\b",
    r"\btrabajo\b",

    # French
    r"\bje\s+suis\b",
    r"\bje\s+m'appelle\b",
    r"\bj'étudie\b",
    r"\bj'habite\b",
    r"\bje\s+travaille\b",
]

PERSONAL_QUERY_PATTERNS = [

    # German
    r"\bwer\s+bin\s+ich\b",
    r"\bwie\s+hei[ßs]e\s+ich\b",
    r"\bwo\s+wohne\s+ich\b",
    r"\bwo\s+arbeite\s+ich\b",
    r"\bwas\s+wei[sß]t\s+du\s+uber\s+mich\b",

    # English
    r"\bwho\s+am\s+i\b",
    r"\bwhat\s+is\s+my\s+name\b",
    r"\bwhere\s+do\s+i\s+live\b",
    r"\bwhere\s+do\s+i\s+work\b",
    r"\bwhat\s+do\s+you\s+know\s+about\s+me\b",

    # Spanish
    r"\bquien\s+soy\b",
    r"\bcomo\s+me\s+llamo\b",
    r"\bdonde\s+vivo\b",
    r"\bdonde\s+trabajo\b",
    r"\bque\s+sabes\s+sobre\s+mi\b",

    # French
    r"\bqui\s+suis[- ]?je\b",
    r"\bcomment\s+je\s+m'appelle\b",
    r"\bou\s+j'habite\b",
    r"\bou\s+je\s+travaille\b",
    r"\bque\s+sais[- ]?tu\s+de\s+moi\b",
]

CONVERSATION_PATTERNS = [

    # German
    r"\bwas\s+haben\s+wir\b",
    r"\bworub\w*\s+haben\s+wir\b",
    r"\bworan\s+erinnerst\s+du\s+dich\b",
    r"\bim\s+letz\w*\s+chat\b",
    r"\bhaben\s+wir\s+(geredet|gesprochen|besprochen)\b",

    # English
    r"\bwhat\s+did\s+we\s+(talk|discuss)\b",
    r"\bwhat\s+have\s+we\s+discussed\b",
    r"\bdo\s+you\s+remember\b",
    r"\bin\s+the\s+last\s+chat\b",
    r"\bour\s+previous\s+chat\b",

    # Spanish
    r"\bque\s+hemos\s+(hablado|discutido)\b",
    r"\ben\s+el\s+ultimo\s+chat\b",
    r"\brecuerdas\b",

    # French
    r"\bde\s+quoi\s+avons[- ]?nous\s+parle\b",
    r"\bdans\s+le\s+dernier\s+chat\b",
    r"\bte\s+souviens[- ]?tu\b",
]

KNOWLEDGE_PATTERNS = [

    # German
    r"^was\s+ist\b",
    r"^was\s+bedeutet\b",
    r"^erklare\b",
    r"^wie\s+funktioniert\b",
    r"^wofur\s+ist\b",
    r"^definiere\b",

    # English
    r"^what\s+is\b",
    r"^what\s+does\b",
    r"^explain\b",
    r"^how\s+does\b",
    r"^define\b",

    # Spanish
    r"^que\s+es\b",
    r"^explica\b",
    r"^como\s+funciona\b",
    r"^define\b",

    # French
    r"^qu['e]st[- ]?ce\s+que\b",
    r"^explique\b",
    r"^comment\s+fonctionne\b",
    r"^definis\b",
]


# =========================================================
# HELPERS
# =========================================================

def normalize(text: str) -> str:
    return text.lower().strip()


def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())


def match_any(patterns, text):
    return any(re.search(p, text) for p in patterns)


def contains_self_reference(tokens):
    return any(token in tokens for token in ["ich", "i", "yo", "je"])


# =========================================================
# MODEL
# =========================================================

model = get_model()
dimension = model.get_sentence_embedding_dimension()


# =========================================================
# NEURAL PROTOTYPES
# =========================================================

INTENT_PROTOTYPES = {
    "personal_store": [
        "I am 25 years old.",
        "Ich studiere Informatik.",
        "Vivo en Madrid.",
        "Je m'appelle Max."
    ],
    "personal_query": [
        "What is my name?",
        "Wie heiße ich?",
        "Donde vivo?",
        "Qui suis-je?"
    ],
    "conversation_query": [
        "What did we discuss?",
        "Was haben wir besprochen?",
        "Que hemos hablado?",
        "De quoi avons-nous parle?"
    ],
    "academic": [
        "What is TCP?",
        "Was ist netstat?",
        "Explica un algoritmo.",
        "Comment fonctionne Linux?"
    ],
    "general": [
        "How are you?",
        "Wie geht es dir?",
        "Hola que tal?",
        "Bonjour"
    ]
}

intent_labels = []
intent_texts = []

for label, texts in INTENT_PROTOTYPES.items():
    for t in texts:
        intent_labels.append(label)
        intent_texts.append("passage: " + normalize(t))

vectors = model.encode(intent_texts)
vectors = np.array(vectors).astype("float32")
faiss.normalize_L2(vectors)

intent_index = faiss.IndexFlatIP(dimension)
intent_index.add(vectors)


# =========================================================
# NEURAL SCORING
# =========================================================

def neural_scores(question: str):

    q = normalize(question)

    qvec = model.encode(["query: " + q])
    qvec = np.array(qvec).astype("float32")
    faiss.normalize_L2(qvec)

    scores, indices = intent_index.search(qvec, len(intent_labels))

    class_scores = {label: [] for label in INTENT_PROTOTYPES}

    for score, idx in zip(scores[0], indices[0]):
        label = intent_labels[idx]
        class_scores[label].append(float(score))

    return {
        label: max(vals) if vals else 0.0
        for label, vals in class_scores.items()
    }


# =========================================================
# FINAL ROUTER
# =========================================================

def decide_route(question: str, return_confidence=False, last_route: str = None):

    if not question or not question.strip():
        return ("general", 0.0) if return_confidence else "general"

    q = normalize(question)
    tokens = tokenize(q)
    token_count = len(tokens)

    if match_any(PERSONAL_STORE_PATTERNS, q):
        return ("personal_store", 1.0) if return_confidence else "personal_store"

    if match_any(PERSONAL_QUERY_PATTERNS, q):
        return ("personal_query", 1.0) if return_confidence else "personal_query"

    if match_any(CONVERSATION_PATTERNS, q):
        return ("conversation_query", 1.0) if return_confidence else "conversation_query"

    if match_any(KNOWLEDGE_PATTERNS, q):
        return ("academic", 0.99) if return_confidence else "academic"

    if token_count <= LOW_INFO_TOKEN_LIMIT:
        if last_route == "conversation_query":
            return ("conversation_query", 0.7) if return_confidence else "conversation_query"
        return ("general", 0.99) if return_confidence else "general"

    scores = neural_scores(question)

    if last_route == "conversation_query":
        if token_count <= 3 and scores["conversation_query"] > 0.40:
            scores["conversation_query"] += 0.08

    if contains_self_reference(tokens) and scores["personal_query"] > 0.35:
        scores["personal_query"] += 0.03

    sorted_labels = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    best_label, best_score = sorted_labels[0]
    second_score = sorted_labels[1][1]

    if scores["conversation_query"] > 0.42 and \
       scores["conversation_query"] > scores["academic"] * 0.95:
        return ("conversation_query", scores["conversation_query"]) \
            if return_confidence else "conversation_query"

    if best_score < INTENT_THRESHOLD:
        return ("general", best_score) if return_confidence else "general"

    if best_score * (1 - RELATIVE_MARGIN) < second_score:
        return ("general", best_score) if return_confidence else "general"

    return (best_label, best_score) if return_confidence else best_label
