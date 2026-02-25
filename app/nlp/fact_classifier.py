"""Auxiliary NLP classifiers for personal-fact and knowledge-need detection.

Intent classification logic:
- `classify_and_extract_fact` uses multilingual lexical rules first, then LLM
  JSON fallback when no rule is matched.
- `classify_knowledge_need` uses deterministic A/B/C/D heuristics, then LLM
  fallback to recover unsupported phrasings.

Parsing and normalization:
- Lowercasing and substring pattern checks across DE/EN/ES/FR markers.
- JSON cleaner extracts fenced or inline object snippets from LLM output.

Interaction with core and memory:
- No direct imports from core, memory, or retrieval in this module.
- Functions are utility-style and can be consumed by higher orchestration layers.

Temporal handling:
- None. Classification uses only the current input string.

Determinism:
- Rule paths are deterministic.
- LLM fallback paths are model-dependent; temperature is set to `0.0` for
  stability but full determinism is not guaranteed.

Failure handling:
- LLM/parsing failures degrade to conservative defaults (`is_fact=False`, `A`).
"""

import json
import re
from generator import generate_answer


# =========================================================
# HELPER: JSON CLEANER
# =========================================================

def _extract_json(text: str):
    """Extract first JSON object candidate from raw model output text."""
    if not text:
        return None

    text = text.strip()

    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        return match.group(0)

    return None


# =========================================================
# PERSONAL FACT CLASSIFIER
# =========================================================

def classify_and_extract_fact(text: str):
    """
    Classify whether input is a personal fact and extract normalized payload.

    Edge cases:
    - Prefix matches (`my ...`, `mein ...`, etc.) force high-confidence positive.
    - Broad lexical patterns can over-match domain terms (for example pet words).
    - Invalid/missing LLM JSON returns a safe negative default.
    """

    lower = text.lower().strip()

    personal_patterns = [
        "i am", "my name is", "i am called",
        "ich bin", "ich heiße", "mein name ist",
        "soy", "me llamo",
        "je suis", "je m'appelle",

        "i live", "i come from", "i am from",
        "ich wohne", "ich komme aus", "ich bin aus",
        "vivo en", "soy de",
        "j'habite", "je viens de",

        "i work", "i study", "i am studying", "i am learning",
        "ich arbeite", "ich studiere", "ich lerne",
        "trabajo", "estudio",
        "je travaille", "j'étudie",

        "my ", "mine ",
        "mein ", "meine ", "meiner ", "meinem ",
        "mi ", "mis ",
        "mon ", "ma ", "mes ",

        "my mother", "my father", "my brother", "my sister",
        "meine mutter", "mein vater", "mein bruder", "meine schwester",
        "mi madre", "mi padre", "mi hermano", "mi hermana",
        "ma mère", "mon père", "mon frère", "ma sœur",

        "my dog", "my cat", "my pet",
        "mein hund", "meine katze", "mein haustier",
        "mi perro", "mi gato", "mi mascota",
        "mon chien", "mon chat", "mon animal",

        "dog", "cat", "pet",
        "hund", "katze", "haustier",
        "perro", "gato", "mascota",
        "chien", "chat", "animal",

        "my girlfriend", "my boyfriend", "my partner",
        "meine freundin", "mein freund", "mein partner",
        "mi novia", "mi novio", "mi pareja",
        "ma copine", "mon copain", "mon partenaire",

        "i have", "i was diagnosed",
        "ich habe", "mir wurde diagnostiziert",
        "tengo", "me diagnosticaron",
        "j'ai", "on m'a diagnostiqué",

        "my hobby", "my hobbies", "i play", "i train",
        "mein hobby", "meine hobbys", "ich spiele", "ich trainiere",
        "mi hobby", "mis hobbies", "juego",
        "mon hobby", "mes hobbies", "je joue"
    ]

    if lower.startswith((
        "my ", "mine ",
        "mein ", "meine ", "meiner ", "meinem ",
        "mi ", "mis ",
        "mon ", "ma ", "mes "
    )):
        return {
            "is_fact": True,
            "confidence": 0.95,
            "fact_text": text.strip()
        }

    if any(p in lower for p in personal_patterns):
        return {
            "is_fact": True,
            "confidence": 0.93,
            "fact_text": text.strip()
        }

    prompt = f"""
You analyze a user input.

Respond ONLY with JSON:

{{
"is_fact": true/false,
"confidence": 0-1,
"fact_text": "..."
}}

Input:
{text}
"""

    try:
        response = generate_answer(
            prompt,
            stream=False,
            temperature=0.0,
            max_tokens=120
        )
    except Exception:
        response = ""

    if not response:
        return {"is_fact": False, "confidence": 0.0, "fact_text": ""}

    try:
        clean = _extract_json(response)
        if not clean:
            raise ValueError("No JSON found")

        data = json.loads(clean)

        return {
            "is_fact": bool(data.get("is_fact", False)),
            "confidence": float(data.get("confidence", 0.0)),
            "fact_text": str(data.get("fact_text", "")).strip()
        }

    except Exception:
        return {"is_fact": False, "confidence": 0.0, "fact_text": ""}


# =========================================================
# SMALLTALK FILTER
# =========================================================

def _is_smalltalk(q: str):
    """Detect smalltalk phrases via multilingual substring matching."""

    q = q.lower()

    SMALLTALK = [
        "hello", "hi", "hey", "how are you", "what's up",
        "hallo", "wie geht", "alles gut", "moin", "servus",
        "hola", "qué tal", "como estas",
        "bonjour", "salut", "ça va"
    ]

    return any(w in q for w in SMALLTALK)


# =========================================================
# KNOWLEDGE NEED CLASSIFIER
# =========================================================

def classify_knowledge_need(question: str):
    """
    Assign coarse knowledge-need class:
    - A: smalltalk/general
    - B: technical/academic
    - C: personal user information
    - D: conversation-history reference

    Failure handling:
    - Returns `A` when LLM fallback fails or cannot be parsed.
    """

    q_lower = question.lower()

    # A = Smalltalk
    if _is_smalltalk(question):
        return "A"

    # 🔥 D = Conversation History (MINIMAL FIX)
    CONVERSATION_PATTERNS = [
        "letzten chat",
        "letzte unterhaltung",
        "worüber haben wir",
        "woran erinnerst du dich",
        "last chat",
        "last conversation",
        "what did we talk",
        "de que hablamos",
        "última conversación",
        "dernière conversation",
        "de quoi avons nous parle"
    ]

    if any(p in q_lower for p in CONVERSATION_PATTERNS):
        return "D"

    # C = Personal
    personal_patterns = [
        "i am", "am i", "who am i", "my name", "what is my name",
        "ich bin", "wer bin ich", "wie heiße ich", "mein name",
        "soy", "quién soy", "como me llamo", "mi nombre",
        "je suis", "qui suis-je", "comment je m'appelle", "mon nom"
    ]

    if any(p in q_lower for p in personal_patterns):
        return "C"

    # B = Technical
    TECH_WORDS = [
        "linux", "kernel", "tcp", "udp", "ip",
        "network", "process", "filesystem",
        "netzwerk", "prozess",
        "red", "proceso", "sistema de archivos",
        "réseau", "processus",
        "port", "socket", "bash", "shell",
        "systemd", "chmod", "netstat",
        "firewall", "routing", "dns"
    ]

    if any(word in q_lower for word in TECH_WORDS):
        return "B"

    # LLM fallback (now supports D)
    prompt = f"""
Classify knowledge need.

A = Smalltalk or general knowledge
B = Technical or academic explanation
C = Personal user information
D = Conversation history or previous chat reference

Return only one letter.

Question:
{question}

Answer:
"""

    try:
        response = generate_answer(
            prompt,
            stream=False,
            temperature=0.0,
            max_tokens=5
        )
    except Exception:
        return "A"

    if not response:
        return "A"

    match = re.search(r"[ABCD]", response.strip().upper())

    if match:
        return match.group(0)

    return "A"
