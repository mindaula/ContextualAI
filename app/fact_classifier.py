import json
import re
from generator import generate_answer


# =========================================================
# HELPER: JSON CLEANER
# =========================================================

def _extract_json(text: str):
    if not text:
        return None

    text = text.strip()

    # Remove Markdown code blocks
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    # Extract first JSON object
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        return match.group(0)

    return None


# =========================================================
# PERSONAL FACT CLASSIFIER
# =========================================================

def classify_and_extract_fact(text: str):

    lower = text.lower().strip()

    # Multilingual personal fact patterns (EN / DE / ES / FR)
    personal_patterns = [

        # Identity
        "i am", "my name is", "i am called",
        "ich bin", "ich heiße", "mein name ist",
        "soy", "me llamo",
        "je suis", "je m'appelle",

        # Location
        "i live", "i come from", "i am from",
        "ich wohne", "ich komme aus", "ich bin aus",
        "vivo en", "soy de",
        "j'habite", "je viens de",

        # Work / Education
        "i work", "i study", "i am studying", "i am learning",
        "ich arbeite", "ich studiere", "ich lerne",
        "trabajo", "estudio",
        "je travaille", "j'étudie",

        # Possession
        "my ", "mine ",
        "mein ", "meine ", "meiner ", "meinem ",
        "mi ", "mis ",
        "mon ", "ma ", "mes ",

        # Family
        "my mother", "my father", "my brother", "my sister",
        "meine mutter", "mein vater", "mein bruder", "meine schwester",
        "mi madre", "mi padre", "mi hermano", "mi hermana",
        "ma mère", "mon père", "mon frère", "ma sœur",

        # Pets
        "my dog", "my cat", "my pet",
        "mein hund", "meine katze", "mein haustier",
        "mi perro", "mi gato", "mi mascota",
        "mon chien", "mon chat", "mon animal",

        # Generic animal keywords
        "dog", "cat", "pet",
        "hund", "katze", "haustier",
        "perro", "gato", "mascota",
        "chien", "chat", "animal",

        # Relationship
        "my girlfriend", "my boyfriend", "my partner",
        "meine freundin", "mein freund", "mein partner",
        "mi novia", "mi novio", "mi pareja",
        "ma copine", "mon copain", "mon partenaire",

        # Health / Condition
        "i have", "i was diagnosed",
        "ich habe", "mir wurde diagnostiziert",
        "tengo", "me diagnosticaron",
        "j'ai", "on m'a diagnostiqué",

        # Hobbies
        "my hobby", "my hobbies", "i play", "i train",
        "mein hobby", "meine hobbys", "ich spiele", "ich trainiere",
        "mi hobby", "mis hobbies", "juego",
        "mon hobby", "mes hobbies", "je joue"
    ]

    # Strong possession-based facts (multilingual)
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

    # Pattern-based detection
    if any(p in lower for p in personal_patterns):
        return {
            "is_fact": True,
            "confidence": 0.93,
            "fact_text": text.strip()
        }

    # LLM fallback
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
        return {
            "is_fact": False,
            "confidence": 0.0,
            "fact_text": ""
        }

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
        return {
            "is_fact": False,
            "confidence": 0.0,
            "fact_text": ""
        }


# =========================================================
# SMALLTALK FILTER
# =========================================================

def _is_smalltalk(q: str):

    q = q.lower()

    SMALLTALK = [

        # English
        "hello", "hi", "hey", "how are you", "what's up",

        # German
        "hallo", "wie geht", "alles gut", "moin", "servus",

        # Spanish
        "hola", "qué tal", "como estas",

        # French
        "bonjour", "salut", "ça va"
    ]

    return any(w in q for w in SMALLTALK)


# =========================================================
# KNOWLEDGE NEED CLASSIFIER
# =========================================================

def classify_knowledge_need(question: str):

    q_lower = question.lower()

    # Smalltalk
    if _is_smalltalk(question):
        return "A"

    # Personal
    personal_patterns = [

        # English
        "i am", "am i", "who am i", "my name", "what is my name",

        # German
        "ich bin", "wer bin ich", "wie heiße ich", "mein name",

        # Spanish
        "soy", "quién soy", "como me llamo", "mi nombre",

        # French
        "je suis", "qui suis-je", "comment je m'appelle", "mon nom"
    ]

    if any(p in q_lower for p in personal_patterns):
        return "C"

    # Technical keywords (multilingual)
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

    # LLM fallback
    prompt = f"""
Classify knowledge need.

A = Smalltalk or general knowledge
B = Technical or academic explanation
C = Personal user information

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

    match = re.search(r"[ABC]", response.strip().upper())

    if match:
        return match.group(0)

    return "A"
