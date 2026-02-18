import requests
import json
import os



# =========================================================
# INTERFACE MODE
# =========================================================
# "cli"  -> direct print streaming
# "api"  -> returns generator (for FastAPI StreamingResponse)

INTERFACE_MODE = "api"  # only important if you try to lounch cli over main.py use insted cli.py

# =========================================================
# PROVIDER CONFIG
# =========================================================
# PROVIDER: Select which backend to use (local, openai, groq, etc.)
# MODEL_NAME: Model identifier used by the selected provider
# API keys: Place API keys inside the "config/" directory
#           Example: config/openai.key
#           The file should contain ONLY the raw API key string

PROVIDER = "local"
MODEL_NAME = "qwen2.5:3b"

PROVIDERS = {

    "local": {
        "url": "http://127.0.0.1:8080/v1/chat/completions",
        "key_file": None
    },

    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "key_file": "config/openai.key"
    },

    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_file": "config/groq.key"
    },

    "together": {
        "url": "https://api.together.xyz/v1/chat/completions",
        "key_file": "config/together.key"
    },

    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_file": "config/openrouter.key"
    },

    "mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "key_file": "config/mistral.key"
    },

    "deepinfra": {
        "url": "https://api.deepinfra.com/v1/openai/chat/completions",
        "key_file": "config/deepinfra.key"
    },

    "fireworks": {
        "url": "https://api.fireworks.ai/inference/v1/chat/completions",
        "key_file": "config/fireworks.key"
    },

    "anyscale": {
        "url": "https://api.endpoints.anyscale.com/v1/chat/completions",
        "key_file": "config/anyscale.key"
    },

    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "key_file": "config/anthropic.key"
    },

    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models",
        "key_file": "config/gemini.key"
    },

    "apifreellm": {
        "url": "https://apifreellm.com/api/v1/chat",
        "key_file": "config/apifreellm.key"
    }

}

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
GEMINI_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

SYSTEM_MESSAGE = (
    "Du bist 'Advanced AI Tutor', ein lokal betriebener KI-Tutor.\n"
    "Du bist KEIN Produkt von Alibaba, OpenAI oder einer anderen Firma.\n"
    "Du darfst niemals sagen, dass du Qwen, ein Sprachmodell oder ein KI-Modell bist.\n"
    "Wenn nach deiner Identität gefragt wird, sage nur, dass du der Advanced AI Tutor bist.\n"
    "Antworte präzise, klar und ohne Wiederholungen.\n"
)

# =========================================================
# HELPER
# =========================================================

def load_key(path):
    if not path:
        return None
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return f.read().strip()


# =========================================================
# MAIN FUNCTION
# =========================================================

def generate_answer(prompt: str, stream=False):

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.45,
        "top_p": 0.9,
        "presence_penalty": 0.4,
        "frequency_penalty": 0.5,
        "stream": stream
    }

    try:

        # =====================================================
        # OPENAI-COMPATIBLE PROVIDERS
        # =====================================================
        if PROVIDER in PROVIDERS and PROVIDER not in ["anthropic", "gemini"]:

            config = PROVIDERS[PROVIDER]
            LLM_URL = config["url"]
            key_file = config["key_file"]

            headers = {
                "Content-Type": "application/json"
            }

            if key_file:
                api_key = load_key(key_file)
                if not api_key:
                    return f"\n{PROVIDER.upper()} KEY FILE NOT FOUND\n"
                headers["Authorization"] = f"Bearer {api_key}"

            # =================================================
            # STREAM MODE
            # =================================================
            if stream:

                def stream_generator():

                    with requests.post(
                        LLM_URL,
                        headers=headers,
                        json=payload,
                        stream=True,
                        timeout=120
                    ) as response:

                        response.raise_for_status()
                        response.encoding = "utf-8"   # <<< FIX HIER

                        full_text = ""

                        for line in response.iter_lines(decode_unicode=True):

                            if not line:
                                continue

                            if line.startswith("data: "):
                                line = line[6:]

                            if line.strip() == "[DONE]":
                                break

                            try:
                                data = json.loads(line)
                            except Exception:
                                continue

                            delta = None

                            if "choices" in data:
                                choice = data["choices"][0]

                                if "delta" in choice and "content" in choice["delta"]:
                                    delta = choice["delta"]["content"]

                                elif "message" in choice and "content" in choice["message"]:
                                    delta = choice["message"]["content"]

                                elif "text" in choice:
                                    delta = choice["text"]

                            elif "message" in data and "content" in data["message"]:
                                delta = data["message"]["content"]

                            if delta:
                                full_text += delta

                                if INTERFACE_MODE == "cli":
                                    print(delta, end="", flush=True)
                                else:
                                    yield delta

                        if INTERFACE_MODE == "cli":
                            return full_text.strip()

                return stream_generator()

            # =================================================
            # NON-STREAM MODE
            # =================================================
            else:

                response = requests.post(
                    LLM_URL,
                    headers=headers,
                    json=payload,
                    timeout=120
                )

                response.raise_for_status()
                data = response.json()

                return data["choices"][0]["message"]["content"].strip()

        # =====================================================
        # ANTHROPIC
        # =====================================================
        elif PROVIDER == "anthropic":

            api_key = load_key("config/anthropic.key")
            if not api_key:
                return "\nANTHROPIC KEY FILE NOT FOUND\n"

            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }

            anthropic_payload = {
                "model": MODEL_NAME,
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": SYSTEM_MESSAGE + "\n\n" + prompt
                    }
                ]
            }

            response = requests.post(
                ANTHROPIC_URL,
                headers=headers,
                json=anthropic_payload,
                timeout=120
            )

            response.raise_for_status()
            data = response.json()

            return data["content"][0]["text"].strip()

        # =====================================================
        # GEMINI
        # =====================================================
        elif PROVIDER == "gemini":

            api_key = load_key("config/gemini.key")
            if not api_key:
                return "\nGEMINI KEY FILE NOT FOUND\n"

            url = GEMINI_URL_TEMPLATE.format(
                model=MODEL_NAME,
                key=api_key
            )

            gemini_payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": SYSTEM_MESSAGE + "\n\n" + prompt}
                        ]
                    }
                ]
            }

            response = requests.post(
                url,
                json=gemini_payload,
                timeout=120
            )

            response.raise_for_status()
            data = response.json()

            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        else:
            return "\nINVALID PROVIDER\n"

    except KeyboardInterrupt:
        return ""

    except Exception as e:
        return f"\nLLM ERROR: {e}\n"
