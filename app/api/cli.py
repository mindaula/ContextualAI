"""
Full-featured interactive CLI adapter for ContextualAI.

Architectural role:
- Exposes terminal interaction, including runtime domain switching.
- Provides startup observability for domain indexes and memory state.
- Delegates all model reasoning to `app.core.engine.process_message`.

Interface responsibilities:
- Resolve and maintain the active domain pointer.
- Expose local hard trigger commands for session/domain control.
- Render engine output in streamed or scalar form.

Request lifecycle (per user turn, CLI):
1. Read stdin.
2. Handle local control commands (`exit`/`quit`, `empty chat`/`clear chat`, `/mode`).
3. Route normal text prompts to the core engine with the active domain.
4. Print streamed chunks or scalar responses.

Input validation behavior:
- Empty input is ignored.
- `/mode` validates requested domain against discovered domain names.
- Startup aborts if no default domain can be established.

Hard trigger handling:
- Implements only `/mode` as a local command.
- Does not implement `/memory`, `/academic`, or `/general` routing here.

Error handling strategy:
- Domain index read failures degrade to zero-count status reporting.
- EOF and keyboard interrupts terminate loop without traceback output.

Response formatting:
- Iterable outputs are printed chunk-by-chunk.
- Non-iterable outputs are printed as a single value.

Side effects:
- Reads `knowledge/` directory and FAISS index files for status metrics.
- Writes to stdout extensively for operator feedback.
- Archives or discards active conversation sessions via memory manager.

Determinism considerations:
- Default/current domain follows sorted filesystem domain names.
- Chunk output segmentation depends on engine iterable behavior.
"""

from dotenv import load_dotenv

load_dotenv()

import sys
import os
import asyncio
import faiss

import app.memory.memory_system as memory_system
from app.memory.conversation_memory import index as conversation_index
import app.memory.conversation_manager as conversation_manager
from app.core.engine import process_message


# =========================================================
# DOMAIN MANAGEMENT
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")

current_domain = None


def get_available_domains():
    """Return sorted domain names discovered under `knowledge/`."""
    if not os.path.exists(KNOWLEDGE_DIR):
        return []

    return sorted([
        d for d in os.listdir(KNOWLEDGE_DIR)
        if os.path.isdir(os.path.join(KNOWLEDGE_DIR, d))
    ])


def get_default_domain():
    """Return first available domain or raise when none are present."""
    domains = get_available_domains()
    if not domains:
        raise RuntimeError("No domains found in knowledge/")
    return domains[0]


def set_domain(domain_name):
    """Update global domain pointer when the requested domain exists."""
    global current_domain

    domains = get_available_domains()
    if domain_name not in domains:
        return False

    current_domain = domain_name
    return True


def get_domain_chunk_count(domain):
    """
    Return FAISS vector count for `<domain>/academic.index`.

    Error handling strategy:
    - Missing index path -> 0
    - Index read failures -> 0
    """
    index_path = os.path.join(KNOWLEDGE_DIR, domain, "academic.index")

    if not os.path.exists(index_path):
        return 0

    try:
        index = faiss.read_index(index_path)
        return index.ntotal
    except Exception:
        return 0


# =========================================================
# UTF-8 SAFE OUTPUT
# Best-effort stdout encoding normalization for interactive terminals.
# =========================================================

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    except Exception:
        pass


# =========================================================
# MAIN
# =========================================================

def main():
    """
    Run CLI loop with domain controls and session lifecycle commands.

    Endpoint-equivalent responsibilities:
    - Resolve initial active domain.
    - Display domain and memory status metadata.
    - Dispatch non-control user text to core.

    Error handling strategy:
    - Domain initialization failure aborts startup with message.
    - Conversation-memory read failures degrade to zero-count reporting.
    - EOF/interrupt are handled without stack traces.
    """
    global current_domain

    try:
        current_domain = get_default_domain()
    except Exception as e:
        print(f"Domain initialization error: {e}")
        return

    print("Advanced AI Tutor started. (Type 'exit' to quit)")
    print(f"Active domain: {current_domain}\n")
    print("-" * 60)

    # =====================================================
    # Memory status for all discovered domains (observability only)
    # =====================================================

    print("DOMAIN MEMORY STATUS:\n")

    domains = get_available_domains()
    for d in domains:
        chunks = get_domain_chunk_count(d)
        marker = " (active)" if d == current_domain else ""
        print(f"{d}: {chunks} academic chunks{marker}")

    print(f"\nPersonal facts loaded: {memory_system.personal_index.ntotal}")

    try:
        print(f"Conversation index entries loaded: {conversation_index.ntotal}")
        chat_history = conversation_manager.get_recent_messages(limit=1000)
        print(f"Current session messages loaded: {len(chat_history)}")
    except Exception:
        print("Conversation index entries loaded: 0")
        print("Current session messages loaded: 0")

    print("-" * 60)

    # =====================================================
    # Input loop
    # =====================================================

    while True:

        try:
            question = input("Question: ").strip()

        except EOFError:
            print("\nChat discarded (EOF received).")
            conversation_manager.discard_current_session()
            break

        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break

        if not question:
            continue

        # EXIT
        if question.lower() in ("exit", "quit"):
            # Explicitly persist session state before process exit.
            print("Saving session...")
            conversation_manager.archive_session()
            print("Shutting down.")
            break

        # CLEAR CHAT
        if question.lower() in ("empty chat", "clear chat"):
            # Explicitly drop in-memory current session context.
            conversation_manager.discard_current_session()
            print("Chat cleared.")
            continue

        # MODE COMMAND (local hard trigger for domain management)
        if question.lower().startswith("/mode"):
            parts = question.split()

            if len(parts) == 1 or parts[1].lower() == "help":
                print("\nAvailable domains:")
                for d in domains:
                    print(f" - {d}")
                print("\nUsage:")
                print(" /mode <domain_name>")
                print(" /mode help")
                print(f"\nCurrent domain: {current_domain}\n")
                continue

            new_domain = parts[1]

            if set_domain(new_domain):
                print(f"\nSwitched to domain: {current_domain}\n")
            else:
                print(f"\nDomain '{new_domain}' not found.\n")

            continue

        # NORMAL QUESTION FLOW

        print("\nResponse:\n")

        response = asyncio.run(process_message(question, domain=current_domain))

        if hasattr(response, "__iter__") and not isinstance(response, str):
            full_text = ""
            for chunk in response:
                print(chunk, end="", flush=True)
                full_text += chunk
            print()
        else:
            print(response)

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
