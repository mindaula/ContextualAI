"""
Minimal interactive CLI entrypoint for ContextualAI.

Architectural role:
- Provides a terminal-only interface over the core engine.
- Displays memory index counts for operator visibility at startup.
- Delegates prompt processing to `app.core.engine.process_message`.

Interface responsibilities:
- Accept stdin prompts and render assistant output to stdout.
- Handle local session control commands.
- Select a default knowledge domain at startup when available.

Request lifecycle (per user turn, CLI):
1. Read a single line from stdin.
2. Handle local control commands (`exit`/`quit`, `empty chat`/`clear chat`).
3. Forward regular prompts to `app.core.engine.process_message`.
4. Render streamed iterables or single response values to stdout.

Input validation behavior:
- Empty input is ignored and does not call core.
- Startup domain selection failure falls back to `None` domain.

Hard trigger handling:
- Only local CLI commands listed above are handled here.
- Route triggers such as `/memory`, `/academic`, `/general` are not parsed here.

Error handling strategy:
- Handles EOF and keyboard interrupts without traceback output.
- Handles memory-status read failures with fallback status text.

Response formatting:
- Prints chunked responses incrementally when iterable output is returned.
- Prints scalar response values directly.

Side effects:
- Reads knowledge directory to pick a default domain.
- Prints memory/session metadata to stdout.
- Calls memory session archival/discard operations via conversation manager.

Determinism considerations:
- Default domain selection depends on filesystem directory order after sorting.
- Output timing and chunk boundaries depend on engine return behavior.
"""

import sys
import asyncio
import os

from app.memory.memory_system import personal_index
from app.memory.conversation_memory import index as conversation_index
import app.memory.conversation_manager as conversation_manager
from app.core.engine import process_message


# =========================================================
# UTF-8 SAFE STDOUT
# Configures best-effort UTF-8 console output without failing startup.
# =========================================================

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    except Exception:
        pass


# =========================================================
# MAIN APPLICATION LOOP
# =========================================================

def main():
    """
    Run the interactive terminal session.

    Error handling strategy:
    - Startup domain probing failures fall back to `None` domain.
    - Memory status probing failures print a fallback message.
    - EOF and keyboard interrupts are handled gracefully.

    Interaction with core:
    - Calls `process_message(question, domain=active_domain)` for non-control
      user inputs.
    """
    active_domain = None
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        knowledge_dir = os.path.join(base_dir, "knowledge")
        if os.path.isdir(knowledge_dir):
            domains = sorted(
                d for d in os.listdir(knowledge_dir)
                if os.path.isdir(os.path.join(knowledge_dir, d))
            )
            if domains:
                active_domain = domains[0]
    except Exception:
        active_domain = None

    print("Advanced AI Tutor started. (Type 'exit' to quit)\n")
    print("-" * 60)

    # -----------------------------------------------------
    # Memory status overview (observability only; no control-flow impact)
    # -----------------------------------------------------
    try:
        print("MEMORY STATUS:")
        print("Academic chunks loaded: domain-scoped")
        print(f"Personal facts loaded:  {personal_index.ntotal}")
        print(f"Conversation messages loaded: {conversation_index.ntotal}")
        print("-" * 60)
    except Exception:
        print("Memory status unavailable.")
        print("-" * 60)

    # -----------------------------------------------------
    # Interactive loop
    # -----------------------------------------------------
    while True:

        try:
            question = input("Question: ").strip()

        except EOFError:
            print("\nSession discarded.")
            conversation_manager.discard_current_session()
            break

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            # Session is explicitly persisted on normal exit.
            print("Archiving session...")
            conversation_manager.archive_session()
            print("Shutting down.")
            break

        if question.lower() in ("empty chat", "clear chat"):
            # Session is explicitly discarded on clear commands.
            conversation_manager.discard_current_session()
            print("Chat cleared.")
            continue

        print("\nResponse:\n")

        response = asyncio.run(process_message(question, domain=active_domain))
        if hasattr(response, "__iter__") and not isinstance(response, (str, bytes)):
            for chunk in response:
                print(chunk, end="", flush=True)
            print()
        elif response is not None:
            print(response)

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
