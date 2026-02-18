import sys

from app.memory_system import academic_index, personal_index
from app.conversation_memory import index as conversation_index
from app import conversation_manager
from app.engine import process_message


# =========================================================
# UTF-8 SAFE STDOUT
# =========================================================
# Ensures proper UTF-8 output handling across environments

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    except Exception:
        pass


# =========================================================
# MAIN APPLICATION LOOP
# =========================================================

def main():

    print("Advanced AI Tutor started. (Type 'exit' to quit)\n")
    print("-" * 60)

    # -----------------------------------------------------
    # Memory Status Overview
    # -----------------------------------------------------
    try:
        print("MEMORY STATUS:")
        print(f"Academic chunks loaded: {academic_index.ntotal}")
        print(f"Personal facts loaded:  {personal_index.ntotal}")
        print(f"Conversation messages loaded: {conversation_index.ntotal}")
        print("-" * 60)
    except Exception:
        print("Memory status unavailable.")
        print("-" * 60)

    # -----------------------------------------------------
    # Interactive Loop
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
            print("Archiving session...")
            conversation_manager.archive_session()
            print("Shutting down.")
            break

        if question.lower() in ("empty chat", "clear chat"):
            conversation_manager.discard_current_session()
            print("Chat cleared.")
            continue

        print("\nResponse:\n")

        process_message(question)

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
