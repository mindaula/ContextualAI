import sys

from app.memory_system import academic_index, personal_index
from app.conversation_memory import index as conversation_index
from app import conversation_manager
from app.engine import process_message


# Ensure UTF-8 safe console output (prevents encoding issues on some systems)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    except Exception:
        pass


def main():
    """
    Command-line interface entry point for the AI tutor.

    Initializes memory statistics and starts an interactive
    chat loop with session management support.
    """

    print("Advanced AI Tutor started. (Type 'exit' to quit)\n")
    print("-" * 60)

    try:
        print("MEMORY STATUS:")
        print(f"Academic chunks loaded: {academic_index.ntotal}")
        print(f"Personal facts loaded:  {personal_index.ntotal}")

        try:
            chat_history = conversation_manager.get_recent_messages(limit=1000)
            print(f"Conversation messages loaded: {len(chat_history)}")
        except Exception:
            print("Conversation messages loaded: 0")

        print("-" * 60)

    except Exception as e:
        print(f"Memory counter error: {e}")
        print("-" * 60)

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

        if question.lower() in ("exit", "quit"):
            print("Saving session...")
            conversation_manager.archive_session()
            print("Shutting down.")
            break

        if question.lower() in ("empty chat", "clear chat"):
            conversation_manager.discard_current_session()
            print("Chat cleared.")
            continue

        print("\nResponse:\n")

        response = process_message(question)

        # Streaming support:
        # If the engine returns a generator, stream tokens incrementally.
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
