import os
import json
import threading
from datetime import datetime

from app import conversation_memory


# =========================================================
# Configuration
# =========================================================
MAX_TOKENS = 6000  # Maximum allowed tokens per session
BACKUP_DIR = "conversation_backups"
SESSION_LOG_PATH = "current_session.json"


# =========================================================
# Session State
# =========================================================
session_messages = []
current_token_count = 0
last_route = None
session_lock = threading.Lock()


# =========================================================
# Token Estimation
# =========================================================
def estimate_tokens(text):
    """
    Rough token estimation based on character length.

    Assumes approximately 4 characters per token.
    This is used as a lightweight heuristic to prevent
    excessive context growth.
    """
    if not text:
        return 0
    return max(1, len(str(text)) // 4)


# =========================================================
# Crash Recovery
# =========================================================
def recover_unsaved_session():
    """
    Recovers a previously unsaved session if a crash occurred.

    If a session log file exists, it is archived and optionally
    re-indexed into long-term conversation memory.
    """

    if not os.path.exists(SESSION_LOG_PATH):
        return

    try:
        with open(SESSION_LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or not data:
            os.remove(SESSION_LOG_PATH)
            return

        os.makedirs(BACKUP_DIR, exist_ok=True)

        timestamp = datetime.utcnow()
        filename = f"recovered_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(BACKUP_DIR, filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "created_at": timestamp.isoformat(),
                "recovered": True,
                "messages": data
            }, f, indent=2, ensure_ascii=False)

        if data:
            conversation_memory.add_session_to_memory(data)

        os.remove(SESSION_LOG_PATH)

    except Exception:
        pass


# =========================================================
# Message Handling
# =========================================================
def add_message(role, content):
    """
    Adds a message to the current session and updates token usage.

    Automatically archives the session if the token limit is exceeded.
    """
    global current_token_count

    if not content:
        return

    message = {
        "role": str(role),
        "content": str(content)
    }

    should_archive = False

    with session_lock:
        session_messages.append(message)

        try:
            with open(SESSION_LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(session_messages, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        token_estimate = estimate_tokens(content)
        current_token_count += token_estimate

        if current_token_count >= MAX_TOKENS:
            should_archive = True

    if should_archive:
        archive_session()


# =========================================================
# Session Archiving
# =========================================================
def archive_session():
    """
    Archives the current session to disk and indexes it
    into long-term conversation memory.
    """
    global session_messages, current_token_count

    with session_lock:
        if not session_messages:
            return

        messages_copy = list(session_messages)
        total_tokens = current_token_count

        session_messages = []
        current_token_count = 0

    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)

        timestamp = datetime.utcnow()
        filename = f"session_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(BACKUP_DIR, filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "created_at": timestamp.isoformat(),
                "total_tokens": total_tokens,
                "messages": messages_copy
            }, f, indent=2, ensure_ascii=False)

    except Exception:
        pass

    try:
        if messages_copy:
            conversation_memory.add_session_to_memory(messages_copy)
    except Exception:
        pass

    try:
        if os.path.exists(SESSION_LOG_PATH):
            os.remove(SESSION_LOG_PATH)
    except Exception:
        pass


# =========================================================
# Discard Current Session
# =========================================================
def discard_current_session():
    """
    Clears the current session without archiving it.
    """
    global session_messages, current_token_count

    with session_lock:
        session_messages = []
        current_token_count = 0

    try:
        if os.path.exists(SESSION_LOG_PATH):
            os.remove(SESSION_LOG_PATH)
    except Exception:
        pass


# =========================================================
# Route Memory
# =========================================================
def set_last_route(route):
    """
    Stores the last routing decision for contextual continuity.
    """
    global last_route
    with session_lock:
        last_route = route


def get_last_route():
    """
    Returns the last routing decision.
    """
    with session_lock:
        return last_route


# =========================================================
# Helpers
# =========================================================
def get_recent_messages(limit=10):
    """
    Returns the most recent messages from the current session.
    """
    with session_lock:
        if not session_messages:
            return []
        return session_messages[-limit:]


def clear_session():
    discard_current_session()


# =========================================================
# Temporal Memory Resolution
# =========================================================
def get_last_archived_session():
    """
    Returns the last fully reconstructed archived session.
    """
    return conversation_memory.get_last_session()


# =========================================================
# Automatic Recovery Trigger
# =========================================================
recover_unsaved_session()
