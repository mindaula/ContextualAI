"""Short-term session memory manager with controlled long-term handoff.

Purpose of this abstraction:
    Maintain the active chat session as a thread-safe in-memory buffer and mirror it
    to disk (`current_session.json`) so an interrupted process can recover unfinished
    messages on next startup.

Short-term vs long-term memory:
    - Short-term memory is process-local state (`session_messages`,
      `current_token_count`, `last_route`) used for current-turn continuity.
    - Long-term memory is owned by `app.memory.conversation_memory`, where archived
      sessions are embedded and indexed for semantic retrieval.

Interaction with `conversation_memory`:
    - Recovery and archive paths forward message batches to
      `conversation_memory.add_session_to_memory`.
    - Archive additionally requests summary indexing through
      `conversation_memory.add_session_summary_to_memory`.
    - Read-only inspection of last archived session delegates to
      `conversation_memory.get_last_session`.

Memory index separation (personal vs conversation):
    This module never reads or writes personal memory artifacts (`personal.index`,
    `personal_meta.json`). Its persistence boundary is conversation memory only:
    `conversation_index.faiss`, `conversation_meta.json`, optional summary index/meta,
    and session backup/log files.

External dependencies:
    - Standard library: `os`, `json`, `threading`, `datetime`, `logging`.
    - Internal dependency: `app.memory.conversation_memory` for long-term conversation
      indexing and retrieval.
"""

import os
import json
import threading
import logging
from datetime import datetime

import app.memory.conversation_memory as conversation_memory


logger = logging.getLogger(__name__)


MAX_TOKENS = 6000
BACKUP_DIR = "conversation_backups"
SESSION_LOG_PATH = "current_session.json"


session_messages = []
current_token_count = 0
last_route = None
session_lock = threading.Lock()


def estimate_tokens(text):
    """Estimate token usage for short-term archive-threshold accounting.

    Input:
        text: Arbitrary message content to estimate.

    Output:
        Integer token estimate using an approximate `4 chars ~= 1 token` heuristic.
        Returns `0` for empty input and at least `1` for non-empty input.

    Routing logic:
        No route selection is performed. The return value is consumed by
        `add_message` to decide whether the short-term session should be archived.

    Side effects:
        None.
    """
    if not text:
        return 0
    return max(1, len(str(text)) // 4)


def recover_unsaved_session():
    """Recover unfinished short-term session data from disk into conversation memory.

    Input:
        None.

    Output:
        None.

    Routing logic:
        No routing decision is changed. Recovery only restores persisted context so
        future conversation-retrieval routes can access the recovered messages.

    Side effects:
        - Reads `current_session.json` when it exists.
        - Writes `conversation_backups/recovered_*.json` snapshot files.
        - Persists recovered messages via `conversation_memory.add_session_to_memory`.
        - Deletes `current_session.json` after successful recovery handling.

    Failure handling:
        All exceptions are suppressed; recovery is best-effort and non-blocking.
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


def add_message(role, content):
    """Append a message to active short-term state and sync it to session log.

    Input:
        role: Message role label (`user`, `assistant`, etc.).
        content: Message payload text.

    Output:
        None.

    Routing logic:
        Does not assign routes. Route continuity is stored independently via
        `set_last_route`, while this function only handles message persistence.

    Side effects:
        - Appends to in-memory `session_messages`.
        - Increments `current_token_count`.
        - Rewrites `current_session.json` to mirror the in-memory buffer.
        - Triggers `archive_session` when token usage reaches `MAX_TOKENS`.

    Failure handling:
        Session log write errors are suppressed; in-memory state remains updated.
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


def archive_session():
    """Finalize active short-term session and persist it to long-term conversation memory.

    Input:
        None.

    Output:
        None.

    Routing logic:
        No route selection occurs here. This function only commits message history so
        downstream conversation retrieval can access archived sessions semantically.

    Side effects:
        - Writes `conversation_backups/session_*.json` archive snapshots.
        - Persists raw conversation entries via `conversation_memory.add_session_to_memory`.
        - Persists session summary entries via `conversation_memory.add_session_summary_to_memory`.
        - Resets `session_messages` and `current_token_count`.
        - Removes `current_session.json` after a successful archive.

    Memory index boundary:
        Only conversation memory indexes are affected. Personal memory indexes are not
        read or modified by this function.

    Failure handling:
        Backup or long-term persistence failures are logged and re-raised.
    """
    global session_messages, current_token_count

    with session_lock:
        if not session_messages:
            return

        messages_copy = list(session_messages)
        total_tokens = current_token_count

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
            logger.exception("Failed to write session backup during archive")
            raise

        try:
            if messages_copy:
                session_hash = conversation_memory.add_session_to_memory(messages_copy)
                conversation_memory.add_session_summary_to_memory(messages_copy, session_hash)
        except Exception:
            logger.exception("Failed to persist session to long-term memory during archive")
            raise

        session_messages = []
        current_token_count = 0

        try:
            if os.path.exists(SESSION_LOG_PATH):
                os.remove(SESSION_LOG_PATH)
        except Exception:
            logger.exception("Failed to delete %s after successful archive", SESSION_LOG_PATH)
            raise


def discard_current_session():
    """Clear active short-term messages without writing to long-term memory.

    Input:
        None.

    Output:
        None.

    Routing logic:
        Removes current message context only. `last_route` is intentionally left
        unchanged and must be updated separately if needed.

    Side effects:
        - Resets in-memory `session_messages`.
        - Resets `current_token_count`.
        - Removes `current_session.json` if present.

    Failure handling:
        Session log deletion failures are suppressed.
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


def set_last_route(route):
    """Store the most recent route label for follow-up-aware request handling.

    Input:
        route: Route identifier produced by orchestration (for example `academic`).

    Output:
        None.

    Routing logic:
        Persists route continuity metadata used by query-rewrite and intent logic in
        later turns. No route computation is performed here.

    Side effects:
        Updates global `last_route` under lock.
    """
    global last_route
    with session_lock:
        last_route = route


def get_last_route():
    """Return the latest stored route label from short-term session state.

    Input:
        None.

    Output:
        Route string or `None` if no route has been recorded.

    Routing logic:
        Read-only support for route-aware rewrite/routing components.

    Side effects:
        None.
    """
    with session_lock:
        return last_route


def get_recent_messages(limit=10):
    """Return the most recent short-term messages from the active session.

    Input:
        limit: Maximum number of trailing messages to return.

    Output:
        List of message dictionaries in original chronological order.

    Routing logic:
        Supplies immediate context for follow-up rewriting and route selection.

    Side effects:
        None.
    """
    with session_lock:
        if not session_messages:
            return []
        return session_messages[-limit:]


def clear_session():
    """Compatibility wrapper that delegates to `discard_current_session`.

    Input:
        None.

    Output:
        None.

    Routing logic:
        Matches `discard_current_session`; no route decisions are made here.

    Side effects:
        Inherits all side effects from `discard_current_session`.
    """
    discard_current_session()


def get_last_assistant_message():
    """Fetch the latest assistant response from current short-term message history.

    Input:
        None.

    Output:
        Assistant message content string or `None` when unavailable.

    Routing logic:
        Used by transform-style follow-up flows that operate on the prior assistant
        answer.

    Side effects:
        None.
    """
    with session_lock:
        for msg in reversed(session_messages):
            if msg.get("role") == "assistant":
                return msg.get("content")
    return None


def get_last_archived_session():
    """Return the most recent archived conversation session from long-term memory.

    Input:
        None.

    Output:
        The list payload returned by `conversation_memory.get_last_session`.

    Routing logic:
        Read-only helper for components that need archived-session inspection.

    Side effects:
        Reads long-term conversation-memory state through `conversation_memory`.
        Personal memory indexes are not accessed.
    """
    return conversation_memory.get_last_session()


# Recover unfinished short-term session log on module import (best-effort).
recover_unsaved_session()
