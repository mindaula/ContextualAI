"""
Multimodal file-input preprocessing utilities for API adapters.

Architectural role:
- Convert file references into extracted text that can be appended to a prompt.
- Enforce path/size/extension constraints before extraction.
- Provide adapter-level preprocessing only (no endpoint registration).

Processing lifecycle:
1. Resolve each input reference (`data:`, local path, or `file://` URL).
2. Pre-validate base64 payload size before decode/write.
3. Validate normalized path, base-directory scope, size, and extension.
4. Extract text by file type (OCR/text parsers).
5. Return user message augmented with extracted content and processing metadata.

Hard trigger handling:
- No route-style trigger parsing is implemented in this module.
- This module is content preprocessing only.

Interaction with core:
- No direct core invocation occurs in this module.
- Caller is expected to pass `augmented_text` downstream.

Error handling strategy:
- Per-file failures are captured and embedded as a generic textual marker.
- Processing continues for remaining files; function returns best-effort output.

Response formatting:
- Returns a dictionary with:
  - `augmented_text` containing original user text plus extracted context blocks.
  - `metadata` containing `files_processed`.

Side effects:
- Creates the allowed upload directory at import time.
- Writes base64 payloads to temporary files under the allowed directory.
- Removes temporary base64-decoded files in `finally`, including error paths.

Determinism considerations:
- OCR output and parser behavior may vary across dependency/runtime versions.
- Temporary filenames and ordering follow input sequence plus OS temp behavior.
"""

import os
import base64
import tempfile
from typing import List, Dict, Optional
from urllib.parse import urlparse, unquote

from PIL import Image
import pytesseract
import pdfplumber
import pandas as pd
import docx


# ============================================================
# CONFIG
# ============================================================

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp",
    ".pdf", ".txt", ".csv", ".docx"
}
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
ALLOWED_FILE_BASE_DIR = os.path.realpath(
    os.getenv("FILE_INPUT_BASE_DIR", os.path.join(PROJECT_ROOT, "uploads"))
)
os.makedirs(ALLOWED_FILE_BASE_DIR, exist_ok=True)


# ============================================================
# PUBLIC ENTRYPOINT
# ============================================================

def handle_files(
    file_inputs: List[str],
    user_message: str
) -> Dict:
    """
    Build prompt augmentation text from a list of file references.

    Adapter-facing responsibilities:
    - Preserve original user message when no valid/extractable files are found.
    - Attach extracted file content between explicit boundary markers.
    - Report `files_processed` count in metadata.
    """

    extracted_chunks = []
    metadata = {"files_processed": 0}

    for file_ref in file_inputs:
        temp_file_path = None
        try:
            file_path = _resolve_input(file_ref)

            if file_ref.startswith("data:"):
                temp_file_path = file_path

            if not file_path:
                continue

            _validate_file(file_path)

            text = _extract_content(file_path)

            if text:
                extracted_chunks.append(text)
                metadata["files_processed"] += 1

        except Exception as e:
            # Best-effort strategy: append a sanitized marker and continue.
            extracted_chunks.append("[File processing error]")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass

    if not extracted_chunks:
        return {
            "augmented_text": user_message,
            "metadata": metadata
        }

    combined_context = "\n\n--- FILE CONTENT START ---\n\n"
    combined_context += "\n\n".join(extracted_chunks)
    combined_context += "\n\n--- FILE CONTENT END ---\n\n"

    augmented = f"{user_message}\n\n{combined_context}"

    return {
        "augmented_text": augmented,
        "metadata": metadata
    }


# ============================================================
# INPUT RESOLUTION
# ============================================================

def _resolve_input(file_ref: str) -> Optional[str]:
    """
    Resolve an input reference to a local filesystem path.

    Supported formats:
    - data URL (decoded to temp file)
    - file URL (local host only)
    - plain local path

    Validation behavior:
    - Path existence checks are performed for file URLs and local paths.
    - Remote file URL hosts are rejected.
    """

    # Base64 image/file
    if file_ref.startswith("data:"):
        return _save_base64_to_temp(file_ref)

    # file:// URL
    if file_ref.startswith("file://"):
        parsed = urlparse(file_ref)

        # Reject remote hosts in file URLs.
        if parsed.netloc not in ("", "localhost"):
            return None

        candidate = unquote(parsed.path or "")
        normalized = _normalize_path(candidate)
        if normalized and os.path.exists(normalized):
            return normalized
        return None

    # local path
    normalized = _normalize_path(file_ref)
    if normalized and os.path.exists(normalized):
        return normalized

    return None


def _save_base64_to_temp(data_url: str) -> str:
    """
    Decode a data URL into a temporary file under the allowed base directory.

    Input validation behavior:
    - Applies an approximate decoded-size check before decode/write.
    - Raises `ValueError` when payload exceeds configured maximum size.
    """
    header, encoded = data_url.split(",", 1)
    padding = 0
    if encoded.endswith("=="):
        padding = 2
    elif encoded.endswith("="):
        padding = 1
    approx_decoded_size = (len(encoded) * 3) // 4 - padding
    if approx_decoded_size > MAX_FILE_SIZE_BYTES:
        raise ValueError("File exceeds max size limit")

    ext = ".tmp"
    if "image/png" in header:
        ext = ".png"
    elif "image/jpeg" in header:
        ext = ".jpg"
    elif "image/webp" in header:
        ext = ".webp"
    elif "application/pdf" in header:
        ext = ".pdf"

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=ext,
        dir=ALLOWED_FILE_BASE_DIR,
    )
    temp_file.write(base64.b64decode(encoded))
    temp_file.close()

    return temp_file.name


# ============================================================
# VALIDATION
# ============================================================

def _normalize_path(path: str) -> Optional[str]:
    """Expand and canonicalize a path; return `None` for empty input."""
    if not path:
        return None
    expanded = os.path.expanduser(path)
    return os.path.realpath(expanded)


def _is_allowed_path(path: str) -> bool:
    """Return whether `path` is inside `ALLOWED_FILE_BASE_DIR` after normalization."""
    try:
        normalized = os.path.realpath(path)
        return os.path.commonpath([normalized, ALLOWED_FILE_BASE_DIR]) == ALLOWED_FILE_BASE_DIR
    except Exception:
        return False

def _validate_file(path: str):
    """
    Enforce access and file-type constraints before extraction.

    Validation behavior:
    - Rejects empty/invalid paths.
    - Rejects paths outside allowed base directory.
    - Rejects non-existent files.
    - Rejects files larger than configured max size.
    - Rejects unsupported extensions.
    """
    normalized = _normalize_path(path)
    if not normalized:
        raise ValueError("Invalid file path")

    if not _is_allowed_path(normalized):
        raise ValueError("Access denied: path is outside allowed directory")

    if not os.path.exists(normalized):
        raise ValueError("File does not exist")

    size_mb = os.path.getsize(normalized) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError("File exceeds max size limit")

    _, ext = os.path.splitext(normalized)

    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type")


# ============================================================
# EXTRACTION ROUTER
# ============================================================

def _extract_content(path: str) -> str:
    """Dispatch extraction based on normalized file extension."""

    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        return _extract_image(path)

    if ext == ".pdf":
        return _extract_pdf(path)

    if ext == ".txt":
        return _extract_txt(path)

    if ext == ".csv":
        return _extract_csv(path)

    if ext == ".docx":
        return _extract_docx(path)

    return ""


# ============================================================
# IMAGE PROCESSING (OCR + Metadata)
# ============================================================

def _extract_image(path: str) -> str:
    """Return image metadata plus OCR text (best-effort OCR)."""
    img = Image.open(path)

    ocr_text = ""
    try:
        ocr_text = pytesseract.image_to_string(img)
    except Exception:
        pass

    return (
        f"[Image: {os.path.basename(path)}]\n"
        f"Resolution: {img.size}\n"
        f"OCR Extracted Text:\n{ocr_text}"
    )


# ============================================================
# PDF
# ============================================================

def _extract_pdf(path: str) -> str:
    """Extract text from each PDF page and concatenate with newlines."""
    text = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")

    return "\n".join(text)


# ============================================================
# TEXT
# ============================================================

def _extract_txt(path: str) -> str:
    """Read UTF-8 text with decoding errors ignored."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ============================================================
# CSV
# ============================================================

def _extract_csv(path: str) -> str:
    """Load CSV into a dataframe and serialize the first 20 rows."""
    df = pd.read_csv(path)
    return df.head(20).to_string()


# ============================================================
# DOCX
# ============================================================

def _extract_docx(path: str) -> str:
    """Extract paragraph text from a DOCX document."""
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)
