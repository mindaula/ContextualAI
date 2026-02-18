# ingest_books.py
# Semantic academic ingestion pipeline with repository support

import sys
import os
import re
import argparse
from datetime import datetime
import fitz  # PDF text extraction (PyMuPDF)
from ebooklib import epub
from bs4 import BeautifulSoup
import memory_system


# =========================
# Configuration
# =========================

SUPPORTED_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".cpp", ".c",
    ".go", ".rs", ".md", ".txt", ".json", ".yaml", ".yml"
}

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", "venv",
    "build", "dist", ".idea"
}


# =========================
# Metadata file loader
# =========================

def load_metadata_file():
    """
    Loads metadata definitions from metadata.txt.

    Expected format per line:
    name | key=value | key=value
    """
    metadata_map = {}

    if not os.path.exists("metadata.txt"):
        return metadata_map

    with open("metadata.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue

            parts = [p.strip() for p in line.split("|")]
            name = parts[0]

            data = {}
            for part in parts[1:]:
                if "=" in part:
                    k, v = part.split("=", 1)
                    data[k.strip()] = v.strip()

            metadata_map[name] = data

    return metadata_map


# =========================
# Text extraction
# =========================

def extract_pdf_text(path):
    """
    Extracts plain text from a PDF file.
    """
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text


def extract_epub_text(path):
    """
    Extracts text content from an EPUB file.
    """
    book = epub.read_epub(path)
    text = ""
    for item in book.get_items():
        if item.get_type() == 9:  # Document type
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text += soup.get_text() + "\n"
    return text


# =========================
# Cleaning
# =========================

def clean_text(text):
    """
    Normalizes whitespace in extracted text.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_code_text(text):
    """
    Removes extremely long or malformed code lines.
    """
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        if len(line) > 500:
            continue
        if re.search(r"[{};]{20,}", line):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


# =========================
# Chunking
# =========================

def semantic_chunk_text(text, max_words=300):
    """
    Splits text into semantically grouped chunks
    based on paragraph boundaries.
    """
    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        words = para.split()
        word_count = len(words)

        if word_count == 0:
            continue

        if current_length + word_count <= max_words:
            current_chunk.append(para)
            current_length += word_count
        else:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_length = word_count

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def semantic_chunk_code(text, max_lines=80):
    """
    Splits source code into fixed-size line blocks.
    """
    lines = text.split("\n")
    chunks = []

    for i in range(0, len(lines), max_lines):
        chunk = "\n".join(lines[i:i + max_lines])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


# =========================
# Metadata normalization
# =========================

def normalize_year(year):
    """
    Ensures year is a valid integer.
    """
    try:
        return int(year)
    except Exception:
        return datetime.now().year


# =========================
# Ingest single file (books)
# =========================

def ingest_file(filepath, source=None, source_type="book", year=None):

    filepath = os.path.abspath(filepath)

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"Processing file: {filepath}")

    if filepath.lower().endswith(".pdf"):
        raw_text = extract_pdf_text(filepath)
    elif filepath.lower().endswith(".epub"):
        raw_text = extract_epub_text(filepath)
    else:
        print("Unsupported file format.")
        return

    if not raw_text.strip():
        print("No text extracted.")
        return

    cleaned = clean_text(raw_text)
    chunks = semantic_chunk_text(cleaned)

    if not chunks:
        print("No chunks generated.")
        return

    file_year = year
    if not file_year:
        try:
            file_year = datetime.fromtimestamp(
                os.path.getmtime(filepath)
            ).year
        except Exception:
            file_year = datetime.now().year

    file_year = normalize_year(file_year)
    file_source = source if source else os.path.basename(filepath)

    metadata = {
        "source": file_source,
        "type": source_type,
        "year": file_year,
        "ingested_at": datetime.now().year
    }

    memory_system.add_academic_chunks(
        texts=chunks,
        metadata=metadata
    )

    print(f"{len(chunks)} chunks stored.")


# =========================
# Ingest directory (repository)
# =========================

def ingest_directory(dirpath, cli_type="repo", cli_source=None, cli_year=None, metadata_map=None):

    dirpath = os.path.abspath(dirpath)
    repo_name = os.path.basename(dirpath)

    print(f"Processing repository: {repo_name}")

    all_chunks = []

    for root, dirs, files in os.walk(dirpath):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in SUPPORTED_CODE_EXTENSIONS:
                continue

            filepath = os.path.join(root, file)

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
            except Exception:
                continue

            cleaned = clean_code_text(raw)
            chunks = semantic_chunk_code(cleaned)
            all_chunks.extend(chunks)

    if not all_chunks:
        print("No supported files found.")
        return

    meta_from_file = metadata_map.get(repo_name, {}) if metadata_map else {}

    source = cli_source or meta_from_file.get("source") or repo_name
    year = cli_year or meta_from_file.get("year")
    source_type = cli_type or meta_from_file.get("type", "repo")

    year = normalize_year(year)

    metadata = {
        "source": source,
        "type": source_type,
        "year": year,
        "ingested_at": datetime.now().year
    }

    memory_system.add_academic_chunks(
        texts=all_chunks,
        metadata=metadata
    )

    print(f"{len(all_chunks)} code chunks stored.")


# =========================
# Main entry point
# =========================

def main():

    parser = argparse.ArgumentParser(description="Semantic academic ingestion")

    parser.add_argument("filepath", help="File | Directory | *")
    parser.add_argument("--type", default="book", help="book | repo | notes")
    parser.add_argument("--source", default=None)
    parser.add_argument("--year", type=int, default=None)

    args = parser.parse_args()

    metadata_map = load_metadata_file()

    if args.filepath == "*":

        entries = os.listdir(os.getcwd())

        for entry in entries:
            if os.path.isdir(entry):
                ingest_directory(
                    entry,
                    cli_type=args.type,
                    cli_source=args.source,
                    cli_year=args.year,
                    metadata_map=metadata_map
                )
            elif entry.lower().endswith((".pdf", ".epub")):
                ingest_file(
                    entry,
                    source=args.source,
                    source_type=args.type,
                    year=args.year
                )

    else:

        if os.path.isdir(args.filepath):
            ingest_directory(
                args.filepath,
                cli_type=args.type,
                cli_source=args.source,
                cli_year=args.year,
                metadata_map=metadata_map
            )
        else:
            ingest_file(
                args.filepath,
                source=args.source,
                source_type=args.type,
                year=args.year
            )

    print("Ingestion completed.")


if __name__ == "__main__":
    main()
