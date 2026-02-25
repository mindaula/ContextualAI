"""Academic ingestion CLI for retrieval memory population.

Architectural role:
    Converts local files or repository trees into normalized text chunks and pushes
    them into domain-scoped academic memory through `app.memory.memory_system`.

Pipeline summary:
    1. Resolve file/batch inputs and metadata source.
    2. Extract raw text from supported formats.
    3. Clean and chunk content with deterministic heuristics.
    4. Forward chunks + metadata to `memory_system.add_academic_chunks(...)`.

Retrieval/ranking relation:
    This module does not perform retrieval ranking or scoring. It only controls what
    data becomes searchable later.

FAISS interaction:
    Indirect only. Index writes are handled by memory-layer APIs.

Determinism and performance:
    Cleaning/chunking functions are deterministic for fixed input text. Batch runtime
    scales with file count, parser cost (PDF/EPUB), and chunk volume.
"""

import sys
import os
import re
import glob
import argparse
from datetime import datetime
import fitz
from ebooklib import epub
from bs4 import BeautifulSoup
import app.memory.memory_system as memory_system


SUPPORTED_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".cpp", ".c",
    ".go", ".rs", ".md", ".txt", ".json", ".yaml", ".yml"
}

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", "venv",
    "build", "dist", ".idea"
}


def load_metadata_file_from_path(path):
    """Parse a `metadata.txt` mapping file used by batch ingestion.

    Args:
        path: Absolute or relative path to metadata mapping file.

    Returns:
        Dict keyed by filename. Values are metadata dictionaries parsed from
        `key=value` segments separated by `|`.

    Edge cases:
        - Blank lines and malformed lines (missing `|`) are skipped.
        - Duplicate filenames overwrite earlier entries.
    """
    metadata_map = {}

    with open(path, "r", encoding="utf-8") as f:
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


def extract_pdf_text(path):
    """Extract plain text from all pages of a PDF.

    Args:
        path: Path to a PDF file.

    Returns:
        Concatenated page text with newline separators.

    Performance:
        Linear in page count and parser complexity.
    """
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text


def extract_epub_text(path):
    """Extract plain text from EPUB document items.

    Args:
        path: Path to an EPUB file.

    Returns:
        Concatenated extracted text from HTML-like content items.

    Edge cases:
        Only items with type id `9` are processed.
    """
    book = epub.read_epub(path)
    text = ""
    for item in book.get_items():
        if item.get_type() == 9:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text += soup.get_text() + "\n"
    return text


def clean_text(text):
    """Normalize whitespace in natural-language text blocks.

    Args:
        text: Raw extracted text.

    Returns:
        Whitespace-normalized text with repeated blank lines collapsed.

    Determinism:
        Deterministic regular-expression transformation.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_code_text(text):
    """Drop noisy code lines before chunking repository files.

    Args:
        text: Raw source-code text.

    Returns:
        Filtered code string excluding extreme outlier lines.

    Heuristics:
        - Drops lines longer than 500 characters.
        - Drops lines with repeated punctuation patterns likely to be minified noise.

    Determinism:
        Deterministic for fixed input.
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


def semantic_chunk_text(text, max_words=300):
    """Chunk prose text into paragraph groups bounded by word count.

    Args:
        text: Cleaned prose text.
        max_words: Soft maximum words per chunk.

    Returns:
        List of chunk strings preserving paragraph order.

    Chunking strategy:
        Paragraph-first accumulation; when adding a paragraph would exceed the cap,
        a new chunk starts.

    Determinism and performance:
        Deterministic. Runtime is linear in paragraph count.
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
    """Chunk code text into fixed-size line windows.

    Args:
        text: Cleaned source-code text.
        max_lines: Maximum line count per chunk.

    Returns:
        List of non-empty code chunks.

    Determinism and performance:
        Deterministic fixed-step slicing with linear runtime in line count.
    """
    lines = text.split("\n")
    chunks = []

    for i in range(0, len(lines), max_lines):
        chunk = "\n".join(lines[i:i + max_lines])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def normalize_year(year):
    """Normalize year metadata to integer with fallback to current year.

    Args:
        year: User-supplied or metadata-supplied year value.

    Returns:
        Integer year.

    Edge cases:
        Parsing failures fall back to `datetime.now().year`.
    """
    try:
        return int(year)
    except Exception:
        return datetime.now().year


def ingest_file(filepath, source=None, source_type="book", year=None, domain=None):
    """Ingest a single PDF/EPUB into academic memory.

    Args:
        filepath: Path to input document.
        source: Source label persisted in metadata (defaults to filename).
        source_type: Metadata type value (for example `book`, `notes`, `repo`).
        year: Publication/reference year.
        domain: Target academic domain folder.

    Returns:
        None.

    Side effects:
        - Reads file from disk.
        - Writes academic memory artifacts indirectly via `memory_system`.
        - Emits progress/status messages to stdout.

    Edge cases:
        - Missing file, unsupported extension, empty extraction, or empty chunk set
          produce early-return status messages.
    """

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

    file_year = normalize_year(year)
    file_source = source if source else os.path.basename(filepath)

    metadata = {
        "source": file_source,
        "type": source_type,
        "year": file_year,
        "ingested_at": datetime.now().year
    }

    memory_system.add_academic_chunks(
        texts=chunks,
        metadata=metadata,
        domain=domain
    )

    print(f"{len(chunks)} chunks stored.")


def ingest_directory(dirpath, cli_type="repo", cli_source=None, cli_year=None, metadata_map=None, domain=None):
    """Ingest supported text/code files from a directory tree.

    Args:
        dirpath: Root directory to scan.
        cli_type: Metadata `type` value to apply to all chunks.
        cli_source: Optional source label override.
        cli_year: Optional year override.
        metadata_map: Unused compatibility parameter.
        domain: Target academic domain folder.

    Returns:
        None.

    Side effects:
        - Recursively scans filesystem.
        - Reads supported files.
        - Writes academic memory artifacts indirectly via `memory_system`.

    Performance considerations:
        Total runtime and memory usage scale with file count and total extracted text.
        All chunks are accumulated in memory before persistence.
    """

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

    year = normalize_year(cli_year)

    metadata = {
        "source": cli_source or repo_name,
        "type": cli_type,
        "year": year,
        "ingested_at": datetime.now().year
    }

    memory_system.add_academic_chunks(
        texts=all_chunks,
        metadata=metadata,
        domain=domain
    )

    print(f"{len(all_chunks)} code chunks stored.")


def main():
    """CLI entrypoint for single-file or batch academic ingestion.

    Control flow:
        - Validates argument shape.
        - Distinguishes single-file mode vs batch mode (directory or glob).
        - In batch mode, requires a `metadata.txt` mapping in the target directory.
        - Dispatches ingestion per file and reports progress.

    Error handling:
        - Uses `argparse` validation errors for invalid option combinations.
        - Uses `sys.exit(1)` when required batch metadata mapping is missing.
    """

    parser = argparse.ArgumentParser(description="Semantic academic ingestion")
    parser.add_argument("filepath", nargs="+", help="File(s) or directory")
    parser.add_argument("--type", default=None, help="book | repo | notes (required in single-file mode)")
    parser.add_argument("--source", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--domain", required=True, help="Target knowledge domain")

    args = parser.parse_args()

    if len(args.filepath) != 1:
        parser.error(
            "Pass exactly one path. Multiple explicit file paths are not allowed. "
            "Use a single wildcard pattern or a directory for batch mode."
        )

    path_spec = args.filepath[0]
    is_batch_mode = os.path.isdir(path_spec) or any(ch in path_spec for ch in ["*", "?", "["])

    domain_path = os.path.join("knowledge", args.domain)
    if not os.path.exists(domain_path):
        print(f"Creating new domain directory: {domain_path}")
        os.makedirs(domain_path)

    if is_batch_mode:
        if args.type is not None or args.source is not None or args.year is not None:
            parser.error("Do not use --type/--source/--year in batch mode. Use metadata.txt instead.")

        if os.path.isdir(path_spec):
            base_dir = os.path.abspath(path_spec)
            target_paths = [
                os.path.join(base_dir, name)
                for name in os.listdir(base_dir)
                if os.path.isfile(os.path.join(base_dir, name))
            ]
        else:
            base_dir = os.path.dirname(os.path.abspath(path_spec))
            target_paths = [p for p in sorted(glob.glob(path_spec)) if os.path.isfile(p)]

        metadata_path = os.path.join(base_dir, "metadata.txt")
        if not os.path.exists(metadata_path):
            print("metadata.txt required in target directory for batch ingestion.")
            sys.exit(1)

        metadata_map = load_metadata_file_from_path(metadata_path)

        for path in target_paths:
            filename = os.path.basename(path)
            meta = metadata_map.get(filename)

            if not meta:
                print(f"No metadata entry found for {filename}")
                continue

            ingest_file(
                path,
                source=meta.get("source"),
                source_type=meta.get("type", "book"),
                year=meta.get("year"),
                domain=args.domain
            )
    else:
        if args.type is None:
            parser.error("--type is required in single-file mode.")

        ingest_file(
            path_spec,
            source=args.source,
            source_type=args.type,
            year=args.year,
            domain=args.domain
        )

    print("Ingestion completed.")


if __name__ == "__main__":
    main()
