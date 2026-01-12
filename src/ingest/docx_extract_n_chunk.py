# src/pipelines/data_extraction/simple_data_extraction.py
"""
Extract plain text from DOCX files, grouping content under the lowest
heading within each branch of the heading tree.  For each chunk we
record the full hierarchy of parent headings **and their levels** so
that downstream components can trace the origin of the text.
The output format for each chunk is now:
  {
      "doc_title": "<file‑name>",
      "chunk": "<text‑chunk>",
      "heading_path": [
          {"title": "Heading 1 title", "level": 1},
          {"title": "Heading 2 title", "level": 2},
          ...
      ]
  }

Programmatic usage:
  from pipelines.run_extraction_n_import import get_all_chunks
  chunks = get_all_chunks("/path/to/docx_folder")
"""

import sys
from pathlib import Path
import typer

from src.utils.general_util import write_jsonl

try:
    from docx import Document
    from docx.document import Document as DocumentObject
except ImportError:
    sys.exit("ERROR: python-docx not installed. Run 'pip install python-docx'.")


def get_file_paths(dir_path: str | Path, extension: str) -> list[Path]:
    """Return a list of all *extension* files under *dir_path* (recursively)."""
    directory = Path(dir_path)
    json_paths = list(directory.glob(extension))
    return json_paths

def _heading_level(style_name: str | None) -> int | None:
    """
    Convert a style name like 'Heading 1' into an integer level.
    Return None if the style is not a heading.
    """
    if style_name and style_name.startswith("Heading"):
        try:
            return int(style_name.split(" ")[1])
        except (IndexError, ValueError):
            return None
    return None


def extract_chunks_from_doc(doc: DocumentObject, title: str, max_chunk_len: int = 2000) -> list[dict[str, str]]:
    """
    Extract text chunks from a single Document object.
    Each chunk is a flat dict:
    {
        "doc_title":  <file‑name>,
        "heading_2":  <level‑2 heading or "na">,
        "heading_3":  <level‑3 heading or "na">,
        "heading_4":  <level‑4 heading or "na">,
        "chunk":      <text‑chunk>
    }
    """
    chunks: list[dict[str, str]] = []

    # Keeps the most recent heading for each level that we have seen.
    heading_stack: dict[int, str] = {}
    current_text: list[str] = []

    def flush_chunk() -> None:
        """Write the current paragraph buffer as a chunk (if non‑empty)."""
        nonlocal current_text
        if not current_text:
            return
        chunk_text = "\n".join(current_text).strip()
        if not chunk_text:
            return

        # Build the flat heading columns.  We only care about levels 2‑4.
        heading_2 = heading_stack.get(2, "na")
        heading_3 = heading_stack.get(3, "na")
        heading_4 = heading_stack.get(4, "na")

        chunks.append({
            "doc_title":  title,
            "heading_2":  heading_2,
            "heading_3":  heading_3,
            "heading_4":  heading_4,
            "chunk":      chunk_text,
        })
        current_text = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        level = _heading_level(para.style.name)
        if level is not None:
            # A heading signals the end of the previous chunk.
            flush_chunk()

            # Update the heading stack – keep all levels < current,
            # replace/add the current level, and drop deeper levels.
            heading_stack[level] = text
            for lvl in list(heading_stack.keys()):
                if lvl > level:
                    del heading_stack[lvl]

            # The heading itself starts a new chunk context.
            current_text.append(text)
            continue

        # Normal paragraph – add to the current chunk.
        current_text.append(text)

        # Optional: enforce a maximum chunk size by characters.
        # (uncomment if you need early flushing of large blocks)
        # if sum(len(p) for p in current_text) >= max_chunk_len:
        #     flush_chunk()

    # Flush whatever remains after processing all paragraphs.
    flush_chunk()
    return chunks

def get_all_chunks(input_dir: str | Path, max_chunk_len: int = 2000) -> list[dict[str, str|int]]:
    """
    Walk ``input_dir`` for ``*.docx`` files and return a single list
    containing all extracted chunks from every document.
    """
    docx_paths: list[Path] = get_file_paths(input_dir, "*.docx")
    all_chunks: list[dict[str, str]] = []

    for docx_path in docx_paths:
        print(f"Processing '{docx_path}' …")
        doc = Document(str(docx_path))
        title = Path(docx_path).stem
        all_chunks.extend(extract_chunks_from_doc(doc, title, max_chunk_len))

    chunk_ids: list[dict[str, int]] = [{"chunk_id": i} for i in range(len(all_chunks))]
    chunks_with_ids: list[dict[str, str | int]] = [{**cid, **chunk} for cid, chunk in zip(chunk_ids, all_chunks)]

    return chunks_with_ids


def main(
    input_dir: Path = typer.Argument("data/input", help="Root directory containing DOCX files"),
    output_path: Path = typer.Argument("RAG_pipeline/data/output/chunks.jsonl", help="Destination JSONL file"),
) -> None:
    """
    Command‑line entry point – uses typer for argument parsing.
    """
    chunks = get_all_chunks(input_dir)

    typer.echo(f"Found {len(chunks)} chunk(s). Writing to '{output_path}' …")
    write_jsonl(chunks, output_path)
    typer.echo("Done!")

if __name__ == "__main__":
    typer.run(main)