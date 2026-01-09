# src/pipelines/data_extraction/simple_data_extraction.py
# -------------------------------------------------------
# Extract plain text from DOCX files, ignoring images, tables, charts, etc.
# Returns a list of dictionaries: {"doc_title": "<file‑name>", "chunk": "<text‑chunk>"}
#
# CLI usage:
#   python src/pipelines/data_extraction/simple_data_extraction.py <input‑dir> <output‑jsonl>
#
# Programmatic usage:
#   from src.pipelines.data_extraction.simple_data_extraction import get_all_chunks
#   chunks = get_all_chunks("/path/to/docx_folder")
#
# Dependencies:
#   pip install python-docx
#
# Author: <Your Name>
# -------------------------------------------------------

import json
import sys
from pathlib import Path
from typing import List, Dict
import typer

try:
    from docx import Document
except ImportError:
    sys.exit("ERROR: python-docx not installed. Run 'pip install python-docx'.")

def get_file_paths(dir_path: str | Path, extension: str) ->  list[Path]:
    """Return a list of all *extension* files under *dir_path* (recursively)."""
    directory = Path(dir_path)
    json_paths = list(directory.glob(extension))
    return json_paths

def extract_chunks_from_doc(doc: Document, title: str, max_chunk_len: int = 2000) -> List[Dict[str, str]]:
    """
    Extract text chunks from a single ``Document`` object.

    Parameters
    ----------
    doc : Document
        The python‑docx Document instance.
    title : str
        Title (stem) of the file.
    max_chunk_len : int, optional
        If a chunk grows beyond this many characters, it is flushed
        early to keep chunks reasonably sized.

    Returns
    -------
    List[Dict[str, str]]
        One dictionary per chunk.
    """
    chunks: List[Dict[str, str]] = []
    current: List[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            # Skip empty paragraphs
            continue

        # Treat headings as new chunk boundaries
        if para.style.name.startswith("Heading"):
            # Flush the current chunk first
            if current:
                chunk_text = "\n".join(current).strip()
                if chunk_text:
                    chunks.append({"doc_title": title, "chunk": chunk_text})
                current = []

            # Start a new chunk with the heading itself
            current.append(text)
            continue

        # Normal paragraph – add to the current chunk
        current.append(text)

        # # Optional: enforce a maximum chunk size by characters
        # if sum(len(p) for p in current) >= max_chunk_len:
        #     chunk_text = "\n".join(current).strip()
        #     if chunk_text:
        #         chunks.append({"doc_title": title, "chunk": chunk_text})
        #     current = []

    # Flush any remaining content after the loop
    if current:
        chunk_text = "\n".join(current).strip()
        if chunk_text:
            chunks.append({"doc_title": title, "chunk": chunk_text})

    return chunks


def get_all_chunks(input_dir: str | Path, max_chunk_len: int = 2000) -> List[Dict[str, str]]:
    """
    Walk ``input_dir`` for ``*.docx`` files and return a single list
    containing all extracted chunks from every document.

    Parameters
    ----------
    input_dir : str | Path
        Directory containing DOCX files.
    max_chunk_len : int, optional
        Chunk size limit used by ``extract_chunks_from_doc``.

    Returns
    -------
    List[Dict[str, str]]
        Combined list of chunks from all documents.
    """
    docx_paths: List[Path] = get_file_paths(input_dir, "*.docx")
    all_chunks: List[Dict[str, str]] = []

    for docx_path in docx_paths:
        print(f"Processing '{docx_path}' …")
        doc = Document(str(docx_path))
        title = Path(docx_path).stem
        all_chunks.extend(extract_chunks_from_doc(doc, title, max_chunk_len))

    return all_chunks


def write_jsonl(chunks: List[Dict[str, str]], output_path: str) -> None:
    """
    Persist a list of chunk dictionaries as JSON Lines.

    Parameters
    ----------
    chunks : List[Dict[str, str]]
        The data to write.
    output_path : str
        Destination file path.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump(chunk, f, ensure_ascii=False)
            f.write("\n")


def main(
    input_dir: Path = typer.Argument("data/input", help="Root directory containing DOCX files"),
    output_path: Path = typer.Argument("/home/krasimir.angelov@nccdi.local/AI/AI/Code/RAG_pipeline/data/output/chunks.jsonl", help="Destination JSONL file"),
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
