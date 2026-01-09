from test_pipeline.utils.general_util import read_text, save_text

# Sample text to chunk
sample_text = read_text(r"test_pipeline/demo_docx/sample1.md")

def chunk(text: str, max_chars: int = 500, overlap: int = 100) -> list[str]:
    """
    Split a large string into overlapping chunks suitable for RAG retrieval.

    Parameters
    ----------
    text : str
        The raw document text to split.
    max_chars : int, default 500
        The maximum number of characters per chunk.
    overlap : int, default 100
        The number of characters that successive chunks will share.

    Returns
    -------
    List[str]
        A list of chunk strings, each of length <= max_chars. The
        overlap between consecutive chunks is approximately ``overlap`` characters.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must be non‑negative")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    step = max_chars - overlap
    chunks = []

    # Ensure we process the entire string
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk_text = text[start:end]
        chunks.append(chunk_text)
        # Advance by step, not by max_chars to create overlap
        start += step

    return chunks

if __name__ == "__main__":
    chunks = chunk(text=sample_text, max_chars=500, overlap=100)
    save_text(str(chunks), r"test_pipeline/demo_docx/sample1_chunked.md")
