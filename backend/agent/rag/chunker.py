"""
Text sanitization and sliding-window semantic chunking utilities.
Optimized for mathematical text, paper transcripts, and project specifications.
"""

import re


def clean_text(text: str) -> str:
    """Sanitize raw document text, removing page headers, footers, and redundant line breaks."""
    text = re.sub(r"--- PAGE BREAK ---", " ", text)
    text = re.sub(r"Page \d+ of \d+", " ", text)
    text = re.sub(r"K J Subramanyam: Preprint submitted to Elsevier", " ", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 150) -> list[str]:
    """
    Splits continuous text into overlapping semantic windows.
    Maintains sentence and paragraph boundaries when possible.
    """
    words = text.split()
    if not words:
        return []

    word_chunk_size = max(50, chunk_size // 5)
    word_overlap = max(10, overlap // 5)
    chunks = []
    i = 0

    while i < len(words):
        chunk_words = words[i : i + word_chunk_size]
        chunk = " ".join(chunk_words).strip()
        if len(chunk) > 80:
            chunks.append(chunk)
        i += word_chunk_size - word_overlap
        if i >= len(words) - word_overlap:
            break

    return chunks
