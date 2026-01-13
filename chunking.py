from __future__ import annotations

from typing import Iterable, List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_pages(
    page_texts: Iterable[tuple[str, dict]],
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> List[tuple[str, dict]]:
    """
    Chunk page texts into overlapping segments suitable for embeddings.

    Each output is (chunk_text, metadata), where metadata includes:
      - page_number
      - chunk_id (per page)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: List[tuple[str, dict]] = []

    for text, metadata in page_texts:
        if not text.strip():
            continue

        split_texts = splitter.split_text(text)
        for idx, chunk_text in enumerate(split_texts):
            chunk_metadata = dict(metadata)
            chunk_metadata["chunk_id"] = idx
            chunks.append((chunk_text, chunk_metadata))

    return chunks

