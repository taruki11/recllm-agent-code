from __future__ import annotations

from .schema import Chunk, Document


def chunk_document(
    document: Document,
    chunk_size: int = 700,
    overlap: int = 100,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = document.text.strip()
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_id = f"{document.doc_id}_chunk_{index:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    title=document.title,
                    source_path=document.source_path,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    metadata={
                        **document.metadata,
                        "doc_type": document.doc_type,
                        "chunk_index": index,
                    },
                )
            )
            index += 1
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 700,
    overlap: int = 100,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return chunks
