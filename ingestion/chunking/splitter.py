from dataclasses import dataclass
from config import get_settings

settings = get_settings()


@dataclass
class Chunk:
    content: str
    index: int
    source_id: str
    source_type: str
    metadata: dict


def chunk_text(
    text: str,
    source_id: str,
    source_type: str,
    metadata: dict | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Simple overlap-aware text splitter.
    Splits on sentence boundaries where possible, falls back to hard cut.
    """
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap
    metadata = metadata or {}

    # Prefer splitting at paragraph / sentence boundaries
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    raw_chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= size:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                raw_chunks.append(current)
            # If para itself is too long, hard-split it
            if len(para) > size:
                for i in range(0, len(para), size - overlap):
                    raw_chunks.append(para[i : i + size])
            else:
                current = para

    if current:
        raw_chunks.append(current)

    # Add overlap: prepend tail of previous chunk to current
    overlapped: list[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and overlap > 0:
            tail = raw_chunks[i - 1][-overlap:]
            chunk = tail + " " + chunk
        overlapped.append(chunk.strip())

    return [
        Chunk(
            content=c,
            index=idx,
            source_id=source_id,
            source_type=source_type,
            metadata=metadata,
        )
        for idx, c in enumerate(overlapped)
        if c
    ]
