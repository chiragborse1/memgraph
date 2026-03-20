import pytest
from ingestion.chunking.splitter import chunk_text


def test_basic_chunking():
    text = "Hello world. " * 100
    chunks = chunk_text(text, source_id="test.txt", source_type="text")
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.content) <= 600  # size + some overlap tolerance


def test_empty_text():
    chunks = chunk_text("", source_id="empty.txt", source_type="text")
    assert chunks == []


def test_chunk_metadata():
    chunks = chunk_text("Some text here.", source_id="doc.pdf", source_type="pdf", metadata={"pages": 5})
    assert chunks[0].source_id == "doc.pdf"
    assert chunks[0].source_type == "pdf"
    assert chunks[0].metadata["pages"] == 5


def test_chunk_indices_are_sequential():
    text = ("paragraph " * 50 + "\n\n") * 10
    chunks = chunk_text(text, source_id="test.txt", source_type="text")
    indices = [c.index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_overlap_applied():
    # Make two distinct paragraphs that will each become a chunk
    para_a = "Alpha beta gamma delta epsilon. " * 20   # ~640 chars
    para_b = "Zeta eta theta iota kappa. " * 20
    text = para_a + "\n\n" + para_b
    chunks = chunk_text(text, source_id="test.txt", source_type="text", chunk_size=512, chunk_overlap=64)
    if len(chunks) >= 2:
        # Second chunk should contain tail of first
        tail = chunks[0].content[-30:]
        assert tail[:10] in chunks[1].content
