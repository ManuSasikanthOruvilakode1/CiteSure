from citesure.ingestion.chunker import RecursiveChunker, split_markdown_sections
from citesure.types import Document


def test_sections_track_header_path_and_skip_code_fences():
    md = "# API\nintro\n## Auth\nuse keys\n```\n# not a header\n```\n# Errors\ncodes"
    sections = split_markdown_sections(md)
    assert [s for s, _ in sections] == ["API", "API > Auth", "Errors"]
    assert "# not a header" in sections[1][1]


def test_chunks_respect_size_budget():
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5)
    text = " ".join(f"word{i}." for i in range(200))
    pieces = chunker.split_text(text)
    assert len(pieces) > 5
    assert all(len(p.split()) <= 20 for p in pieces)


def test_overlap_carries_tail_into_next_chunk():
    chunker = RecursiveChunker(chunk_size=10, chunk_overlap=4)
    text = "\n".join(f"line {i} here" for i in range(12))  # 3 tokens per line
    pieces = chunker.split_text(text)
    assert len(pieces) >= 2
    first_tail = pieces[0].splitlines()[-1]
    assert pieces[1].startswith(first_tail)


def test_all_content_is_preserved():
    chunker = RecursiveChunker(chunk_size=15, chunk_overlap=3)
    words = [f"w{i}" for i in range(100)]
    pieces = chunker.split_text(" ".join(words))
    seen = {w for p in pieces for w in p.split()}
    assert seen == set(words)


def test_chunk_document_ids_are_unique_and_carry_metadata():
    doc = Document(text="# A\n" + "x " * 50 + "\n# B\ny", source="doc.md", metadata={"format": "md"})
    chunks = RecursiveChunker(chunk_size=20, chunk_overlap=2).chunk_document(doc)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert chunks[0].section == "A" and chunks[-1].section == "B"
    assert all(c.metadata["format"] == "md" for c in chunks)
