"""Markdown-header-aware recursive chunking with token overlap.

Documents are first split on Markdown headers so that a chunk never straddles two
sections, and each chunk carries its header path (e.g. ``Auth > Tokens``) as metadata.
Sections larger than the token budget are split recursively on progressively finer
separators (paragraph -> line -> sentence -> word) and greedily re-merged with overlap.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

from citesure.types import Chunk, Document

TokenCounter = Callable[[str], int]

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def whitespace_token_count(text: str) -> int:
    """Cheap tokenizer-free approximation: one token per whitespace-delimited word."""
    return len(text.split())


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown into ``(header_path, body)`` pairs, ignoring headers inside code fences."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(title for _, title in stack), body))
        buf.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else _HEADER_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2)))
        else:
            buf.append(line)
    flush()
    return sections


class RecursiveChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 51,
        token_counter: TokenCounter = whitespace_token_count,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.count = token_counter

    def split_text(self, text: str) -> list[str]:
        pieces = self._split_recursive(text.strip(), _SEPARATORS)
        return self._merge(pieces)

    def chunk_document(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section, body in split_markdown_sections(doc.text):
            for text in self.split_text(body):
                idx = len(chunks)
                digest = hashlib.sha1(f"{doc.source}|{idx}|{text}".encode()).hexdigest()[:12]
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.source}#{idx}-{digest}",
                        text=text,
                        source=doc.source,
                        section=section,
                        metadata={**doc.metadata, "chunk_index": idx},
                    )
                )
        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        return [chunk for doc in docs for chunk in self.chunk_document(doc)]

    # -- internals ---------------------------------------------------------

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Break text into pieces each within the token budget, keeping separators attached."""
        if self.count(text) <= self.chunk_size:
            return [text] if text else []
        for i, sep in enumerate(separators):
            if sep in text:
                parts = [p + sep for p in text.split(sep)]
                parts[-1] = parts[-1][: -len(sep)]
                out: list[str] = []
                for part in parts:
                    if self.count(part) > self.chunk_size:
                        out.extend(self._split_recursive(part, separators[i + 1 :]))
                    elif part:
                        out.append(part)
                return out
        # No separator left: hard split on words.
        words = text.split()
        return [
            " ".join(words[i : i + self.chunk_size])
            for i in range(0, len(words), self.chunk_size)
        ]

    def _merge(self, pieces: list[str]) -> list[str]:
        """Greedily pack pieces into chunks, carrying a token-bounded tail as overlap."""
        chunks: list[str] = []
        window: list[str] = []
        window_tokens = 0
        for piece in pieces:
            n = self.count(piece)
            if window and window_tokens + n > self.chunk_size:
                chunks.append("".join(window).strip())
                # Keep trailing pieces up to the overlap budget as the start of the next chunk.
                tail: list[str] = []
                tail_tokens = 0
                for prev in reversed(window):
                    t = self.count(prev)
                    if tail_tokens + t > self.chunk_overlap or tail_tokens + t + n > self.chunk_size:
                        break
                    tail.insert(0, prev)
                    tail_tokens += t
                window, window_tokens = tail, tail_tokens
            window.append(piece)
            window_tokens += n
        if window:
            chunks.append("".join(window).strip())
        return [c for c in chunks if c]
