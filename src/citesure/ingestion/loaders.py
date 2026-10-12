"""Load raw documents (Markdown, plain text, PDF) from disk."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from citesure.types import Document

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}
PDF_SUFFIXES = {".pdf"}


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def load_document(path: Path, root: Path | None = None) -> Document | None:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
    elif suffix in PDF_SUFFIXES:
        text = _read_pdf(path)
    else:
        return None
    if not text.strip():
        return None

    source = path.relative_to(root).as_posix() if root else path.name
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return Document(
        text=text,
        source=source,
        metadata={"format": suffix.lstrip("."), "modified": modified},
    )


def load_directory(directory: Path) -> list[Document]:
    """Recursively load every supported document under ``directory`` in a stable order."""
    docs = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            doc = load_document(path, root=directory)
            if doc is not None:
                docs.append(doc)
    return docs
