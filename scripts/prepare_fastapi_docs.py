"""Turn the FastAPI documentation (MIT license) into a self-contained Markdown corpus.

FastAPI's pages pull code examples from ``docs_src/`` through MkDocs include markers, so the
raw Markdown has holes where the code belongs. This script inlines those files, rewrites
MkDocs admonition/tab fences into plain Markdown, and skips pages that are not product
documentation (release notes, contributor and translation admin, generated API stubs).

Usage:
  git clone --depth 1 --filter=blob:none --sparse https://github.com/fastapi/fastapi.git fastapi-src
  git -C fastapi-src sparse-checkout set docs/en/docs docs_src
  python scripts/prepare_fastapi_docs.py fastapi-src
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

# Pages that are not product documentation, or that only render at build time.
EXCLUDE = {
    "release-notes.md", "_llm-test.md", "fastapi-people.md", "help-fastapi.md",
    "translations.md", "translation-banner.md", "external-links.md", "management.md",
    "management-tasks.md", "newsletter.md", "contributing.md",
}
EXCLUDE_DIRS = {"reference"}  # mkdocstrings stubs (`::: fastapi.APIRouter`), empty as Markdown

LANG = {".py": "python", ".js": "javascript", ".ts": "typescript", ".html": "html",
        ".css": "css", ".json": "json", ".toml": "toml", ".sh": "bash", ".txt": "text"}

# {* ../../docs_src/x/tutorial001_py310.py hl[3] *}  and  {!> ../../docs_src/x/file.js!}
_STAR_INCLUDE = re.compile(r"\{\*\s*(\S+)[^*]*\*\}")
_BANG_INCLUDE = re.compile(r"\{!>?\s*([^!]+?)\s*!\}")
_ADMONITION = re.compile(r"^/{3,4}\s*([\w-]+)?(?:\s*\|\s*(.*))?\s*$")
_HEADING_ANCHOR = re.compile(r"[ \t]*\{\s*#[\w-]+\s*\}[ \t]*$", re.MULTILINE)  # "# Title { #title }"
# Terminal-recording markup around console examples; it carries no content.
_TERMINAL_MARKUP = re.compile(r"</?(?:font|span|u)\b[^>]*>|^\s*<div class=\"termy\">\s*$|^\s*</div>\s*$",
                              re.MULTILINE)


def _clean_page(text: str) -> str:
    """Strip site-only markup from a page's own text (run before code is inlined)."""
    return _TERMINAL_MARKUP.sub("", _HEADING_ANCHOR.sub("", text))


def _inline(match: re.Match, repo: Path) -> str:
    rel = match.group(1).strip()
    target = repo / rel.removeprefix("../../")
    if not target.is_file():
        return f"(code example `{rel}` not available)"
    code = target.read_text(encoding="utf-8", errors="replace").rstrip()
    return f"```{LANG.get(target.suffix, '')}\n{code}\n```"


def _rewrite_admonitions(text: str) -> str:
    """`/// tip | Title` -> `**Tip: Title**`; `//// tab | Python 3.10+` -> `**Python 3.10+:**`."""
    out = []
    for line in text.splitlines():
        m = _ADMONITION.match(line)
        if not m:
            out.append(line)
        elif m.group(1) == "tab":
            out.append(f"**{m.group(2) or 'Tab'}:**")
        elif m.group(1):
            label = m.group(1).capitalize()
            out.append(f"**{label}: {m.group(2)}**" if m.group(2) else f"**{label}:**")
        # a bare `///` closes a block: drop it
    return "\n".join(out) + "\n"


def convert(repo: Path, out_dir: Path) -> tuple[int, int]:
    docs = repo / "docs" / "en" / "docs"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    pages = inlined = 0
    for src in sorted(docs.rglob("*.md")):
        rel = src.relative_to(docs)
        if rel.name in EXCLUDE or rel.parts[0] in EXCLUDE_DIRS:
            continue
        text = src.read_text(encoding="utf-8", errors="replace")
        inlined += len(_STAR_INCLUDE.findall(text)) + len(_BANG_INCLUDE.findall(text))
        text = _clean_page(text)
        text = _STAR_INCLUDE.sub(lambda m: _inline(m, repo), text)
        text = _BANG_INCLUDE.sub(lambda m: _inline(m, repo), text)
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_rewrite_admonitions(text), encoding="utf-8")
        pages += 1
    # Licence and provenance, without file extensions so the indexer skips them.
    shutil.copyfile(repo / "LICENSE", out_dir / "LICENSE")
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    ).stdout.strip()  # provenance only; "unknown" if git is unavailable
    (out_dir / "SOURCE").write_text(
        "FastAPI documentation (https://github.com/fastapi/fastapi, docs/en/docs)\n"
        f"Commit: {commit or 'unknown'}\nLicense: MIT (see LICENSE)\n"
        "Converted by scripts/prepare_fastapi_docs.py\n",
        encoding="utf-8",
    )
    return pages, inlined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", type=Path, help="sparse checkout of github.com/fastapi/fastapi")
    parser.add_argument("--out", type=Path, default=Path("data/documents/fastapi"))
    args = parser.parse_args()
    pages, inlined = convert(args.repo, args.out)
    print(f"Wrote {pages} pages to {args.out} ({inlined} code examples inlined)")


if __name__ == "__main__":
    main()
