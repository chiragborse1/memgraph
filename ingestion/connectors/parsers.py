"""
Connectors — each returns (source_id: str, text: str, metadata: dict).
New connectors just need to implement the same signature.
"""

import re
from pathlib import Path
import httpx
import pymupdf                        # fitz
from docx import Document as DocxDoc
from bs4 import BeautifulSoup


# ── PDF ──────────────────────────────────────────────────────────────────────

def parse_pdf(file_path: str | Path) -> tuple[str, str, dict]:
    path = Path(file_path)
    doc = pymupdf.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    text = "\n\n".join(pages)
    doc.close()
    return (
        path.name,
        _clean(text),
        {"pages": len(pages), "file": path.name},
    )


# ── DOCX ─────────────────────────────────────────────────────────────────────

def parse_docx(file_path: str | Path) -> tuple[str, str, dict]:
    path = Path(file_path)
    doc = DocxDoc(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    return (
        path.name,
        _clean(text),
        {"paragraphs": len(paragraphs), "file": path.name},
    )


# ── Plain text / Markdown ─────────────────────────────────────────────────────

def parse_text(file_path: str | Path) -> tuple[str, str, dict]:
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return path.name, _clean(text), {"file": path.name}


# ── Web URL ───────────────────────────────────────────────────────────────────

async def parse_url(url: str) -> tuple[str, str, dict]:
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": "MemGraph/1.0"})
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "aside", "iframe"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title else url
    # Prefer <article> or <main>, fall back to <body>
    container = soup.find("article") or soup.find("main") or soup.body
    text = container.get_text(separator="\n") if container else soup.get_text()

    return (
        url,
        _clean(text),
        {"url": url, "title": title},
    )


# ── Router ────────────────────────────────────────────────────────────────────

async def ingest_source(source: str) -> tuple[str, str, str, dict]:
    """
    Auto-detect source type and return (source_id, source_type, text, metadata).
    source can be: file path OR URL.
    """
    if source.startswith("http://") or source.startswith("https://"):
        source_id, text, meta = await parse_url(source)
        return source_id, "web", text, meta

    path = Path(source)
    ext = path.suffix.lower()

    if ext == ".pdf":
        source_id, text, meta = parse_pdf(path)
        return source_id, "pdf", text, meta
    elif ext == ".docx":
        source_id, text, meta = parse_docx(path)
        return source_id, "docx", text, meta
    elif ext in {".txt", ".md", ".rst"}:
        source_id, text, meta = parse_text(path)
        return source_id, "text", text, meta
    else:
        raise ValueError(f"Unsupported source type: {ext}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Normalise whitespace, remove null bytes."""
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
