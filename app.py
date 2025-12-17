"""Simple paper-aware chatbot with local storage."""

from __future__ import annotations

import argparse
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

try:
    from pypdf import PdfReader  # type: ignore
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore

DATA_DIR = Path(__file__).parent / "data"
STORE_PATH = DATA_DIR / "papers.json"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\w']+", text.lower())


def _chunk_text(text: str, max_len: int = 800) -> List[str]:
    chunks: List[str] = []
    buffer: List[str] = []
    buffer_len = 0
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        separator = 1 if buffer else 0
        if buffer_len + separator + len(paragraph) > max_len and buffer:
            chunks.append("\n".join(buffer))
            buffer = []
            buffer_len = 0
            separator = 0
        buffer.append(paragraph)
        buffer_len += separator + len(paragraph)
    if buffer:
        chunks.append("\n".join(buffer))
    return chunks or [text]


def _load_json(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        if PdfReader is None:
            raise RuntimeError("pypdf is required for PDF ingestion. Install dependencies first.")
        reader = PdfReader(str(path))
        return "\n".join(filter(None, (page.extract_text() or "" for page in reader.pages)))
    return path.read_text(encoding="utf-8")


@dataclass
class Paper:
    id: str
    title: str
    source: str
    chunks: List[str]

    @classmethod
    def from_text(cls, title: str, source: str, text: str) -> "Paper":
        return cls(id=str(uuid.uuid4()), title=title or Path(source).stem, source=source, chunks=_chunk_text(text))


@dataclass
class ScoredChunk:
    score: float
    chunk: str
    title: str
    paper_id: str


class PaperStore:
    def __init__(self, store_path: Path = STORE_PATH):
        self.store_path = store_path
        self.data_dir = store_path.parent

    def load(self) -> List[Paper]:
        raw = _load_json(self.store_path)
        return [Paper(**item) for item in raw]

    def save(self, papers: Sequence[Paper]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        serialised = [paper.__dict__ for paper in papers]
        self.store_path.write_text(json.dumps(serialised, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_paper_from_text(self, title: str, source: str, text: str) -> Paper:
        papers = self.load()
        paper = Paper.from_text(title, source, text)
        papers.append(paper)
        self.save(papers)
        return paper

    def add_paper_from_path(self, path: Path, title: str | None = None) -> Paper:
        text = extract_text(path)
        return self.add_paper_from_text(title or path.stem, str(path), text)

    def search(self, query: str, top_k: int = 1) -> List[ScoredChunk]:
        papers = self.load()
        if not papers:
            return []
        top_k = max(1, top_k)
        q_tokens = set(_tokenize(query))
        scored: List[ScoredChunk] = []
        for paper in papers:
            for chunk in paper.chunks:
                c_tokens = set(_tokenize(chunk))
                overlap = len(q_tokens & c_tokens)
                score = overlap / (len(q_tokens) + 1e-6)
                scored.append(ScoredChunk(score=score, chunk=chunk, title=paper.title, paper_id=paper.id))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


def cli() -> None:
    parser = argparse.ArgumentParser(description="Chat using your ingested papers.")
    sub = parser.add_subparsers(dest="command", required=True)

    add_cmd = sub.add_parser("add", help="Add a paper (pdf or txt).")
    add_cmd.add_argument("path", type=Path, help="Path to paper file (.pdf or .txt).")
    add_cmd.add_argument("--title", help="Optional title to store.")

    chat_cmd = sub.add_parser("chat", help="Ask a question based on stored papers.")
    chat_cmd.add_argument("question", help="Your question")
    chat_cmd.add_argument("--top", type=int, default=1, help="How many passages to show.")

    list_cmd = sub.add_parser("list", help="List stored papers.")
    list_cmd.add_argument("--verbose", action="store_true", help="Show chunk counts.")

    args = parser.parse_args()
    store = PaperStore()

    if args.command == "add":
        paper = store.add_paper_from_path(args.path, title=args.title)
        print(f"Stored '{paper.title}' from {paper.source} with {len(paper.chunks)} chunks.")
    elif args.command == "chat":
        hits = store.search(args.question, top_k=max(1, args.top))
        if not hits:
            print("No relevant passages found. Add papers first with `python app.py add <path>`.")
            return
        for idx, hit in enumerate(hits, start=1):
            print(f"[{idx}] {hit.title} (score: {hit.score:.2f})")
            print(hit.chunk)
            print("-" * 40)
    elif args.command == "list":
        papers = store.load()
        if not papers:
            print("No papers stored yet.")
            return
        for paper in papers:
            extra = f" ({len(paper.chunks)} chunks)" if args.verbose else ""
            print(f"- {paper.title}{extra}")


if __name__ == "__main__":
    cli()
