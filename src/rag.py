"""Retrieval-augmented Q&A over the Java handbook (Markdown).

Retrieval is TF-IDF + cosine similarity over section chunks, so it runs offline with no API key.
`answer(..., llm=True)` optionally sends the retrieved passages to Claude for a grounded answer.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path(__file__).resolve().parent.parent / "data"
HEADING = re.compile(r"^###\s+(\d+\.\d+)\s+(.*)$")
MAX_WORDS = 220


@dataclass
class Chunk:
    section: str
    title: str
    text: str


def chunk_markdown(text, max_words=MAX_WORDS):
    """Split on '### N.M Title' headings; long sections become overlapping word windows."""
    sections, cur = [], None
    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            cur = [m.group(1), m.group(2).strip(), []]
            sections.append(cur)
        elif cur is not None:
            cur[2].append(line)
    chunks = []
    for num, title, lines in sections:
        words = " ".join(lines).split()
        step = max(max_words // 2, 1)
        for start in range(0, max(len(words), 1), step):
            window = words[start:start + max_words]
            if window:
                chunks.append(Chunk(num, title, f"{title}. " + " ".join(window)))
            if start + max_words >= len(words):
                break
    return chunks


class Retriever:
    def __init__(self, chunks, ngram_range=(1, 1)):
        self.chunks = chunks
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=ngram_range)
        self.matrix = self.vec.fit_transform(c.text for c in chunks)

    @classmethod
    def from_dir(cls, directory=DATA, **kw):
        chunks = []
        for path in sorted(Path(directory).glob("*.md")):
            chunks += chunk_markdown(path.read_text(encoding="utf-8"))
        return cls(chunks, **kw)

    def search(self, query, k=3):
        scores = (self.matrix @ self.vec.transform([query]).T).toarray().ravel()
        return [(self.chunks[i], float(scores[i])) for i in np.argsort(-scores)[:k]]

    def ranked_sections(self, query, k=5):
        """Distinct section numbers in rank order (a section may own several chunks)."""
        seen = []
        for chunk, _ in self.search(query, k=k * 3):
            if chunk.section not in seen:
                seen.append(chunk.section)
        return seen[:k]


def answer(question, retriever, k=3, llm=False):
    hits = retriever.search(question, k=k)
    context = "\n\n".join(f"[{c.section} {c.title}]\n{c.text}" for c, _ in hits)
    if not llm:
        best, score = hits[0]
        return f"(section {best.section}: {best.title}, score {score:.2f})\n{best.text[:600]}"
    import anthropic  # optional dependency
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-5-5", max_tokens=600,
        messages=[{"role": "user", "content":
                   "Answer using ONLY the handbook excerpts below; cite section numbers; "
                   f"say so if they don't contain the answer.\n\n{context}\n\nQuestion: {question}"}])
    return msg.content[0].text


if __name__ == "__main__":
    import sys
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    q = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))
    print(answer(q, Retriever.from_dir(), llm="--llm" in flags))
