"""Chunking and the in-memory vector index.

Six short articles, so the index is one numpy matrix searched by dot product.
`search` is the seam a pgvector or Qdrant backend would sit behind.

The embedder is injected as a plain callable, which is what lets tests/ run with
no network and no API key.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

# Articles are chunked whole unless they exceed this, then split on paragraph
# boundaries. An oversized single paragraph is left whole, since splitting
# mid-sentence retrieves badly.
MAX_CHUNK_CHARS = 800

Embedder = Callable[[list[str]], list[list[float]]]

_HEADING_LINE = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$")


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of the knowledge base.

    `id` is opaque and stable per start-up. It is what the model cites, so it
    never types a filename itself.
    """

    id: str
    source: str
    heading: str
    text: str

    @property
    def embedding_text(self) -> str:
        """Heading plus body: the heading carries topic words the body omits."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text


@dataclass(frozen=True)
class Hit:
    """A chunk paired with its similarity to the query."""

    chunk: Chunk
    score: float


def chunk_document(source: str, text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split one article into chunks, preserving the heading on each.

    Ids are assigned by `build_index` once every document is known, so the
    chunks returned here carry a placeholder id.
    """
    chunks: list[Chunk] = []
    for heading, body in _sections(text):
        for piece in _pack_paragraphs(body, max_chars):
            chunks.append(Chunk(id="", source=source, heading=heading, text=piece))
    return chunks


def _sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs on heading lines."""
    sections: list[tuple[str, str]] = []
    heading = ""
    body: list[str] = []

    for line in text.splitlines():
        match = _HEADING_LINE.match(line)
        if match:
            if body:
                sections.append((heading, "\n".join(body).strip()))
                body = []
            heading = match.group("title")
        else:
            body.append(line)

    if body:
        sections.append((heading, "\n".join(body).strip()))
    return [(h, b) for h, b in sections if b]


def _pack_paragraphs(body: str, max_chars: int) -> list[str]:
    """Group paragraphs into pieces of at most `max_chars`, never splitting one."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    current: list[str] = []

    for paragraph in paragraphs:
        candidate = current + [paragraph]
        if current and len("\n\n".join(candidate)) > max_chars:
            pieces.append("\n\n".join(current))
            current = [paragraph]
        else:
            current = candidate

    if current:
        pieces.append("\n\n".join(current))
    return pieces


class VectorIndex:
    """Chunks plus their L2-normalised embeddings, searched by dot product.

    Normalising once at build time turns cosine similarity into a single matrix
    multiply at query time, which is why there is no division in `search`.
    """

    def __init__(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"got {len(chunks)} chunks but {vectors.shape[0]} vectors"
            )
        self._chunks = list(chunks)
        self._matrix = _normalise_rows(vectors)
        self._by_id = {chunk.id: chunk for chunk in self._chunks}

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        """Look up a chunk by id, or None if the id is unknown."""
        return self._by_id.get(chunk_id)

    def search(self, query_vector: Sequence[float], top_k: int) -> list[Hit]:
        """Return the `top_k` most similar chunks, highest score first."""
        if not self._chunks:
            return []
        query = _normalise_rows(np.asarray([query_vector], dtype=np.float32))[0]
        scores = self._matrix @ query
        # argsort is ascending, so take the tail and reverse it.
        best = np.argsort(scores)[::-1][:top_k]
        return [Hit(chunk=self._chunks[i], score=float(scores[i])) for i in best]


def build_index(
    documents: Iterable[tuple[str, str]],
    embedder: Embedder,
    max_chars: int = MAX_CHUNK_CHARS,
) -> VectorIndex:
    """Chunk every document, embed the chunks in one call, and index them."""
    chunks: list[Chunk] = []
    for source, text in documents:
        for chunk in chunk_document(source, text, max_chars):
            # Ids are positional and opaque: the model cites "c3", never a filename.
            chunks.append(
                Chunk(
                    id=f"c{len(chunks)}",
                    source=chunk.source,
                    heading=chunk.heading,
                    text=chunk.text,
                )
            )

    if not chunks:
        raise ValueError("no chunks to index: is data/ empty?")

    vectors = np.asarray(
        embedder([chunk.embedding_text for chunk in chunks]), dtype=np.float32
    )
    return VectorIndex(chunks, vectors)


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving all-zero rows alone."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)
