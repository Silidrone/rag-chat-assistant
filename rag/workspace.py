"""A caller-supplied knowledge base, and the cache that keeps it affordable.

The articles in `data/` are the default. A caller may send their own set with a
question instead, which is what lets the UI offer an editable knowledge base on
a public URL without the service holding shared mutable state: your edits
travel with your request, and nobody can change what another visitor sees.

Embedding is cached per chunk, keyed by the exact text that gets embedded, so
editing one article re-embeds that article and nothing else. The shipped set is
embedded once at start-up and never again.

Limits exist because every distinct chunk is a paid embedding call and this
endpoint is public. They are deliberately generous enough to paste a real
help centre into, and small enough that nobody can run up a bill with one POST.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence

from .index import Embedder, VectorIndex, build_index

MAX_ARTICLES = 24
MAX_ARTICLE_CHARS = 8_000
MAX_TOTAL_CHARS = 60_000

# Bounded so a stream of distinct knowledge bases cannot grow the process
# without limit. Evicts oldest first; a miss costs one embedding call.
MAX_CACHED_VECTORS = 4_096

_VECTORS: OrderedDict[str, list[float]] = OrderedDict()


class WorkspaceError(ValueError):
    """The supplied knowledge base is unusable, and the caller can fix it."""


def validate(raw: object) -> list[tuple[str, str]]:
    """Turn an untrusted `articles` payload into (source, text) pairs.

    Rejects rather than repairs. A silently truncated knowledge base would
    produce answers that look grounded against text the caller never sent.
    """
    if not isinstance(raw, list) or not raw:
        raise WorkspaceError("articles must be a non-empty list")
    if len(raw) > MAX_ARTICLES:
        raise WorkspaceError(f"at most {MAX_ARTICLES} articles, got {len(raw)}")

    documents: list[tuple[str, str]] = []
    seen: set[str] = set()
    total = 0

    for position, item in enumerate(raw):
        if not isinstance(item, dict):
            raise WorkspaceError(f"article {position} is not an object")

        source = str(item.get("source", "")).strip()
        text = str(item.get("text", "")).strip()

        if not source:
            raise WorkspaceError(f"article {position} has no source name")
        if "/" in source or "\\" in source:
            raise WorkspaceError(f"source {source!r} must not contain a path separator")
        if source.lower() in seen:
            raise WorkspaceError(f"duplicate source {source!r}")
        if not text:
            raise WorkspaceError(f"article {source!r} is empty")
        if len(text) > MAX_ARTICLE_CHARS:
            raise WorkspaceError(
                f"article {source!r} is {len(text)} characters, limit is {MAX_ARTICLE_CHARS}"
            )

        seen.add(source.lower())
        total += len(text)
        if total > MAX_TOTAL_CHARS:
            raise WorkspaceError(f"knowledge base exceeds {MAX_TOTAL_CHARS} characters")

        documents.append((source, text))

    return documents


def caching_embedder(embedder: Embedder) -> Embedder:
    """Wrap an embedder so identical text is only ever paid for once.

    Order is preserved and duplicates within one batch collapse to a single
    upstream entry, which matters because `build_index` embeds every chunk and
    an edit usually changes one of them.
    """

    def embed(texts: list[str]) -> list[list[float]]:
        keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]

        missing: list[str] = []
        missing_keys: list[str] = []
        for key, text in zip(keys, texts):
            if key in _VECTORS or key in missing_keys:
                continue
            missing_keys.append(key)
            missing.append(text)

        if missing:
            for key, vector in zip(missing_keys, embedder(missing)):
                _VECTORS[key] = vector
                _VECTORS.move_to_end(key)
            while len(_VECTORS) > MAX_CACHED_VECTORS:
                _VECTORS.popitem(last=False)

        resolved = []
        for key in keys:
            _VECTORS.move_to_end(key)
            resolved.append(_VECTORS[key])
        return resolved

    return embed


def index_for(documents: Sequence[tuple[str, str]], embedder: Embedder) -> VectorIndex:
    """Build an index over `documents`, paying only for text not seen before."""
    return build_index(documents, caching_embedder(embedder))


def cache_size() -> int:
    """How many chunk vectors are held. Reported by /health."""
    return len(_VECTORS)
