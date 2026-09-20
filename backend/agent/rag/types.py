"""
Type definitions and data transfer objects for the Portfolio RAG system.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KnowledgeChunk:
    """Represents an atomic, searchable knowledge passage with semantic metadata."""
    id: str
    source: str
    title: str
    section: str
    category: str
    text: str
    keywords: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """Represents a scored search result returned by the vector retriever."""
    id: str
    source: str
    title: str
    section: str
    category: str
    text: str
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "section": self.section,
            "category": self.category,
            "text": self.text,
            "score": round(self.score, 4),
            "dense_score": round(self.dense_score, 4),
            "sparse_score": round(self.sparse_score, 4),
            "rrf_score": round(self.rrf_score, 4),
        }
