"""
Vector Embedding and RAG Sub-System for Jithendra's Voice AI Portfolio.
Modular package providing corpus ingestion, Reciprocal Rank Fusion (RRF),
dense neural retrieval, and LRU vector caching.
"""

from .chunker import clean_text, chunk_text
from .corpus import build_full_corpus
from .retriever import (
    PortfolioVectorRetriever,
    get_retriever,
    search_knowledge_base,
)
from .types import KnowledgeChunk, SearchResult

__all__ = [
    "KnowledgeChunk",
    "SearchResult",
    "PortfolioVectorRetriever",
    "get_retriever",
    "search_knowledge_base",
    "clean_text",
    "chunk_text",
    "build_full_corpus",
]
