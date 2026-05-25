"""
Knowledge Base Module for Quantify Trading Platform

Provides RAG-based knowledge retrieval from financial books and documents.

Usage:
    from knowledge_base import KnowledgeService

    # Initialize with existing index
    kb = KnowledgeService(index_path="knowledge_base/data/knowledge_base")

    # Search for relevant knowledge
    results = kb.search("margin of safety investing", k=5)

    # Add new documents (e.g., new PDFs)
    kb.add_pdfs(["new_book.pdf"])
"""

try:
    from .service import KnowledgeService
    from .turbovec_retriever import KnowledgeRetriever, TurboVecRetriever
except ImportError:
    # Fallback for direct imports when not in package context
    from service import KnowledgeService
    from turbovec_retriever import KnowledgeRetriever, TurboVecRetriever

__all__ = ["KnowledgeService", "KnowledgeRetriever", "TurboVecRetriever"]
__version__ = "1.0.0"
