#!/usr/bin/env python3
"""
Knowledge Service - Main interface for TradingAgents to access knowledge base.

This service provides:
1. Search knowledge base for relevant context
2. Add new PDFs to existing knowledge base (incremental)
3. Inject knowledge into agent prompts

Usage with TradingAgents:
    from knowledge_base import KnowledgeService

    kb = KnowledgeService()
    context = kb.search("value investing", k=3)
    prompt = kb.inject_knowledge(prompt, "value investing", k=3)
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

KB_DIR = Path(__file__).parent

# Import from same directory - handle both package and direct contexts
try:
    from .turbovec_retriever import KnowledgeRetriever
except ImportError:
    from turbovec_retriever import KnowledgeRetriever


class KnowledgeService:
    """
    Main interface for TradingAgents to access the knowledge base.

    Provides semantic search, incremental updates, and prompt injection.
    """

    DEFAULT_INDEX_PATH = str(KB_DIR / "data" / "knowledge_base")
    DEFAULT_OLLAMA_URL = "http://localhost:11434"
    DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"

    def __init__(
        self,
        index_path: Optional[str] = None,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        auto_load: bool = True,
    ):
        """
        Initialize Knowledge Service.

        Args:
            index_path: Path to TurboVec index (default: knowledge_base/data/knowledge_base)
            ollama_url: Ollama API URL for embeddings
            embedding_model: Ollama embedding model name
            auto_load: If True, load existing index on init
        """
        self.index_path = index_path or self.DEFAULT_INDEX_PATH
        self.ollama_url = ollama_url
        self.embedding_model = embedding_model
        self.retriever: Optional[KnowledgeRetriever] = None

        if auto_load:
            self.load_index()

    def load_index(self) -> int:
        """Load existing knowledge base index."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"Knowledge base index not found at {self.index_path}. "
                "Run 'python build_knowledge_base.py' first."
            )

        self.retriever = KnowledgeRetriever(
            ollama_base_url=self.ollama_url,
            embedding_model=self.embedding_model,
            index_path=self.index_path,
        )
        return len(self.retriever)

    def search(
        self,
        query: str,
        k: int = 5,
        category: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base for relevant context.

        Args:
            query: Search query text
            k: Number of results to return
            category: Filter by category (e.g., "Investing_Books", "Psychological_Books")
            min_similarity: Minimum similarity threshold

        Returns:
            List of results with content, metadata, and similarity score
        """
        if self.retriever is None:
            raise RuntimeError("Knowledge base not loaded. Call load_index() first.")

        results = self.retriever.search(query=query, k=k, category=category)

        # Filter by minimum similarity
        if min_similarity > 0:
            results = [r for r in results if r.get("similarity", 0) >= min_similarity]

        return results

    def get_context(
        self,
        query: str,
        k: int = 5,
        max_tokens: int = 2000,
        category: Optional[str] = None,
    ) -> str:
        """
        Get formatted context string for agent prompts.

        Args:
            query: Search query
            k: Number of results
            max_tokens: Maximum total tokens in context
            category: Filter by category

        Returns:
            Formatted context string
        """
        results = self.search(query, k=k, category=category)

        if not results:
            return "No relevant knowledge found."

        # Build context from results
        context_parts = []
        total_length = 0

        for i, r in enumerate(results, 1):
            book = r["metadata"].get("book_title", "Unknown")
            # Extract just the filename from the full path
            if "/" in book:
                book = book.split("/")[-1].replace("_", " ").replace(".pdf", "")

            content = r["content"].strip()
            # Truncate if too long
            if len(content) > 500:
                content = content[:500] + "..."

            context_parts.append(f"[{book}]\n{content}")
            total_length += len(context_parts[-1])

            if total_length > max_tokens * 4:  # Rough char to token conversion
                break

        return "\n\n---\n\n".join(context_parts)

    def inject_knowledge(
        self,
        prompt: str,
        query: str,
        k: int = 5,
        section_name: str = "Relevant Knowledge",
        **kwargs,
    ) -> str:
        """
        Inject relevant knowledge into an agent prompt.

        Args:
            prompt: Original agent prompt
            query: What knowledge to search for
            k: Number of results
            section_name: Section header in final prompt
            **kwargs: Additional args for get_context()

        Returns:
            Enhanced prompt with knowledge context
        """
        context = self.get_context(query, k=k, **kwargs)

        enhanced_prompt = f"""{prompt}

== {section_name} ==
Use the following knowledge from financial books and research to inform your analysis:

{context}

== End {section_name} ==
"""
        return enhanced_prompt

    def add_pdfs(
        self,
        pdf_paths: List[str],
        market_data_dir: Optional[str] = None,
    ) -> int:
        """
        Add new PDFs to the knowledge base (incremental update).

        Args:
            pdf_paths: List of PDF file paths to add
            market_data_dir: Optional market data directory for categorization

        Returns:
            Number of chunks added
        """
        # Import helper functions - handle both package and direct contexts
        try:
            from .build_knowledge_base import extract_text_from_pdf, chunk_text_semantic
        except ImportError:
            from build_knowledge_base import extract_text_from_pdf, chunk_text_semantic
        import hashlib
        import numpy as np

        if self.retriever is None:
            raise RuntimeError("Knowledge base not loaded.")

        total_chunks = 0

        for pdf_path in pdf_paths:
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                print(f"Warning: PDF not found: {pdf_path}")
                continue

            # Extract and chunk
            text = extract_text_from_pdf(str(pdf_file))
            book_title = pdf_file.stem.replace("_", " ").replace("-", " ")

            chunks = chunk_text_semantic(text, book_title)

            # Determine category
            category = "Unknown"
            if market_data_dir:
                md_path = Path(market_data_dir)
                for cat_dir in md_path.iterdir():
                    if cat_dir.is_dir() and pdf_file.parent == cat_dir:
                        category = cat_dir.name
                        break

            # Add metadata
            for chunk in chunks:
                chunk["metadata"]["category"] = category

            # Generate embeddings
            texts_to_embed = [c["content"] for c in chunks]
            embeddings = []

            for text in texts_to_embed:
                try:
                    emb = self.retriever.get_embedding(text)
                    embeddings.append(emb)
                except Exception as e:
                    print(f"Embedding failed: {e}")
                    embeddings.append(np.zeros(self.retriever.turbovec.dim))

            embeddings = np.array(embeddings)

            # Add to index
            added = self.retriever.turbovec.add_documents(chunks, embeddings)
            total_chunks += added

            print(f"Added {added} chunks from {pdf_file.name}")

        # Save updated index
        self.retriever.save(self.index_path)
        print(f"Saved updated index to {self.index_path}")

        return total_chunks

    def stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        if self.retriever is None:
            return {"status": "not_loaded"}

        # Count by category
        categories = {}
        for doc in self.retriever.turbovec.documents:
            cat = doc.metadata.get("category", "Unknown")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "status": "loaded",
            "total_documents": len(self.retriever),
            "index_path": self.index_path,
            "categories": categories,
        }

    def __len__(self) -> int:
        """Return number of documents in index."""
        if self.retriever is None:
            return 0
        return len(self.retriever)


# Global instance for easy import
_kb_instance: Optional[KnowledgeService] = None


def get_knowledge_service() -> KnowledgeService:
    """Get or create global KnowledgeService instance."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeService()
    return _kb_instance


def search_knowledge(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Convenience function for quick knowledge search."""
    kb = get_knowledge_service()
    return kb.search(query, k=k)
