#!/usr/bin/env python3
"""
Build Knowledge Base from Market Data PDFs

Complete pipeline:
1. Parse PDFs (PyMuPDF or dots.mocr)
2. Chunk text with semantic boundaries
3. Generate embeddings with Ollama
4. Build TurboVec index for RAG retrieval

Usage:
    python build_knowledge_base.py --market-data-dir ../market_data/ --output ./data/knowledge_base/
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import pymupdf
except ImportError:
    os.system("pip install pymupdf")
    import pymupdf

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    os.system("pip install langchain-text-splitters")
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from turbovec_retriever import KnowledgeRetriever, TurboVecRetriever


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        doc = pymupdf.open(pdf_path)
        text_parts = []

        for page in doc:
            text_parts.append(page.get_text())

        doc.close()
        return "\n\n".join(text_parts)
    except Exception as e:
        raise Exception(f"Failed to extract text from {pdf_path}: {e}")


def chunk_text_semantic(text: str, book_title: str, chunk_size: int = 2000, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split text into semantically coherent chunks.
    Uses LangChain's text splitter with markdown awareness.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )

    chunks = text_splitter.split_text(text)

    # Add metadata to each chunk
    chunk_dicts = []
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 50:  # Skip very small chunks
            continue

        chunk_id = hashlib.md5(f"{book_title}_{i}_{chunk[:100]}".encode()).hexdigest()[:12]
        chunk_dicts.append({
            "id": chunk_id,
            "content": chunk,
            "metadata": {
                "book_title": book_title,
                "chunk_index": i,
                "chunk_size": len(chunk),
            },
        })

    return chunk_dicts


def load_all_pdfs(market_data_dir: Path, logger) -> List[Dict[str, Any]]:
    """Load all PDFs from market_data directory."""
    all_chunks = []

    # Find all subdirectories (Investing_Books, Psychological_Books, etc.)
    categories = [d for d in market_data_dir.iterdir() if d.is_dir()]

    if not categories:
        logger.warning(f"No subdirectories found in {market_data_dir}")
        return all_chunks

    for category_dir in categories:
        pdf_files = sorted(category_dir.glob("*.pdf"))

        if not pdf_files:
            logger.debug(f"No PDFs in {category_dir.name}")
            continue

        logger.info(f"Category: {category_dir.name} ({len(pdf_files)} PDFs)")

        for pdf_file in pdf_files:
            try:
                logger.info(f"  Extracting: {pdf_file.name}...")
                text = extract_text_from_pdf(str(pdf_file))

                if not text or len(text.strip()) < 100:
                    logger.warning(f"    Empty or very short: {pdf_file.name}")
                    continue

                # Chunk the text
                book_title = pdf_file.stem.replace("_", " ").replace("-", " ")
                chunks = chunk_text_semantic(text, book_title)

                logger.info(f"    {len(chunks)} chunks from {len(text):,} chars")

                # Add category to metadata
                for chunk in chunks:
                    chunk["metadata"]["category"] = category_dir.name

                all_chunks.extend(chunks)

            except Exception as e:
                logger.error(f"  ERROR {pdf_file.name}: {e}")

    return all_chunks


def build_knowledge_base(
    market_data_dir: str,
    output_dir: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    ollama_url: str = "http://localhost:11434",
    embedding_model: str = "mxbai-embed-large",
    verbose: bool = False,
):
    """Main pipeline to build knowledge base."""
    import logging

    log_level = "DEBUG" if verbose else "INFO"
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    logger = logging.getLogger("build_knowledge_base")

    market_path = Path(market_data_dir)
    output_path = Path(output_dir)

    if not market_path.exists():
        logger.error(f"Market data directory not found: {market_path}")
        sys.exit(1)

    # Step 1: Load and chunk all PDFs
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and chunking PDFs")
    logger.info("=" * 60)

    all_chunks = load_all_pdfs(market_path, logger)

    if not all_chunks:
        logger.error("No chunks extracted. Check if PDFs exist in market_data/")
        sys.exit(1)

    logger.info("")
    logger.info(f"Total chunks: {len(all_chunks)}")

    # Step 2: Initialize TurboVec retriever
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Initializing TurboVec Retriever")
    logger.info("=" * 60)

    retriever = KnowledgeRetriever(
        ollama_base_url=ollama_url,
        embedding_model=embedding_model,
        index_path=None,  # Start fresh
    )

    # Step 3: Generate embeddings and build index
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Generating Embeddings & Building Index")
    logger.info("=" * 60)

    documents_for_index = []
    texts_to_embed = []

    for chunk in all_chunks:
        documents_for_index.append({
            "id": chunk["id"],
            "content": chunk["content"],
            "metadata": chunk["metadata"],
        })
        texts_to_embed.append(chunk["content"])

    # Generate embeddings in batches
    logger.info(f"Generating {len(texts_to_embed)} embeddings with Ollama...")
    embeddings = []
    batch_size = 20
    failed = 0

    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i + batch_size]
        batch_embeddings = []

        for text in batch:
            try:
                emb = retriever.get_embedding(text)
                batch_embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Embedding failed: {e}")
                batch_embeddings.append(np.zeros(retriever.turbovec.dim))
                failed += 1

        embeddings.extend(batch_embeddings)
        progress = min(i + batch_size, len(texts_to_embed))
        logger.debug(f"  {progress}/{len(texts_to_embed)} ({progress/len(texts_to_embed)*100:.1f}%)")

    embeddings = np.array(embeddings)

    logger.info(f"Embedding complete. Failed: {failed}/{len(texts_to_embed)}")

    # Add to TurboVec index
    logger.info("Adding to TurboVec index...")
    added = retriever.turbovec.add_documents(documents_for_index, embeddings)
    logger.info(f"Added {added} documents to index")

    # Step 4: Save index
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: Saving Index")
    logger.info("=" * 60)

    output_path.mkdir(parents=True, exist_ok=True)
    retriever.save(str(output_path))

    logger.info(f"Index saved to: {output_path}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"PDFs processed: {len(set(c['metadata']['book_title'] for c in all_chunks))}")
    logger.info(f"Total chunks: {len(all_chunks)}")
    logger.info(f"Documents indexed: {added}")
    logger.info(f"Index size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. Start API: python knowledge_api_turbovec.py --port 3001 --index-path {output_path}")
    logger.info(f"  2. Test search:")
    logger.info(f"     curl -X POST http://localhost:3001/knowledge/search \\")
    logger.info(f"       -H 'Content-Type: application/json' \\")
    logger.info(f"       -d '{{\"query\": \"value investing\", \"limit\": 3}}'")
    logger.info("")

    return {
        "chunks": len(all_chunks),
        "indexed": added,
        "failed_embeddings": failed,
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build knowledge base from market data PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_knowledge_base.py --market-data-dir ../market_data/ --output ./data/knowledge_base/
  python build_knowledge_base.py --market-data-dir ../market_data/ --chunk-size 3000 --verbose
        """
    )

    parser.add_argument("--market-data-dir", type=str, required=True,
                        help="Directory containing PDFs (Investing_Books, Psychological_Books, etc.)")
    parser.add_argument("--output", type=str, default="./data/knowledge_base",
                        help="Output directory for TurboVec index")
    parser.add_argument("--chunk-size", type=int, default=2000,
                        help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                        help="Chunk overlap in characters")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                        help="Ollama API URL")
    parser.add_argument("--embedding-model", type=str, default="mxbai-embed-large",
                        help="Ollama embedding model")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    result = build_knowledge_base(
        market_data_dir=args.market_data_dir,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        ollama_url=args.ollama_url,
        embedding_model=args.embedding_model,
        verbose=args.verbose,
    )

    print(f"\nKnowledge base built successfully!")
    print(f"Index path: {result['output_path']}")


if __name__ == "__main__":
    main()
