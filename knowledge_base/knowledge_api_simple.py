#!/usr/bin/env python3
"""
Knowledge Service - Simple FastAPI REST API for Trading Knowledge Base
Uses Ollama for embeddings and TurboVec for fast semantic search.

Endpoints:
    POST /knowledge/search - Semantic search
    GET  /health - Health check

Usage:
    python knowledge_api_simple.py --port 3001 --index-path data/knowledge_base
"""

import os
import sys
import argparse
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from turbovec_retriever import KnowledgeRetriever

app = FastAPI(title="Trading Knowledge API", version="1.0.0")

# Add CORS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global retriever
retriever: Optional[KnowledgeRetriever] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
    category: Optional[str] = Field(None, description="Filter by category")


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int


class HealthResponse(BaseModel):
    status: str
    documents: int
    index_path: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Knowledge retriever not initialized")

    return {
        "status": "healthy",
        "documents": len(retriever),
        "index_path": retriever.turbovec.index_path or "memory",
    }


@app.post("/knowledge/search", response_model=SearchResponse)
async def search_knowledge(request: SearchRequest):
    """Search knowledge base semantically."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Knowledge retriever not initialized")

    try:
        results = retriever.search(
            query=request.query,
            k=request.limit,
            category=request.category,
        )

        return {
            "results": results,
            "total": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge/{doc_id}")
async def get_document(doc_id: str):
    """Get a specific document by ID."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Knowledge retriever not initialized")

    # Linear search for document by ID
    for doc in retriever.turbovec.documents:
        if doc.id == doc_id:
            return {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
            }

    raise HTTPException(status_code=404, detail="Document not found")


@app.get("/knowledge")
async def list_knowledge(
    offset: int = 0,
    limit: int = 20,
    category: Optional[str] = None,
):
    """List documents with pagination."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Knowledge retriever not initialized")

    docs = retriever.turbovec.documents

    # Filter by category if specified
    if category:
        docs = [d for d in docs if d.metadata.get("category") == category]

    # Paginate
    total = len(docs)
    docs = docs[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "documents": [
            {
                "id": doc.id,
                "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ],
    }


def init_retriever(index_path: str, ollama_url: str, embedding_model: str):
    """Initialize the knowledge retriever."""
    global retriever

    print(f"Initializing Knowledge Retriever...")
    print(f"  Ollama URL: {ollama_url}")
    print(f"  Embedding Model: {embedding_model}")
    print(f"  Index Path: {index_path}")

    retriever = KnowledgeRetriever(
        ollama_base_url=ollama_url,
        embedding_model=embedding_model,
        index_path=index_path,
    )

    print(f"Loaded {len(retriever)} documents from index")
    return retriever


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python knowledge_api_simple.py --port 3001 --index-path data/knowledge_base
        """
    )

    parser.add_argument("--port", type=int, default=3001, help="Port to run server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--index-path", type=str, required=True, help="Path to TurboVec index")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--embedding-model", type=str, default="mxbai-embed-large", help="Embedding model")

    args = parser.parse_args()

    # Initialize retriever
    init_retriever(args.index_path, args.ollama_url, args.embedding_model)

    # Start server
    import uvicorn
    print(f"\nStarting Knowledge API on {args.host}:{args.port}")
    print(f"Endpoints:")
    print(f"  GET  /health - Health check")
    print(f"  POST /knowledge/search - Search knowledge")
    print(f"  GET  /knowledge - List documents")
    print(f"  GET  /knowledge/{{id}} - Get document by ID")
    print()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
