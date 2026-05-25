#!/usr/bin/env python3
"""
Knowledge Service - FastAPI REST API for Trading Expertise Graph
Uses Ollama for embeddings and TurboVec for fast semantic search.

TurboVec provides:
- 4-bit quantized vector storage (4x memory efficiency vs float32)
- Fast approximate nearest neighbor search
- Optional runtime filtering by category

Endpoints:
    POST /knowledge/search - Semantic search for concepts
    GET  /knowledge/:id - Get concept by ID
    GET  /knowledge - List concepts with pagination
    GET  /strategies - Find strategies by conditions
    GET  /health - Health check with status

Usage:
    python knowledge_api_turbovec.py --port 3001
"""

import os
import sys
import json
import argparse
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, Field

from utils import setup_logging, load_config
from turbovec_retriever import KnowledgeRetriever

logger = setup_logging(name="knowledge_api")

# Configuration
config = load_config()
OLLAMA_BASE_URL = config.get("llm", {}).get("base_url", "http://localhost:11434")
EMBEDDING_MODEL = config.get("llm", {}).get("embedding_model", "mxbai-embed-large")
TURBOVEC_INDEX_PATH = os.getenv(
    "TURBOVEC_INDEX_PATH",
    str(Path(__file__).parent / "data" / "turbovec_index")
)

# Global retriever instance
retriever: Optional[KnowledgeRetriever] = None


# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results to return")
    category: Optional[str] = Field(default=None, description="Filter by category")


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    category: Optional[str]
    count: int


class StrategyResponse(BaseModel):
    strategies: List[Dict[str, Any]]
    filters: Dict[str, Any]
    count: int


class ConceptResponse(BaseModel):
    concept: Dict[str, Any]
    relationship_count: int


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    checks: Dict[str, Any]


app = FastAPI(
    title="Trading Knowledge Service (TurboVec)",
    description="Semantic search and retrieval for trading expertise using TurboVec",
    version="2.0.0",
)

# CORS middleware
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize TurboVec retriever on startup."""
    global retriever

    logger.info("Initializing TurboVec retriever...")
    retriever = KnowledgeRetriever(
        ollama_base_url=OLLAMA_BASE_URL,
        embedding_model=EMBEDDING_MODEL,
        index_path=TURBOVEC_INDEX_PATH,
    )

    logger.info(f"Loaded {len(retriever)} concepts from {TURBOVEC_INDEX_PATH}")

    # Initialize rate limiter
    try:
        from redis import Redis
        redis = Redis.from_url("redis://localhost")
        await FastAPILimiter.init(redis)
        logger.info("Rate limiter initialized (10 req/min)")
    except Exception as e:
        logger.warning(f"Rate limiter disabled: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with TurboVec and Ollama status."""
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "checks": {},
    }

    # Check TurboVec
    try:
        if retriever is not None:
            health["checks"]["turbovec"] = {
                "status": "ok",
                "documents": len(retriever),
                "index_path": TURBOVEC_INDEX_PATH,
            }
        else:
            health["checks"]["turbovec"] = {"status": "error", "error": "Not initialized"}
            health["status"] = "unhealthy"
    except Exception as e:
        health["checks"]["turbovec"] = {"status": "error", "error": str(e)}
        health["status"] = "unhealthy"

    # Check Ollama
    try:
        with urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) as response:
            if response.ok:
                health["checks"]["ollama"] = {
                    "status": "ok",
                    "url": OLLAMA_BASE_URL,
                    "model": EMBEDDING_MODEL,
                }
            else:
                health["checks"]["ollama"] = {"status": "error", "error": f"HTTP {response.status}"}
                health["status"] = "degraded"
    except Exception as e:
        health["checks"]["ollama"] = {"status": "error", "error": str(e)}
        health["status"] = "degraded"

    return health


@app.post("/knowledge/search", response_model=SearchResponse, dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def search_knowledge(request: SearchRequest):
    """
    Semantic search for trading concepts using TurboVec vector similarity.

    Args:
        query: Search query text
        limit: Maximum results (default: 10)
        category: Optional category filter

    Returns:
        List of matching concepts with similarity scores
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Knowledge retriever not initialized")

    logger.info(f"Searching for: {request.query[:50]}... (limit={request.limit}, category={request.category})")

    try:
        results = retriever.search(
            query=request.query,
            k=request.limit,
            category=request.category,
        )

        logger.info(f"Found {len(results)} results")
        return {"results": results, "query": request.query, "category": request.category, "count": len(results)}

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/knowledge", response_model=Dict[str, Any], dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def list_concepts(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    category: Optional[str] = Query(default=None),
):
    """
    List concepts with pagination.

    Args:
        limit: Items per page (default: 20)
        offset: Page offset (default: 0)
        category: Optional category filter

    Returns:
        Paginated list of concepts
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Knowledge retriever not initialized")

    try:
        # Get all documents and apply pagination
        all_docs = list(retriever.turbovec.documents.values())

        # Filter by category if specified
        if category:
            all_docs = [d for d in all_docs if d.metadata.get("category") == category]

        # Apply pagination
        paginated = all_docs[offset:offset + limit]

        concepts = [
            {
                "id": doc.id,
                "concept_name": doc.metadata.get("concept_name", ""),
                "category": doc.metadata.get("category", ""),
                "description": doc.content[:500],
                "source_book": doc.metadata.get("source_book", ""),
                "confidence": doc.metadata.get("confidence", "medium"),
            }
            for doc in paginated
        ]

        return {
            "concepts": concepts,
            "pagination": {"limit": limit, "offset": offset, "total": len(all_docs)},
        }

    except Exception as e:
        logger.error(f"List concepts error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list concepts: {str(e)}")


@app.get("/knowledge/{concept_id}", response_model=ConceptResponse)
async def get_concept(concept_id: str):
    """
    Get concept by ID.

    Args:
        concept_id: Concept ID

    Returns:
        Concept details
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Knowledge retriever not initialized")

    try:
        if concept_id not in retriever.turbovec.documents:
            raise HTTPException(status_code=404, detail="Concept not found")

        doc = retriever.turbovec.documents[concept_id]
        concept = {
            "id": concept_id,
            "concept_name": doc.metadata.get("concept_name", ""),
            "category": doc.metadata.get("category", ""),
            "description": doc.content,
            "interpretation": doc.metadata.get("interpretation", ""),
            "conditions": doc.metadata.get("conditions", ""),
            "market_context": doc.metadata.get("market_context", ""),
            "source_book": doc.metadata.get("source_book", ""),
            "confidence": doc.metadata.get("confidence", "medium"),
        }

        return {
            "concept": concept,
            "relationship_count": 0,  # Relationships not supported in TurboVec yet
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get concept error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get concept: {str(e)}")


@app.get("/strategies", response_model=StrategyResponse)
async def get_strategies(
    patterns: Optional[str] = Query(default=None, description="Comma-separated pattern names"),
    indicators: Optional[str] = Query(default=None, description="Comma-separated indicator names"),
    regime: Optional[str] = Query(default=None, description="Market regime filter"),
    asset_class: Optional[str] = Query(default=None, description="Asset class filter"),
):
    """
    Find trading strategies by conditions.

    Note: This endpoint requires a strategies table. For TurboVec-only mode,
    it returns an empty list until strategies are added.
    """
    logger.info(f"Looking up strategies: patterns={patterns}, indicators={indicators}")

    return {
        "strategies": [],
        "filters": {
            "patterns": patterns.split(",") if patterns else [],
            "indicators": indicators.split(",") if indicators else [],
            "regime": regime,
            "asset_class": asset_class,
        },
        "count": 0,
    }


@app.post("/knowledge/bulk-add", response_model=Dict[str, Any])
async def bulk_add_concepts(request: Body):
    """
    Add multiple concepts to the knowledge base.

    Request body should be a list of concept dicts with:
    - concept_name
    - description
    - category
    - conditions (optional)
    - market_context (optional)
    - source_book (optional)

    Returns:
        Number of concepts added
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Knowledge retriever not initialized")

    concepts = request if isinstance(request, list) else request.get("concepts", [])

    if not concepts:
        raise HTTPException(status_code=400, detail="No concepts provided")

    try:
        added = retriever.add_knowledge_base(concepts)
        retriever.save(TURBOVEC_INDEX_PATH)

        return {
            "added": added,
            "total": len(retriever),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Bulk add error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add concepts: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Knowledge Service API (TurboVec)")
    parser.add_argument("--port", type=int, default=3001, help="Port to run server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")

    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("Knowledge Service API (TurboVec)")
    logger.info("=" * 50)
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"TurboVec index: {TURBOVEC_INDEX_PATH}")
    logger.info(f"Ollama: {OLLAMA_BASE_URL} ({EMBEDDING_MODEL})")
    logger.info("")
    logger.info("Endpoints:")
    logger.info("  POST /knowledge/search     - Semantic search for concepts")
    logger.info("  GET  /knowledge            - List concepts with pagination")
    logger.info("  GET  /knowledge/:id        - Get concept by ID")
    logger.info("  GET  /strategies           - Find strategies (placeholder)")
    logger.info("  POST /knowledge/bulk-add   - Add concepts to index")
    logger.info("  GET  /health               - Health check with status")
    logger.info("")
    logger.info("Example:")
    logger.info(f"  curl -X POST http://localhost:{args.port}/knowledge/search \\")
    logger.info(f"    -H 'Content-Type: application/json' \\")
    logger.info(f"    -d '{{\"query\": \"bullish reversal patterns\", \"limit\": 5}}'")

    try:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    except ImportError:
        logger.error("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)


if __name__ == "__main__":
    main()
