#!/usr/bin/env python3
"""
Knowledge Service - FastAPI REST API for Trading Expertise Graph
Uses Ollama for local embeddings and PostgreSQL + pgvector for semantic search.

Endpoints:
    POST /knowledge/search - Semantic search for concepts
    GET  /knowledge/:id - Get concept by ID with relationships
    GET  /knowledge - List concepts with pagination
    GET  /strategies - Find strategies by conditions
    GET  /health - Health check with status

Usage:
    python scripts/knowledge_api.py              # Start server on port 3001
    python scripts/knowledge_api.py --port 3001  # Custom port
"""

import os
import sys
import json
import argparse
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from functools import lru_cache

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, Field
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from urllib.request import urlopen, Request

from utils import setup_logging, load_config

logger = setup_logging(name="knowledge_api")

# Connection pool (initialized in lifespan)
db_pool: Optional[pool.ThreadedConnectionPool] = None

# Embedding cache
embedding_cache: Dict[str, List[float]] = {}


# Configuration
config = load_config()
DATABASE_URL = os.getenv("DATABASE_URL") or config.get("database", {}).get(
    "url", "postgresql://localhost:5432/trading_knowledge"
)
OLLAMA_BASE_URL = config.get("llm", {}).get("base_url", "http://localhost:11434")
EMBEDDING_MODEL = config.get("llm", {}).get("embedding_model", "mxbai-embed-large")


def get_db_connection():
    """Get database connection from pool."""
    global db_pool
    if db_pool is None:
        raise RuntimeError("Database pool not initialized")
    return db_pool.getconn()


def release_db_connection(conn):
    """Return connection to pool."""
    global db_pool
    if db_pool is not None:
        db_pool.putconn(conn)


@lru_cache(maxsize=1024)
def get_embedding_cached(text_hash: str, text: str) -> List[float]:
    """Generate embedding with LRU cache."""
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {"model": EMBEDDING_MODEL, "prompt": text}

    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            embedding = result.get("embedding", [])
            if not embedding:
                raise ValueError("Empty embedding from Ollama")
            return embedding
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


def get_embedding(text: str) -> List[float]:
    """Generate embedding using Ollama with caching."""
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    return get_embedding_cached(text_hash, text)


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
    relationships: List[Dict[str, Any]]
    relationship_count: int


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    checks: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global db_pool
    logger.info("Knowledge Service starting up...")

    # Initialize database connection pool
    db_pool = pool.ThreadedConnectionPool(
        minconn=2, maxconn=10, dsn=DATABASE_URL
    )
    logger.info(f"Database pool initialized: {DATABASE_URL.split('@')[-1]}")

    # Initialize rate limiter (Redis-based)
    try:
        from redis import Redis
        redis = Redis.from_url("redis://localhost")
        await FastAPILimiter.init(redis)
        logger.info("Rate limiter initialized (10 req/min)")
    except Exception as e:
        logger.warning(f"Rate limiter disabled: {e}")

    yield

    # Cleanup
    if db_pool:
        db_pool.closeall()
        logger.info("Database connections closed")

    logger.info("Knowledge Service shutting down...")


app = FastAPI(
    title="Trading Knowledge Service",
    description="Semantic search and retrieval for trading expertise",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - restricted to known origins in production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with database and Ollama status."""
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "checks": {},
    }

    # Check database
    try:
        conn = get_db_connection()
        release_db_connection(conn)
        health["checks"]["database"] = {"status": "ok", "url": DATABASE_URL.split("@")[-1]}
    except Exception as e:
        health["checks"]["database"] = {"status": "error", "error": str(e)}
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
    Semantic search for trading concepts using vector similarity.

    Args:
        query: Search query text
        limit: Maximum results (default: 10)
        category: Optional category filter

    Returns:
        List of matching concepts with similarity scores
    """
    logger.info(f"Searching for: {request.query[:50]}... (limit={request.limit}, category={request.category})")

    try:
        query_embedding = get_embedding(request.query)

        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Build query with optional category filter
        category_filter = "AND category = %(category)s" if request.category else ""
        sql = f"""
            SELECT id, concept_name, category, description, interpretation, conditions,
                   market_context, timeframes, source_book, confidence,
                   1 - (embedding <=> %(embedding)s) as similarity
            FROM concepts
            WHERE 1=1 {category_filter}
            ORDER BY embedding <=> %(embedding)s
            LIMIT %(limit)s
        """

        params = {
            "embedding": query_embedding,
            "limit": request.limit,
        }
        if request.category:
            params["category"] = request.category

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        # Parse timeframes array
        results = []
        for row in rows:
            row_dict = dict(row)
            if isinstance(row_dict.get("timeframes"), str):
                try:
                    row_dict["timeframes"] = json.loads(row_dict["timeframes"])
                except json.JSONDecodeError:
                    row_dict["timeframes"] = [row_dict["timeframes"]]
            results.append(row_dict)

        cursor.close()
        release_db_connection(conn)

        logger.info(f"Found {len(results)} results")
        return {"results": results, "query": request.query, "category": request.category, "count": len(results)}

    except HTTPException:
        if conn:
            release_db_connection(conn)
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        if conn:
            release_db_connection(conn)
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
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        params = [limit, offset]
        where_clause = ""
        param_index = 3

        if category:
            where_clause = "WHERE category = $3"
            params.append(category)

        sql = f"""
            SELECT id, concept_name, category, description, interpretation,
                   source_book, confidence
            FROM concepts
            {where_clause}
            ORDER BY id
            LIMIT $1 OFFSET $2
        """

        cursor.execute(sql, params)
        concepts = [dict(row) for row in cursor.fetchall()]

        # Get total count
        count_sql = "SELECT COUNT(*) FROM concepts WHERE category = $1" if category else "SELECT COUNT(*) FROM concepts"
        cursor.execute(count_sql, [category] if category else [])
        total = cursor.fetchone()["count"]

        cursor.close()
        release_db_connection(conn)

        return {
            "concepts": concepts,
            "pagination": {"limit": limit, "offset": offset, "total": total},
        }

    except Exception as e:
        logger.error(f"List concepts error: {e}")
        if conn:
            release_db_connection(conn)
        raise HTTPException(status_code=500, detail=f"Failed to list concepts: {str(e)}")


@app.get("/knowledge/{concept_id}", response_model=ConceptResponse)
async def get_concept(concept_id: int):
    """
    Get concept by ID with related concepts.

    Args:
        concept_id: Concept ID

    Returns:
        Concept details with relationships
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Get concept
        cursor.execute("SELECT * FROM concepts WHERE id = %s", [concept_id])
        row = cursor.fetchone()

        if not row:
            cursor.close()
            release_db_connection(conn)
            raise HTTPException(status_code=404, detail="Concept not found")

        concept = dict(row)

        # Get relationships
        related_sql = """
            SELECT c2.*, r.relationship_type
            FROM relationships r
            JOIN concepts c1 ON r.from_concept_id = c1.id
            JOIN concepts c2 ON r.to_concept_id = c2.id
            WHERE c1.id = %s OR c2.id = %s
        """
        cursor.execute(related_sql, [concept_id])
        relationships = [dict(r) for r in cursor.fetchall()]

        cursor.close()
        release_db_connection(conn)

        # Parse timeframes
        if isinstance(concept.get("timeframes"), str):
            try:
                concept["timeframes"] = json.loads(concept["timeframes"])
            except json.JSONDecodeError:
                concept["timeframes"] = [concept["timeframes"]]

        return {
            "concept": concept,
            "relationships": relationships,
            "relationship_count": len(relationships),
        }

    except HTTPException:
        if conn:
            release_db_connection(conn)
        raise
    except Exception as e:
        logger.error(f"Get concept error: {e}")
        if conn:
            release_db_connection(conn)
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

    Args:
        patterns: Comma-separated pattern names
        indicators: Comma-separated indicator names
        regime: Market regime filter
        asset_class: Asset class filter

    Returns:
        List of matching strategies with used patterns and required indicators
    """
    conn = None
    try:
        pattern_array = [p.strip() for p in patterns.split(",") if p.strip()] if patterns else []
        indicator_array = [i.strip() for i in indicators.split(",") if i.strip()] if indicators else []

        logger.info(f"Looking up strategies: patterns={pattern_array}, indicators={indicator_array}")

        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        sql = """
            SELECT DISTINCT s.id, s.name, s.description, s.asset_class, s.market_regime,
                   s.entry_logic, s.exit_logic, s.stop_loss_rules, s.position_sizing_rules,
                   s.risk_reward_ratio,
                   array_agg(DISTINCT p.concept_name) FILTER (WHERE p.concept_name IS NOT NULL) as used_patterns,
                   array_agg(DISTINCT i.concept_name) FILTER (WHERE i.concept_name IS NOT NULL) as required_indicators
            FROM strategies s
            LEFT JOIN relationships r ON s.id = r.from_concept_id
            LEFT JOIN concepts p ON r.to_concept_id = p.id AND r.relationship_type = 'USES_PATTERN'
            LEFT JOIN concepts i ON r.to_concept_id = i.id AND r.relationship_type = 'REQUIRES_INDICATOR'
            WHERE 1=1
        """

        params = []
        param_index = 1

        if pattern_array:
            sql += f" AND p.concept_name = ANY(${param_index})"
            params.append(pattern_array)
            param_index += 1

        if indicator_array:
            sql += f" AND i.concept_name = ANY(${param_index})"
            params.append(indicator_array)
            param_index += 1

        if regime:
            sql += f" AND s.market_regime = ${param_index}"
            params.append(regime)
            param_index += 1

        if asset_class:
            sql += f" AND s.asset_class = ${param_index}"
            params.append(asset_class)
            param_index += 1

        sql += " GROUP BY s.id"
        cursor.execute(sql, params)
        strategies = [dict(s) for s in cursor.fetchall()]

        cursor.close()
        release_db_connection(conn)

        return {
            "strategies": strategies,
            "filters": {
                "patterns": pattern_array,
                "indicators": indicator_array,
                "regime": regime,
                "asset_class": asset_class,
            },
            "count": len(strategies),
        }

    except Exception as e:
        logger.error(f"Strategy lookup error: {e}")
        if conn:
            release_db_connection(conn)
        raise HTTPException(status_code=500, detail=f"Strategy lookup failed: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Knowledge Service API")
    parser.add_argument("--port", type=int, default=3001, help="Port to run server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")

    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("Knowledge Service API")
    logger.info("=" * 50)
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Database: {DATABASE_URL.split('@')[-1]}")
    logger.info(f"Ollama: {OLLAMA_BASE_URL} ({EMBEDDING_MODEL})")
    logger.info("")
    logger.info("Endpoints:")
    logger.info("  POST /knowledge/search     - Semantic search for concepts")
    logger.info("  GET  /knowledge            - List concepts with pagination")
    logger.info("  GET  /knowledge/:id        - Get concept by ID with relationships")
    logger.info("  GET  /strategies           - Find strategies by conditions")
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
