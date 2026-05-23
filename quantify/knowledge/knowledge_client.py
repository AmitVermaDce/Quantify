#!/usr/bin/env python3
"""
Knowledge Client - Query the Trading Knowledge Base Service
Provides RAG-style context injection for TradingAgents analysts.

Usage:
    from knowledge_client import query_knowledge, get_trading_context

    # Simple query
    results = query_knowledge("bullish reversal patterns", limit=5)

    # Get context for analyst prompt
    context = get_trading_context("NVDA", "2026-05-20", analyst_type="market")
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

from utils import setup_logging, load_config

logger = setup_logging(name="knowledge_client")

# Default configuration
config = load_config()
DEFAULT_KNOWLEDGE_SERVICE_URL = os.getenv("KNOWLEDGE_SERVICE_URL") or config.get(
    "knowledge_service", {}
).get("url", "http://localhost:3001")
DEFAULT_TIMEOUT = float(os.getenv("KNOWLEDGE_QUERY_TIMEOUT", "2.0"))


class KnowledgeClientError(Exception):
    """Custom exception for knowledge client errors."""

    pass


def query_knowledge(
    query: str,
    limit: int = 5,
    category: Optional[str] = None,
    service_url: str = None,
    timeout: float = None,
) -> List[Dict[str, Any]]:
    """
    Semantic search for trading concepts.

    Args:
        query: Search query text
        limit: Maximum results to return
        category: Optional category filter (Technical Pattern, Candlestick Pattern, etc.)
        service_url: Knowledge service URL (default: from config)
        timeout: Request timeout in seconds (default: 2.0)

    Returns:
        List of matching concepts with similarity scores

    Example:
        >>> results = query_knowledge("RSI bullish divergence")
        >>> for r in results:
        ...     print(f"{r['concept_name']}: {r['description']}")
    """
    service_url = service_url or DEFAULT_KNOWLEDGE_SERVICE_URL
    timeout = timeout or DEFAULT_TIMEOUT

    try:
        response = requests.post(
            f"{service_url}/knowledge/search",
            json={"query": query, "limit": limit, "category": category},
            timeout=timeout,
        )

        if response.status_code == 404:
            logger.warning("Knowledge service returned 404 - service may be down")
            return []

        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        logger.debug(f"Found {len(results)} concepts for query: {query[:30]}...")
        return results

    except requests.exceptions.Timeout:
        logger.warning(f"Knowledge query timed out after {timeout}s: {query[:30]}...")
        return []
    except requests.exceptions.ConnectionError:
        logger.warning(f"Could not connect to knowledge service at {service_url}")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Knowledge query failed: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from knowledge service: {e}")
        return []


def get_concept_by_id(concept_id: int, service_url: str = None) -> Optional[Dict[str, Any]]:
    """
    Get a specific concept by ID with its relationships.

    Args:
        concept_id: Concept ID
        service_url: Knowledge service URL

    Returns:
        Concept details with relationships or None if not found
    """
    service_url = service_url or DEFAULT_KNOWLEDGE_SERVICE_URL

    try:
        response = requests.get(
            f"{service_url}/knowledge/{concept_id}",
            timeout=DEFAULT_TIMEOUT,
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()
        data = response.json()
        return data.get("concept")

    except requests.exceptions.RequestException:
        return None


def get_strategies(
    patterns: List[str] = None,
    indicators: List[str] = None,
    regime: str = None,
    asset_class: str = None,
    service_url: str = None,
) -> List[Dict[str, Any]]:
    """
    Find trading strategies by conditions.

    Args:
        patterns: List of pattern names to match
        indicators: List of indicator names to match
        regime: Market regime filter
        asset_class: Asset class filter
        service_url: Knowledge service URL

    Returns:
        List of matching strategies
    """
    service_url = service_url or DEFAULT_KNOWLEDGE_SERVICE_URL

    params = {}
    if patterns:
        params["patterns"] = ",".join(patterns)
    if indicators:
        params["indicators"] = ",".join(indicators)
    if regime:
        params["regime"] = regime
    if asset_class:
        params["asset_class"] = asset_class

    try:
        response = requests.get(
            f"{service_url}/strategies",
            params=params,
            timeout=DEFAULT_TIMEOUT,
        )

        if response.status_code != 200:
            return []

        data = response.json()
        return data.get("strategies", [])

    except requests.exceptions.RequestException:
        return []


def format_knowledge_for_prompt(
    results: List[Dict[str, Any]],
    max_items: int = 5,
) -> str:
    """
    Format knowledge results as a prompt injection string.

    Args:
        results: List of concept results
        max_items: Maximum items to include

    Returns:
        Formatted string for LLM prompt injection
    """
    if not results:
        return ""

    lines = ["Relevant trading knowledge:"]
    for i, concept in enumerate(results[:max_items]):
        name = concept.get("concept_name", "Unknown")
        desc = concept.get("description", "")
        interpretation = concept.get("interpretation", "")
        confidence = concept.get("confidence", "")

        line = f"  {i + 1}. {name}: {desc}"
        if interpretation:
            line += f" (Signals: {interpretation})"
        if confidence:
            line += f" [Confidence: {confidence}]"

        lines.append(line)

    return "\n".join(lines)


def get_trading_context(
    ticker: str,
    analysis_date: str,
    analyst_type: str = "market",
    service_url: str = None,
) -> str:
    """
    Get relevant knowledge context for a specific analyst type.

    Args:
        ticker: Stock ticker symbol
        analysis_date: Analysis date (YYYY-MM-DD)
        analyst_type: Type of analyst ("market", "news", "fundamentals", "sentiment", "risk")
        service_url: Knowledge service URL

    Returns:
        Formatted knowledge context string for prompt injection
    """
    # Map analyst types to query templates and categories
    analyst_queries = {
        "market": {
            "query": f"technical patterns {ticker} price action chart analysis",
            "category": "Technical Pattern",
        },
        "news": {
            "query": "market sentiment news bias macroeconomic indicators",
            "category": "Market Bias",
        },
        "fundamentals": {
            "query": f"fundamental analysis valuation metrics {ticker} financial health",
            "category": "Fundamental Metric",
        },
        "sentiment": {
            "query": "market sentiment investor psychology crowd behavior",
            "category": "Market Bias",
        },
        "risk": {
            "query": "risk management position sizing stop loss rules",
            "category": "Risk Rule",
        },
        "technical": {
            "query": f"technical indicators {ticker} RSI MACD moving averages",
            "category": "Indicator",
        },
    }

    query_config = analyst_queries.get(analyst_type, analyst_queries["market"])

    results = query_knowledge(
        query=query_config["query"],
        limit=5,
        category=query_config.get("category"),
        service_url=service_url,
    )

    return format_knowledge_for_prompt(results)


def check_service_health(service_url: str = None) -> Dict[str, Any]:
    """
    Check knowledge service health status.

    Args:
        service_url: Knowledge service URL

    Returns:
        Health status dictionary
    """
    service_url = service_url or DEFAULT_KNOWLEDGE_SERVICE_URL

    try:
        response = requests.get(f"{service_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "unhealthy", "error": str(e), "checks": {}}


# Convenience functions for common queries
def get_candlestick_patterns(service_url: str = None) -> List[Dict[str, Any]]:
    """Get all candlestick patterns from the knowledge base."""
    return query_knowledge(
        "candlestick patterns reversal continuation",
        category="Candlestick Pattern",
        limit=20,
        service_url=service_url,
    )


def get_technical_indicators(service_url: str = None) -> List[Dict[str, Any]]:
    """Get all technical indicators from the knowledge base."""
    return query_knowledge(
        "technical indicators RSI MACD moving averages",
        category="Indicator",
        limit=20,
        service_url=service_url,
    )


def get_risk_rules(service_url: str = None) -> List[Dict[str, Any]]:
    """Get risk management rules from the knowledge base."""
    return query_knowledge(
        "risk management position sizing stop loss",
        category="Risk Rule",
        limit=10,
        service_url=service_url,
    )


def get_market_regimes(service_url: str = None) -> List[Dict[str, Any]]:
    """Get market regime definitions from the knowledge base."""
    return query_knowledge(
        "market regimes bull bear trending ranging",
        category="Market Bias",
        limit=10,
        service_url=service_url,
    )


if __name__ == "__main__":
    # Demo/test the knowledge client
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Client Demo")
    parser.add_argument("--query", type=str, default="bullish reversal patterns")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--service-url", type=str, default=None)

    args = parser.parse_args()

    print(f"Querying knowledge base: '{args.query}'")
    print(f"Service: {args.service_url or DEFAULT_KNOWLEDGE_SERVICE_URL}")
    print("")

    results = query_knowledge(args.query, limit=args.limit, service_url=args.service_url)

    if results:
        print(f"Found {len(results)} concepts:\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r.get('concept_name')} ({r.get('category')})")
            print(f"   {r.get('description')}")
            print(f"   Signals: {r.get('interpretation')}")
            print(f"   Confidence: {r.get('confidence')}")
            print("")
    else:
        print("No results found. Is the knowledge service running?")

    # Check health
    health = check_service_health(args.service_url)
    print(f"\nService Health: {health.get('status', 'unknown')}")
    if health.get("checks"):
        for check, status in health["checks"].items():
            print(f"  {check}: {status.get('status', 'unknown')}")
