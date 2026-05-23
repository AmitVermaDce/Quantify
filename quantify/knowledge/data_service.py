#!/usr/bin/env python3
"""
Market Data Service - REST API for real-time and historical market data.
Provides unified access to Yahoo Finance, Alpha Vantage, and social sentiment.

Usage:
    python scripts/data_service.py              # Start server on port 3002
    python scripts/data_service.py --port 3002  # Custom port
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import setup_logging, load_config
from data.interface import DataProvider
from flask import Flask, request, jsonify, g

logger = setup_logging(name="data_service")
app = Flask(__name__)


def get_data_provider() -> DataProvider:
    """
    Get data provider instance (thread-safe via Flask's g object).

    Returns:
        DataProvider instance for current request context
    """
    if "data_provider" not in g:
        config = load_config()
        vendor = os.getenv("DATA_VENDOR", "yfinance")
        g.data_provider = DataProvider(vendor=vendor, config=config)
    return g.data_provider


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "market-data",
    })


@app.route("/stock/<symbol>", methods=["GET"])
def get_stock_data(symbol: str):
    """
    Get OHLCV stock data.

    Query params:
        start_date: Start date (YYYY-MM-DD), default: 30 days ago
        end_date: End date (YYYY-MM-DD), default: today
        interval: Data interval (1d, 1h, 5m), default: 1d
    """
    try:
        end_date = request.args.get("end_date", datetime.now().strftime("%Y-%m-%d"))
        start_date = request.args.get("start_date", (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))
        interval = request.args.get("interval", "1d")

        data = get_data_provider().get_stock_data(symbol, start_date, end_date, interval)

        if not data:
            return jsonify({"error": f"No data found for {symbol}"}), 404

        return jsonify(data)

    except Exception as e:
        logger.error(f"Stock data error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/indicators/<symbol>", methods=["GET"])
def get_indicators(symbol: str):
    """
    Get technical indicators.

    Query params:
        indicators: Comma-separated list (rsi,macd,sma_50,sma_200,bollinger,atr)
        lookback_days: Days of history, default: 30
        end_date: End date, default: today
    """
    try:
        indicators_param = request.args.get("indicators", "rsi,macd")
        indicators = [i.strip() for i in indicators_param.split(",")]
        lookback_days = int(request.args.get("lookback_days", 30))
        end_date = request.args.get("end_date", datetime.now().strftime("%Y-%m-%d"))

        data = get_data_provider().get_technical_indicators(
            symbol, indicators, end_date, lookback_days
        )

        return jsonify(data)

    except Exception as e:
        logger.error(f"Indicators error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/fundamentals/<symbol>", methods=["GET"])
def get_fundamentals(symbol: str):
    """Get fundamental data (financials, ratios, metrics)."""
    try:
        data = get_data_provider().get_fundamentals(symbol)

        if not data:
            return jsonify({"error": f"No fundamentals found for {symbol}"}), 404

        return jsonify(data)

    except Exception as e:
        logger.error(f"Fundamentals error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/news", methods=["GET"])
def get_news():
    """
    Get news articles.

    Query params:
        symbol: Optional ticker for company-specific news
        limit: Maximum articles, default: 20
    """
    try:
        symbol = request.args.get("symbol")
        limit = int(request.args.get("limit", 20))

        articles = get_data_provider().get_news(symbol, limit)
        return jsonify({"symbol": symbol, "articles": articles, "count": len(articles)})

    except Exception as e:
        logger.error(f"News error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/sentiment/<symbol>", methods=["GET"])
def get_sentiment(symbol: str):
    """
    Get social media sentiment.

    Query params:
        sources: Comma-separated (reddit,stocktwits), default: both
        limit: Maximum posts per source, default: 50
    """
    try:
        sources_param = request.args.get("sources", "reddit,stocktwits")
        sources = [s.strip() for s in sources_param.split(",")]
        limit = int(request.args.get("limit", 50))

        sentiment = get_data_provider().get_sentiment(symbol, sources, limit)
        return jsonify(sentiment)

    except Exception as e:
        logger.error(f"Sentiment error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/symbol/<symbol>/full", methods=["GET"])
def get_full_analysis(symbol: str):
    """
    Get complete analysis package: stock data, indicators, fundamentals, news, sentiment.

    Query params:
        lookback_days: Days for price history, default: 60
    """
    try:
        lookback_days = int(request.args.get("lookback_days", 60))
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        provider = get_data_provider()

        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "stock_data": provider.get_stock_data(symbol, start_date, end_date),
            "indicators": provider.get_technical_indicators(
                symbol, ["rsi", "macd", "sma_50", "sma_200", "bollinger"], end_date, lookback_days
            ),
            "fundamentals": provider.get_fundamentals(symbol),
            "news": provider.get_news(symbol, 10),
            "sentiment": provider.get_sentiment(symbol, ["reddit", "stocktwits"], 30),
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Full analysis error: {e}")
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="Market Data Service")
    parser.add_argument("--port", type=int, default=3002, help="Port to run server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("Market Data Service")
    logger.info("=" * 50)
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info("")
    logger.info("Endpoints:")
    logger.info(f"  GET  /health                    - Health check")
    logger.info(f"  GET  /stock/<symbol>            - OHLCV data")
    logger.info(f"  GET  /indicators/<symbol>       - Technical indicators")
    logger.info(f"  GET  /fundamentals/<symbol>     - Fundamental data")
    logger.info(f"  GET  /news                      - News articles")
    logger.info(f"  GET  /sentiment/<symbol>        - Social sentiment")
    logger.info(f"  GET  /symbol/<symbol>/full      - Complete analysis")
    logger.info("")
    logger.info("Example:")
    logger.info(f"  curl http://localhost:{args.port}/stock/AAPL?start_date=2026-01-01")
    logger.info(f"  curl http://localhost:{args.port}/indicators/NVDA?indicators=rsi,macd")
    logger.info(f"  curl http://localhost:{args.port}/sentiment/TSLA?sources=reddit,stocktwits")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
