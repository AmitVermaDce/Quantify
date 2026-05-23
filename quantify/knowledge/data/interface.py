"""
Vendor abstraction layer for market data providers.
Routes requests to configured vendor with fallback support.
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from .yfinance_data import YahooFinanceClient
from .alpha_vantage import AlphaVantageClient
from .sentiment import RedditSentiment, StockTwitsSentiment


class DataProvider:
    """Unified interface for market data access."""

    def __init__(self, vendor: str = "yfinance", config: Optional[Dict] = None):
        """
        Initialize data provider.

        Args:
            vendor: Primary vendor ("yfinance" or "alpha_vantage")
            config: Optional configuration dict
        """
        self.vendor = vendor
        self.config = config or {}
        self._yfinance = YahooFinanceClient()
        self._alpha_vantage = AlphaVantageClient(
            api_key=self.config.get("alpha_vantage_api_key")
            or os.getenv("ALPHA_VANTAGE_API_KEY")
        )
        self._reddit = RedditSentiment()
        self._stocktwits = StockTwitsSentiment()

    def get_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> Optional[Dict[str, Any]]:
        """
        Get OHLCV stock data.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, 5m, etc.)

        Returns:
            Dictionary with OHLCV data or None on failure
        """
        if self.vendor == "alpha_vantage":
            try:
                return self._alpha_vantage.get_stock(symbol, start_date, end_date)
            except Exception:
                return self._yfinance.get_stock_data(symbol, start_date, end_date)
        return self._yfinance.get_stock_data(symbol, start_date, end_date)

    def get_technical_indicators(
        self,
        symbol: str,
        indicators: List[str],
        end_date: str,
        lookback_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get technical indicators.

        Args:
            symbol: Stock ticker symbol
            indicators: List of indicators (rsi, macd, sma_50, etc.)
            end_date: End date (YYYY-MM-DD)
            lookback_days: Days of history to calculate

        Returns:
            Dictionary of indicator values
        """
        if self.vendor == "alpha_vantage":
            try:
                return self._alpha_vantage.get_indicators(
                    symbol, indicators, end_date, lookback_days
                )
            except Exception:
                return self._yfinance.get_technical_indicators(
                    symbol, indicators, end_date, lookback_days
                )
        return self._yfinance.get_technical_indicators(
            symbol, indicators, end_date, lookback_days
        )

    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get fundamental data (financials, ratios, etc.).

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with fundamental data
        """
        if self.vendor == "alpha_vantage":
            try:
                return self._alpha_vantage.get_fundamentals(symbol)
            except Exception:
                return self._yfinance.get_fundamentals(symbol)
        return self._yfinance.get_fundamentals(symbol)

    def get_news(
        self,
        symbol: Optional[str] = None,
        limit: int = 20,
        lookback_days: int = 7,
    ) -> List[Dict[str, Any]]:
        """
        Get news articles.

        Args:
            symbol: Optional ticker for company-specific news
            limit: Maximum articles to return
            lookback_days: Days to look back

        Returns:
            List of news articles
        """
        if self.vendor == "alpha_vantage" and symbol:
            try:
                return self._alpha_vantage.get_news(symbol, limit)
            except Exception:
                pass
        return self._yfinance.get_news(symbol, limit)

    def get_sentiment(
        self,
        symbol: str,
        sources: List[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Get social media sentiment.

        Args:
            symbol: Stock ticker symbol
            sources: List of sources ("reddit", "stocktwits")
            limit: Maximum posts to fetch

        Returns:
            Sentiment summary by source
        """
        sources = sources or ["reddit", "stocktwits"]
        sentiment = {}

        if "reddit" in sources:
            sentiment["reddit"] = self._reddit.fetch_posts(symbol, limit)

        if "stocktwits" in sources:
            sentiment["stocktwits"] = self._stocktwits.fetch_messages(symbol, limit)

        return sentiment


def route_to_vendor(
    category: str,
    vendor: str = "yfinance",
    fallback: bool = True,
) -> callable:
    """
    Route data request to appropriate vendor.

    Args:
        category: Data category ("core_stock_apis", "technical_indicators", etc.)
        vendor: Preferred vendor
        fallback: Whether to fallback on failure

    Returns:
        Function to call for data retrieval
    """
    vendors = {
        "yfinance": YahooFinanceClient,
        "alpha_vantage": AlphaVantageClient,
    }

    if vendor not in vendors:
        vendor = "yfinance"

    return vendors[vendor]
