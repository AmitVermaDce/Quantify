"""
Market Data Layer for Knowledge Base
Provides real-time and historical market data from multiple vendors.
"""

from .interface import DataProvider, route_to_vendor
from .yfinance_data import YahooFinanceClient
from .alpha_vantage import AlphaVantageClient
from .sentiment import RedditSentiment, StockTwitsSentiment

__all__ = [
    "DataProvider",
    "route_to_vendor",
    "YahooFinanceClient",
    "AlphaVantageClient",
    "RedditSentiment",
    "StockTwitsSentiment",
]
