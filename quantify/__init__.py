"""Quantify - AI-Powered Trading Analysis Platform.

This package provides:
- TradingAgents: Multi-agent trading analysis system
- Knowledge: Semantic search for trading expertise
- Dataflows: Real-time market data from multiple vendors
- Market Data: Yahoo Finance, Alpha Vantage, social sentiment
"""

__version__ = "1.0.0"
__author__ = "Quantify Team"

# Load .env files at package import
try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    load_dotenv(find_dotenv(".env.enterprise", usecwd=True), override=False)
except ImportError:
    pass

# Suppress langchain-core and langgraph-checkpoint deprecation warnings
import warnings
try:
    import langchain_core  # noqa: F401
except ImportError:
    pass

warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects`.*",
    category=PendingDeprecationWarning,
)
