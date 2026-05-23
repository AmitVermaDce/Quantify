"""
Yahoo Finance Data Provider
No API key required - uses public yfinance library.
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

try:
    import yfinance as yf
    from stockstats import StockStats
except ImportError:
    os.system("pip install yfinance stockstats")
    import yfinance as yf
    from stockstats import StockStats

from ..utils import setup_logging

logger = setup_logging(name="yfinance_data")


class YahooFinanceClient:
    """Yahoo Finance data client."""

    def __init__(self, cache_dir: str = None):
        """
        Initialize Yahoo Finance client.

        Args:
            cache_dir: Optional directory for caching OHLCV data
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _safe_ticker(self, symbol: str) -> str:
        """Sanitize ticker symbol for filenames."""
        return symbol.replace("/", "_").replace("\\", "_").replace(".", "_")

    def get_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> Optional[Dict[str, Any]]:
        """
        Get OHLCV stock data from Yahoo Finance.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, 5m, etc.)

        Returns:
            Dictionary with OHLCV data in CSV-like format
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return None

            # Format as list of dicts for JSON serialization
            data = []
            for idx, row in df.iterrows():
                data.append({
                    "date": idx.strftime("%Y-%m-%d"),
                    "open": float(row.get("Open", 0)),
                    "high": float(row.get("High", 0)),
                    "low": float(row.get("Low", 0)),
                    "close": float(row.get("Close", 0)),
                    "volume": int(row.get("Volume", 0)),
                })

            logger.debug(f"Fetched {len(data)} bars for {symbol}")
            return {"symbol": symbol, "data": data, "interval": interval}

        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return None

    def get_technical_indicators(
        self,
        symbol: str,
        indicators: List[str],
        end_date: str,
        lookback_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Calculate technical indicators using stockstats.

        Args:
            symbol: Stock ticker symbol
            indicators: List of indicators (rsi, macd, sma_50, etc.)
            end_date: End date (YYYY-MM-DD)
            lookback_days: Days of history for calculation

        Returns:
            Dictionary of indicator values
        """
        try:
            # Calculate start date for lookback
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=lookback_days * 2)

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_dt.strftime("%Y-%m-%d"), end=end_date)

            if df.empty:
                return {"error": "No data available"}

            # Convert to stockstats format
            stock = StockStats.retype(df)

            results = {"symbol": symbol, "indicators": {}}

            for indicator in indicators:
                try:
                    if indicator == "rsi":
                        results["indicators"]["rsi"] = float(stock["rsi_14"].iloc[-1])
                    elif indicator == "macd":
                        results["indicators"]["macd"] = float(stock["macd"].iloc[-1])
                        results["indicators"]["macds"] = float(stock["macds"].iloc[-1])
                        results["indicators"]["macdh"] = float(stock["macdh"].iloc[-1])
                    elif indicator == "sma_50":
                        results["indicators"]["sma_50"] = float(stock["close_50_sma"].iloc[-1])
                    elif indicator == "sma_200":
                        results["indicators"]["sma_200"] = float(stock["close_200_sma"].iloc[-1])
                    elif indicator == "ema_12":
                        results["indicators"]["ema_12"] = float(stock["close_12_ema"].iloc[-1])
                    elif indicator == "ema_26":
                        results["indicators"]["ema_26"] = float(stock["close_26_ema"].iloc[-1])
                    elif indicator == "bollinger_upper":
                        results["indicators"]["bollinger_upper"] = float(stock["boll_ub"].iloc[-1])
                    elif indicator == "bollinger_lower":
                        results["indicators"]["bollinger_lower"] = float(stock["boll_lb"].iloc[-1])
                    elif indicator == "atr":
                        results["indicators"]["atr"] = float(stock["atr"].iloc[-1])
                    elif indicator == "mfi":
                        results["indicators"]["mfi"] = float(stock["mfi_14"].iloc[-1])
                    else:
                        logger.warning(f"Unknown indicator: {indicator}")
                except (KeyError, IndexError, ValueError) as e:
                    logger.warning(f"Could not calculate {indicator}: {e}")

            return results

        except Exception as e:
            logger.error(f"Technical indicators error for {symbol}: {e}")
            return {"error": str(e)}

    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get fundamental data from Yahoo Finance.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with fundamental metrics
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                return None

            fundamentals = {
                "symbol": symbol,
                "company_name": info.get("longName", info.get("shortName", symbol)),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "ev_to_revenue": info.get("enterpriseToRevenue"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "return_on_equity": info.get("returnOnEquity"),
                "return_on_assets": info.get("returnOnAssets"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "revenue_ttm": info.get("totalRevenue"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "dividend_yield": info.get("dividendYield"),
                "payout_ratio": info.get("payoutRatio"),
                "beta": info.get("beta"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "50_day_ma": info.get("fiftyDayAverage"),
                "200_day_ma": info.get("twoHundredDayAverage"),
                "analyst_rating": info.get("recommendationKey"),
                "target_price": info.get("targetMeanPrice"),
            }

            # Filter out None values
            fundamentals = {k: v for k, v in fundamentals.items() if v is not None}

            return fundamentals

        except Exception as e:
            logger.error(f"Fundamentals error for {symbol}: {e}")
            return None

    def get_news(
        self,
        symbol: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get news articles from Yahoo Finance.

        Args:
            symbol: Optional ticker for company-specific news
            limit: Maximum articles to return

        Returns:
            List of news articles
        """
        try:
            if symbol:
                ticker = yf.Ticker(symbol)
                news = ticker.get_news()[:limit]
            else:
                # Global market news via search
                news = yf.Search(query="market news", news=True).get_news()[:limit]

            articles = []
            for item in news:
                articles.append({
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "link": item.get("link", ""),
                    "published_at": datetime.fromtimestamp(item.get("providerPublishTime", 0)).isoformat() if item.get("providerPublishTime") else None,
                    "summary": item.get("summary", ""),
                })

            return articles

        except Exception as e:
            logger.error(f"News error: {e}")
            return []
