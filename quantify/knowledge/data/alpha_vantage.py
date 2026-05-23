"""
Alpha Vantage Data Provider
Requires API key - set ALPHA_VANTAGE_API_KEY environment variable.
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from ..utils import setup_logging

logger = setup_logging(name="alpha_vantage")


class AlphaVantageClient:
    """Alpha Vantage API client."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str = None):
        """
        Initialize Alpha Vantage client.

        Args:
            api_key: Alpha Vantage API key (or set ALPHA_VANTAGE_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            logger.warning(
                "Alpha Vantage API key not set. Set ALPHA_VANTAGE_API_KEY env var."
            )

    def _make_request(self, function: str, params: Dict = None) -> Optional[Dict]:
        """
        Make API request with rate limit handling.

        Args:
            function: API function name
            params: Additional query parameters

        Returns:
            JSON response or None on failure
        """
        if not self.api_key:
            raise ValueError("Alpha Vantage API key required")

        query_params = {"function": function, "apikey": self.api_key}
        if params:
            query_params.update(params)

        url = f"{self.BASE_URL}&{'&'.join(f'{k}={v}' for k, v in query_params.items())}"

        try:
            with urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

                # Check for rate limit
                if "Note" in data:
                    logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                    raise Exception("Rate limit exceeded")

                # Check for error message
                if "Error Message" in data:
                    logger.error(f"Alpha Vantage error: {data['Error Message']}")
                    return None

                return data

        except HTTPError as e:
            logger.error(f"Alpha Vantage HTTP error: {e.code} {e.reason}")
            return None
        except URLError as e:
            logger.error(f"Alpha Vantage URL error: {e.reason}")
            return None

    def get_stock(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get daily stock data.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary with OHLCV data
        """
        try:
            # Calculate if we need full outputsize
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            days = (end_dt - start_dt).days

            data = self._make_request(
                "TIME_SERIES_DAILY_ADJUSTED",
                {
                    "symbol": symbol,
                    "outputsize": "full" if days > 100 else "compact",
                    "datatype": "json",
                },
            )

            if not data or "Time Series (Daily)" not in data:
                return None

            ts = data["Time Series (Daily)"]
            results = []

            for date_str, values in ts.items():
                if start_date <= date_str <= end_date:
                    results.append({
                        "date": date_str,
                        "open": float(values.get("1. open", 0)),
                        "high": float(values.get("2. high", 0)),
                        "low": float(values.get("3. low", 0)),
                        "close": float(values.get("4. close", 0)),
                        "adjusted_close": float(values.get("5. adjusted close", 0)),
                        "volume": int(values.get("6. volume", 0)),
                        "dividend_amount": float(values.get("7. dividend amount", 0)),
                        "split_coefficient": float(values.get("8. split coefficient", 1)),
                    })

            results.sort(key=lambda x: x["date"])
            return {"symbol": symbol, "data": results, "source": "alpha_vantage"}

        except Exception as e:
            logger.error(f"Alpha Vantage stock data error for {symbol}: {e}")
            return None

    def get_indicators(
        self,
        symbol: str,
        indicators: List[str],
        end_date: str,
        lookback_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get technical indicators from Alpha Vantage.

        Args:
            symbol: Stock ticker symbol
            indicators: List of indicators
            end_date: End date (YYYY-MM-DD)
            lookback_days: Days of history

        Returns:
            Dictionary of indicator values
        """
        indicator_map = {
            "sma_50": ("SMA", {"time_period": 50}),
            "sma_200": ("SMA", {"time_period": 200}),
            "ema_12": ("EMA", {"time_period": 12}),
            "ema_26": ("EMA", {"time_period": 26}),
            "rsi": ("RSI", {"time_period": 14}),
            "macd": ("MACD", {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}),
            "bollinger": ("BBANDS", {"time_period": 20}),
            "atr": ("ATR", {"time_period": 14}),
        }

        results = {"symbol": symbol, "indicators": {}, "source": "alpha_vantage"}

        for indicator in indicators:
            if indicator not in indicator_map:
                logger.warning(f"Unknown indicator: {indicator}")
                continue

            func_name, params = indicator_map[indicator]
            params["symbol"] = symbol
            params["datatype"] = "json"

            data = self._make_request(func_name, params)
            if not data:
                continue

            # Extract latest value
            key = f"Technical Analysis: {func_name}"
            if key in data:
                latest = list(data[key].values())[0] if data[key] else {}

                if indicator == "macd":
                    results["indicators"]["macd"] = float(latest.get("MACD", 0))
                    results["indicators"]["macd_signal"] = float(latest.get("Signal", 0))
                    results["indicators"]["macd_hist"] = float(latest.get("MACD_Hist", 0))
                elif indicator == "bollinger":
                    results["indicators"]["bollinger_upper"] = float(latest.get("Real Upper Band", 0))
                    results["indicators"]["bollinger_middle"] = float(latest.get("Real Middle Band", 0))
                    results["indicators"]["bollinger_lower"] = float(latest.get("Real Lower Band", 0))
                else:
                    # Get first numeric value
                    for v in latest.values():
                        try:
                            results["indicators"][indicator] = float(v)
                            break
                        except (ValueError, TypeError):
                            continue

        return results

    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get fundamental data from Alpha Vantage.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with fundamental metrics
        """
        try:
            # Company overview
            overview = self._make_request("OVERVIEW", {"symbol": symbol})
            if not overview:
                return None

            fundamentals = {
                "symbol": symbol,
                "company_name": overview.get("Name"),
                "description": overview.get("Description"),
                "sector": overview.get("Sector"),
                "industry": overview.get("Industry"),
                "market_cap": overview.get("MarketCapitalization"),
                "pe_ratio": overview.get("PERatio"),
                "forward_pe": overview.get("ForwardPE"),
                "peg_ratio": overview.get("PEGRatio"),
                "price_to_book": overview.get("PriceToBook"),
                "price_to_sales": overview.get("PriceToSalesTrailing12Months"),
                "ev_to_revenue": overview.get("EVToRevenue"),
                "ev_to_ebitda": overview.get("EVToEbitda"),
                "profit_margin": overview.get("ProfitMargin"),
                "operating_margin": overview.get("OperatingMarginTTM"),
                "return_on_equity": overview.get("ReturnOnEquityTTM"),
                "return_on_assets": overview.get("ReturnOnAssetsTTM"),
                "debt_to_equity": overview.get("DebtToEquity"),
                "current_ratio": overview.get("CurrentRatio"),
                "quick_ratio": overview.get("QuickRatio"),
                "revenue_ttm": overview.get("RevenueTTM"),
                "gross_profit_ttm": overview.get("GrossProfitTTM"),
                "ebitda": overview.get("EBITDA"),
                "diluted_eps_ttm": overview.get("DilutedEPSTTM"),
                "dividend_yield": overview.get("DividendYield"),
                "dividend_per_share": overview.get("DividendPerShare"),
                "beta": overview.get("Beta"),
                "52_week_high": overview.get("52WeekHigh"),
                "52_week_low": overview.get("52WeekLow"),
                "50_day_ma": overview.get("50DayMovingAverage"),
                "200_day_ma": overview.get("200DayMovingAverage"),
                "analyst_target_price": overview.get("AnalystTargetPrice"),
                "source": "alpha_vantage",
            }

            # Filter out None values
            fundamentals = {k: v for k, v in fundamentals.items() if v is not None}

            return fundamentals

        except Exception as e:
            logger.error(f"Alpha Vantage fundamentals error for {symbol}: {e}")
            return None

    def get_news(
        self,
        symbol: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get news and sentiment from Alpha Vantage.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum articles to return

        Returns:
            List of news articles
        """
        try:
            data = self._make_request(
                "NEWS_SENTIMENT",
                {
                    "tickers": symbol,
                    "limit": limit,
                    "sort": "LATEST",
                },
            )

            if not data or "feed" not in data:
                return []

            articles = []
            for item in data["feed"][:limit]:
                articles.append({
                    "title": item.get("title", ""),
                    "publisher": item.get("news_source", ""),
                    "link": item.get("url", ""),
                    "published_at": item.get("time_published"),
                    "summary": item.get("summary", ""),
                    "sentiment_score": item.get("overall_sentiment_score"),
                    "sentiment_label": item.get("overall_sentiment_label"),
                })

            return articles

        except Exception as e:
            logger.error(f"Alpha Vantage news error for {symbol}: {e}")
            return []

    def get_insider_transactions(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get insider trading transactions.

        Args:
            symbol: Stock ticker symbol

        Returns:
            List of insider transactions
        """
        try:
            data = self._make_request("INSIDER_TRANSACTIONS", {"symbol": symbol})

            if not data or "data" not in data:
                return None

            transactions = []
            for item in data.get("data", [])[:20]:
                transactions.append({
                    "filing_date": item.get("filing_date"),
                    "transaction_date": item.get("transaction_date"),
                    "insider_name": item.get("name"),
                    "title": item.get("title"),
                    "transaction_type": item.get("transaction_type"),
                    "shares": item.get("shares"),
                    "price_per_share": item.get("value_per_share"),
                    "total_value": item.get("value"),
                    "shares_owned_after": item.get("shares_owned_following_transaction"),
                })

            return transactions

        except Exception as e:
            logger.error(f"Alpha Vantage insider transactions error: {e}")
            return None
