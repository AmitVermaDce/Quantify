"""Outcome resolution and memory log management for trading decisions.

This module handles:
- Benchmark selection for alpha calculation
- Return fetching and alpha calculation
- Pending memory log entry resolution
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple, List, TYPE_CHECKING

import yfinance as yf

from quantify.agents.utils.memory import TradingMemoryLog

if TYPE_CHECKING:
    from .reflection import Reflector

logger = logging.getLogger(__name__)


class OutcomeResolver:
    """Resolves trading outcomes and manages memory log updates.

    This class encapsulates the logic for:
    - Selecting appropriate benchmarks based on ticker exchange
    - Fetching returns and calculating alpha
    - Resolving pending memory log entries with realized outcomes
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration.

        Args:
            config: Configuration dictionary containing benchmark_map and benchmark_ticker
        """
        self.config = config

    def resolve_benchmark(self, ticker: str) -> str:
        """Pick the benchmark ticker for alpha calculation.

        ``config["benchmark_ticker"]`` overrides everything when set; otherwise
        the suffix map matches the ticker's exchange suffix (e.g. ``.T`` for
        Tokyo). US-listed tickers without a dotted suffix fall through to the
        empty-suffix entry (SPY by default). Unrecognised suffixes (including
        US tickers with dots like ``BRK.B``) also fall back to the empty-suffix
        entry, which is the right default because the alpha calculation works
        in USD.

        Args:
            ticker: The ticker symbol to find a benchmark for

        Returns:
            Benchmark ticker symbol for alpha calculation
        """
        explicit = self.config.get("benchmark_ticker")
        if explicit:
            return explicit

        benchmark_map = self.config.get("benchmark_map", {})
        ticker_upper = ticker.upper()

        for suffix, benchmark in benchmark_map.items():
            if suffix and ticker_upper.endswith(suffix.upper()):
                return benchmark

        return benchmark_map.get("", "SPY")

    def fetch_returns(
        self,
        ticker: str,
        trade_date: str,
        holding_days: int = 5,
        benchmark: str | None = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        """Fetch raw and alpha return for ticker over holding_days from trade_date.

        Args:
            ticker: The ticker symbol to fetch returns for
            trade_date: The trade date in YYYY-MM-DD format
            holding_days: Number of days to hold the position (default: 5)
            benchmark: Benchmark ticker for alpha calculation (default: resolved automatically)

        Returns:
            Tuple of (raw_return, alpha_return, actual_holding_days) or
            (None, None, None) if price data is unavailable (too recent, delisted, or network error)
        """
        if benchmark is None:
            benchmark = self.resolve_benchmark(ticker)

        try:
            start = datetime.strptime(trade_date, "%Y-%m-%d")
            end = start + timedelta(days=holding_days + 7)  # buffer for weekends/holidays
            end_str = end.strftime("%Y-%m-%d")

            stock = yf.Ticker(ticker).history(start=trade_date, end=end_str)
            bench = yf.Ticker(benchmark).history(start=trade_date, end=end_str)

            if len(stock) < 2 or len(bench) < 2:
                return None, None, None

            actual_days = min(holding_days, len(stock) - 1, len(bench) - 1)
            raw = float(
                (stock["Close"].iloc[actual_days] - stock["Close"].iloc[0])
                / stock["Close"].iloc[0]
            )
            bench_ret = float(
                (bench["Close"].iloc[actual_days] - bench["Close"].iloc[0])
                / bench["Close"].iloc[0]
            )
            alpha = raw - bench_ret
            return raw, alpha, actual_days

        except Exception as e:
            logger.warning(
                "Could not resolve outcome for %s on %s vs %s (will retry next run): %s",
                ticker, trade_date, benchmark, e,
            )
            return None, None, None

    def resolve_pending_entries(
        self,
        ticker: str,
        memory_log: TradingMemoryLog,
        reflector: Reflector,
    ) -> None:
        """Resolve pending log entries for ticker at the start of a new run.

        Fetches returns for each same-ticker pending entry, generates reflections,
        then writes all updates in a single atomic batch write to avoid redundant I/O.
        Skips entries whose price data is not yet available (too recent or delisted).

        Trade-off: only same-ticker entries are resolved per run. Entries for
        other tickers accumulate until that ticker is run again.

        Args:
            ticker: The ticker symbol to resolve pending entries for
            memory_log: TradingMemoryLog instance for reading/writing entries
            reflector: Reflector instance for generating reflections
        """
        pending = [e for e in memory_log.get_pending_entries() if e["ticker"] == ticker]
        if not pending:
            return

        benchmark = self.resolve_benchmark(ticker)
        updates = []

        for entry in pending:
            raw, alpha, days = self.fetch_returns(
                ticker, entry["date"], benchmark=benchmark,
            )
            if raw is None:
                continue  # price not available yet — try again next run

            reflection = reflector.reflect_on_final_decision(
                final_decision=entry.get("decision", ""),
                raw_return=raw,
                alpha_return=alpha,
                benchmark_name=benchmark,
            )
            updates.append({
                "ticker": ticker,
                "trade_date": entry["date"],
                "raw_return": raw,
                "alpha_return": alpha,
                "holding_days": days,
                "reflection": reflection,
            })

        if updates:
            memory_log.batch_update_with_outcomes(updates)
