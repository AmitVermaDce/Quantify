"""
Social Media Sentiment Data Providers
Fetches sentiment data from Reddit and StockTwits.
"""

import os
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from ..utils import setup_logging

logger = setup_logging(name="sentiment")


class RedditSentiment:
    """Reddit sentiment data provider."""

    SUBREDDITS = ["wallstreetbets", "stocks", "investing", "SecurityAnalysis"]

    def fetch_posts(
        self,
        ticker: str,
        limit: int = 50,
        subreddits: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch Reddit posts for a ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum posts per subreddit
            subreddits: List of subreddits to search

        Returns:
            Sentiment summary and posts
        """
        subreddits = subreddits or self.SUBREDDITS
        all_posts = []

        for subreddit in subreddits:
            try:
                posts = self._search_subreddit(ticker, subreddit, limit)
                all_posts.extend(posts)
            except Exception as e:
                logger.warning(f"Reddit error for r/{subreddit}: {e}")

        # Analyze sentiment
        sentiment = self._analyze_sentiment(all_posts, ticker)

        return {
            "ticker": ticker,
            "source": "reddit",
            "post_count": len(all_posts),
            "sentiment_score": sentiment["score"],
            "sentiment_label": sentiment["label"],
            "bullish_count": sentiment["bullish"],
            "bearish_count": sentiment["bearish"],
            "neutral_count": sentiment["neutral"],
            "posts": all_posts[:20],  # Return top 20 posts
        }

    def _search_subreddit(
        self,
        ticker: str,
        subreddit: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search subreddit for ticker mentions."""
        url = f"https://www.reddit.com/r/{subreddit}/search.json?q={ticker}&sort=new&limit={limit}&include_over_18=on"

        headers = {"User-Agent": "TradingAgents/1.0 (Market Data Aggregator)"}
        req = Request(url, headers=headers)

        try:
            with urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))

            posts = []
            for child in data.get("data", {}).get("children", []):
                post_data = child.get("data", {})
                posts.append({
                    "title": post_data.get("title", ""),
                    "author": post_data.get("author", "deleted"),
                    "score": post_data.get("score", 0),
                    "upvote_ratio": post_data.get("upvote_ratio", 0),
                    "num_comments": post_data.get("num_comments", 0),
                    "created_utc": datetime.fromtimestamp(post_data.get("created_utc", 0)).isoformat() if post_data.get("created_utc") else None,
                    "url": f"https://reddit.com{post_data.get('permalink', '')}",
                    "subreddit": subreddit,
                    "selftext": post_data.get("selftext", "")[:500],  # Truncate
                })

            return posts

        except (HTTPError, URLError) as e:
            logger.debug(f"Reddit API error: {e}")
            return []

    def _analyze_sentiment(
        self,
        posts: List[Dict],
        ticker: str,
    ) -> Dict[str, Any]:
        """
        Analyze sentiment from posts.

        Simple keyword-based sentiment:
        - Bullish: moon, buy, bullish, long, call, rocket, gem
        - Bearish: crash, bearish, short, put, dump, crash, crash
        """
        bullish_keywords = [
            "moon", "buy", "bullish", "long", "call", "rocket", "gem",
            "tendie", "gain", "green", "up", "rip", "hold", "accumulating",
        ]
        bearish_keywords = [
            "crash", "bearish", "short", "put", "dump", "sell", "red",
            "down", "loss", "bag", "tendie down", "margin call", "liquidate",
        ]

        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        for post in posts:
            text = f"{post.get('title', '')} {post.get('selftext', '')}".lower()

            bullish_score = sum(1 for kw in bullish_keywords if kw in text)
            bearish_score = sum(1 for kw in bearish_keywords if kw in text)

            if bullish_score > bearish_score:
                bullish_count += 1
            elif bearish_score > bullish_score:
                bearish_count += 1
            else:
                neutral_count += 1

        total = len(posts) or 1
        score = (bullish_count - bearish_count) / total

        if score > 0.2:
            label = "bullish"
        elif score < -0.2:
            label = "bearish"
        else:
            label = "neutral"

        return {
            "score": round(score, 3),
            "label": label,
            "bullish": bullish_count,
            "bearish": bearish_count,
            "neutral": neutral_count,
        }


class StockTwitsSentiment:
    """StockTwits sentiment data provider."""

    API_URL = "https://api.stocktwits.com/api/2/streams/symbol"

    def fetch_messages(
        self,
        ticker: str,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Fetch StockTwits messages for a ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum messages to fetch

        Returns:
            Sentiment summary and messages
        """
        try:
            url = f"{self.API_URL}/{ticker}.json?limit={limit}&callback="
            headers = {"User-Agent": "TradingAgents/1.0"}
            req = Request(url, headers=headers)

            with urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))

            messages = []
            bullish_count = 0
            bearish_count = 0

            for msg in data.get("messages", []):
                sentiment = msg.get("entities", {}).get("sentiment", {})
                sentiment_label = sentiment.get("basic", "").lower()

                if sentiment_label == "bullish":
                    bullish_count += 1
                elif sentiment_label == "bearish":
                    bearish_count += 1

                messages.append({
                    "body": msg.get("body", ""),
                    "username": msg.get("user", {}).get("username", ""),
                    "sentiment": sentiment_label,
                    "likes": msg.get("likes", {}).get("total", 0),
                    "created_at": msg.get("created_at", ""),
                    "symbols": [s.get("symbol") for s in msg.get("symbols", [])],
                })

            total = len(messages) or 1
            score = (bullish_count - bearish_count) / total

            if score > 0.2:
                label = "bullish"
            elif score < -0.2:
                label = "bearish"
            else:
                label = "neutral"

            return {
                "ticker": ticker,
                "source": "stocktwits",
                "message_count": len(messages),
                "sentiment_score": round(score, 3),
                "sentiment_label": label,
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "messages": messages[:20],  # Return top 20
            }

        except (HTTPError, URLError) as e:
            logger.error(f"StockTwits error for {ticker}: {e}")
            return {
                "ticker": ticker,
                "source": "stocktwits",
                "error": str(e),
                "sentiment_score": 0,
                "sentiment_label": "unknown",
            }
