"""News analyst using base class."""
from quantify.agents.base_analyst import BaseAnalystNode
from quantify.agents.utils.agent_utils import (
    get_global_news,
    get_language_instruction,
    get_news,
)


class NewsAnalyst(BaseAnalystNode):
    """Analyst for news and macroeconomic trends."""

    TOOLS = [get_news, get_global_news]
    REPORT_KEY = "news_report"

    def _get_state_vars(self, state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        return current_date, asset_type, asset_label

    def _build_system_message(self, asset_label: str) -> str:
        return (
            f"You are a news researcher tasked with analyzing recent news and trends over the past week. "
            f"Please write a comprehensive report of the current state of the world that is relevant for trading "
            f"and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for {asset_label}-specific "
            f"or targeted news searches, and get_global_news(curr_date, look_back_days, limit) for broader "
            f"macroeconomic news. Provide specific, actionable insights with supporting evidence to help traders "
            f"make informed decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )

    def _invoke(self, state):
        current_date, asset_type, asset_label = self._get_state_vars(state)
        self.SYSTEM_MESSAGE = self._build_system_message(asset_label)
        return super()._invoke(state)


def create_news_analyst(llm):
    """Create news analyst node."""
    return NewsAnalyst(llm)
