"""Base researcher node for bull/bear debate participants.

Both bull and bear researchers follow the same pattern:
1. Extract debate state and reports from state
2. Build prompt with role-specific instructions (bull vs bear)
3. Invoke LLM
4. Update debate state with new argument
5. Return state update

This base class handles steps 1, 3, 4, 5. Subclasses provide 2.
"""

from typing import Dict


class BaseResearcher:
    """Base class for investment debate researchers."""

    # Override in subclass
    ROLE_NAME: str = ""  # e.g., "Bull Analyst"
    LABEL: str = ""  # e.g., "Bull"
    HISTORY_KEY: str = ""  # e.g., "bull_history"
    OPPONENT_LABEL: str = ""  # e.g., "bear" for Bull Analyst
    OPPONENT_HISTORY_KEY: str = ""  # e.g., "bear_history"
    PROMPT_TEMPLATE: str = ""

    def __init__(self, llm):
        self.llm = llm

    def _build_prompt(self, state: Dict) -> str:
        """Build the prompt with role-specific instructions."""
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        asset_type = state.get("asset_type", "stock")

        target_label = "stock" if asset_type == "stock" else "asset"
        fundamentals_label = (
            "Company fundamentals report"
            if asset_type == "stock"
            else "Asset fundamentals report (may be unavailable for crypto)"
        )

        opponent_label = self.OPPONENT_LABEL.capitalize()
        opponent_arg = f"Last {opponent_label} argument" if self.OPPONENT_LABEL else "Previous arguments"

        return self.PROMPT_TEMPLATE.format(
            target_label=target_label,
            fundamentals_label=fundamentals_label,
            market_research_report=market_research_report,
            sentiment_report=sentiment_report,
            news_report=news_report,
            fundamentals_report=fundamentals_report,
            history=history,
            opponent_arg=opponent_arg,
            current_response=current_response,
        )

    def _update_state(self, state: Dict, argument: str) -> Dict:
        """Update debate state with new argument."""
        investment_debate_state = state["investment_debate_state"]
        return {
            "history": investment_debate_state["history"] + "\n" + argument,
            f"{self.HISTORY_KEY}": investment_debate_state.get(self.HISTORY_KEY, "") + "\n" + argument,
            f"{self.OPPONENT_HISTORY_KEY}": investment_debate_state.get(self.OPPONENT_HISTORY_KEY, ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

    def __call__(self, state: Dict) -> Dict:
        """Execute the researcher node and return state update."""
        prompt = self._build_prompt(state)
        response = self.llm.invoke(prompt)
        argument = f"{self.ROLE_NAME}: {response.content}"
        return {"investment_debate_state": self._update_state(state, argument)}


class BullResearcher(BaseResearcher):
    ROLE_NAME = "Bull Analyst"
    LABEL = "Bull"
    HISTORY_KEY = "bull_history"
    OPPONENT_LABEL = "bear"
    OPPONENT_HISTORY_KEY = "bear_history"
    PROMPT_TEMPLATE = """You are a Bull Analyst advocating for investing in the {target_label}. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

Resources available:
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
{fundamentals_label}: {fundamentals_report}
Conversation history of the debate: {history}
{opponent_arg}: {current_response}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position."""

    def _build_prompt(self, state: Dict) -> str:
        prompt = super()._build_prompt(state)
        from quantify.agents.utils.agent_utils import get_language_instruction
        return prompt + get_language_instruction()


class BearResearcher(BaseResearcher):
    ROLE_NAME = "Bear Analyst"
    LABEL = "Bear"
    HISTORY_KEY = "bear_history"
    OPPONENT_LABEL = "bull"
    OPPONENT_HISTORY_KEY = "bull_history"
    PROMPT_TEMPLATE = """You are a Bear Analyst making the case against investing in the {target_label}. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators. Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.

Key points to focus on:

- Risks and Challenges: Highlight factors like market saturation, financial instability, or macroeconomic threats that could hinder the stock's performance.
- Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning, declining innovation, or threats from competitors.
- Negative Indicators: Use evidence from financial data, market trends, or recent adverse news to support your position.
- Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning, exposing weaknesses or over-optimistic assumptions.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points and debating effectively rather than simply listing facts.

Resources available:

Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
{fundamentals_label}: {fundamentals_report}
Conversation history of the debate: {history}
{opponent_arg}: {current_response}
Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate that demonstrates the risks and weaknesses of investing in the {target_label}."""

    def _build_prompt(self, state: Dict) -> str:
        prompt = super()._build_prompt(state)
        from quantify.agents.utils.agent_utils import get_language_instruction
        return prompt + get_language_instruction()


# Backwards compatibility
def create_bull_researcher(llm):
    """Create bull researcher node (backwards-compatible)."""
    return BullResearcher(llm)


def create_bear_researcher(llm):
    """Create bear researcher node (backwards-compatible)."""
    return BearResearcher(llm)
