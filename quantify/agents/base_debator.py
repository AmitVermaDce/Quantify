"""Base debator node for reducing code duplication across risk analysts.

All 3 risk debators (aggressive, conservative, neutral) follow the same pattern:
1. Extract debate state and reports from state
2. Build prompt with role-specific instructions
3. Invoke LLM
4. Update debate state with new argument
5. Return state update

This base class handles steps 1, 3, 4, 5. Subclasses provide 2.
"""

from typing import Dict


class BaseRiskDebator:
    """Base class for risk analyst debators."""

    # Override in subclass
    ROLE_NAME: str = ""  # e.g., "Aggressive Risk Analyst"
    LABEL: str = ""  # e.g., "Aggressive"
    HISTORY_KEY: str = ""  # e.g., "aggressive_history"
    PROMPT_TEMPLATE: str = ""

    def __init__(self, llm):
        self.llm = llm

    def _get_other_analysts(self) -> Dict[str, str]:
        """Return dict of other analyst labels -> response keys."""
        # Override in subclass
        return {}

    def _build_prompt(self, state: Dict) -> str:
        """Build the prompt with role-specific instructions."""
        trader_decision = state["trader_investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        history = state["risk_debate_state"].get("history", "")

        # Get other analysts' responses
        other_prompts = ""
        for label, response_key in self._get_other_analysts().items():
            response = state["risk_debate_state"].get(response_key, "")
            if response:
                other_prompts += f"Here is the last response from the {label.lower()} analyst: {response} "

        if not other_prompts:
            other_prompts = "If there are no responses from the other viewpoints yet, present your own argument based on the available data."

        return self.PROMPT_TEMPLATE.format(
            trader_decision=trader_decision,
            market_research_report=market_research_report,
            sentiment_report=sentiment_report,
            news_report=news_report,
            fundamentals_report=fundamentals_report,
            history=history,
            other_prompts=other_prompts,
        )

    def _update_state(self, state: Dict, argument: str) -> Dict:
        """Update debate state with new argument."""
        risk_debate_state = state["risk_debate_state"]
        return {
            "history": risk_debate_state["history"] + "\n" + argument,
            f"{self.HISTORY_KEY}": risk_debate_state.get(self.HISTORY_KEY, "") + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": self.LABEL,
            "current_aggressive_response": risk_debate_state.get("current_aggressive_response", "") if self.LABEL != "Aggressive" else argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", "") if self.LABEL != "Conservative" else argument,
            "current_neutral_response": risk_debate_state.get("current_neutral_response", "") if self.LABEL != "Neutral" else argument,
            "count": risk_debate_state["count"] + 1,
        }

    def __call__(self, state: Dict) -> Dict:
        """Execute the debator node and return state update."""
        prompt = self._build_prompt(state)
        response = self.llm.invoke(prompt)
        argument = f"{self.LABEL} Analyst: {response.content}"
        return {"risk_debate_state": self._update_state(state, argument)}


class AggressiveDebator(BaseRiskDebator):
    ROLE_NAME = "Aggressive Risk Analyst"
    LABEL = "Aggressive"
    HISTORY_KEY = "aggressive_history"
    PROMPT_TEMPLATE = """As the Aggressive Risk Analyst, your role is to actively champion high-reward, high-risk opportunities, emphasizing bold strategies and competitive advantages. When evaluating the trader's decision or plan, focus intently on the potential upside, growth potential, and innovative benefits—even when these come with elevated risk. Use the provided market data and sentiment analysis to strengthen your arguments and challenge the opposing views. Specifically, respond directly to each point made by the conservative and neutral analysts, countering with data-driven rebuttals and persuasive reasoning. Highlight where their caution might miss critical opportunities or where their assumptions may be overly conservative. Here is the trader's decision:

{trader_decision}

Your task is to create a compelling case for the trader's decision by questioning and critiquing the conservative and neutral stances to demonstrate why your high-reward perspective offers the best path forward. Incorporate insights from the following sources into your arguments:

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Here is the current conversation history: {history} {other_prompts}

Engage actively by addressing any specific concerns raised, refuting the weaknesses in their logic, and asserting the benefits of risk-taking to outpace market norms. Maintain a focus on debating and persuading, not just presenting data. Challenge each counterpoint to underscore why a high-risk approach is optimal. Output conversationally as if you are speaking without any special formatting."""

    def _get_other_analysts(self):
        return {
            "conservative": "current_conservative_response",
            "neutral": "current_neutral_response",
        }


class ConservativeDebator(BaseRiskDebator):
    ROLE_NAME = "Conservative Risk Analyst"
    LABEL = "Conservative"
    HISTORY_KEY = "conservative_history"
    PROMPT_TEMPLATE = """As the Conservative Risk Analyst, your primary objective is to protect assets, minimize volatility, and ensure steady, reliable growth. You prioritize stability, security, and risk mitigation, carefully assessing potential losses, economic downturns, and market volatility. When evaluating the trader's decision or plan, critically examine high-risk elements, pointing out where the decision may expose the firm to undue risk and where more cautious alternatives could secure long-term gains. Here is the trader's decision:

{trader_decision}

Your task is to actively counter the arguments of the Aggressive and Neutral Analysts, highlighting where their views may overlook potential threats or fail to prioritize sustainability. Respond directly to their points, drawing from the following data sources to build a convincing case for a low-risk approach adjustment to the trader's decision:

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Here is the current conversation history: {history} {other_prompts}

Engage by questioning their optimism and emphasizing the potential downsides they may have overlooked. Address each of their counterpoints to showcase why a conservative stance is ultimately the safest path for the firm's assets. Focus on debating and critiquing their arguments to demonstrate the strength of a low-risk strategy over their approaches. Output conversationally as if you are speaking without any special formatting."""

    def _get_other_analysts(self):
        return {
            "aggressive": "current_aggressive_response",
            "neutral": "current_neutral_response",
        }


class NeutralDebator(BaseRiskDebator):
    ROLE_NAME = "Neutral Risk Analyst"
    LABEL = "Neutral"
    HISTORY_KEY = "neutral_history"
    PROMPT_TEMPLATE = """As the Neutral Risk Analyst, your role is to provide a balanced perspective, weighing both the potential benefits and risks of the trader's decision or plan. You prioritize a well-rounded approach, evaluating the upsides and downsides while factoring in broader market trends, potential economic shifts, and diversification strategies.Here is the trader's decision:

{trader_decision}

Your task is to challenge both the Aggressive and Conservative Analysts, pointing out where each perspective may be overly optimistic or overly cautious. Use insights from the following data sources to support a moderate, sustainable strategy to adjust the trader's decision:

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Here is the current conversation history: {history} {other_prompts}

Engage actively by analyzing both sides critically, addressing weaknesses in the aggressive and conservative arguments to advocate for a more balanced approach. Challenge each of their points to illustrate why a moderate risk strategy might offer the best of both worlds, providing growth potential while safeguarding against extreme volatility. Focus on debating rather than simply presenting data, aiming to show that a balanced view can lead to the most reliable outcomes. Output conversationally as if you are speaking without any special formatting."""

    def _get_other_analysts(self):
        return {
            "aggressive": "current_aggressive_response",
            "conservative": "current_conservative_response",
        }


# Backwards compatibility
def create_aggressive_debator(llm):
    """Create aggressive debator node (backwards-compatible)."""
    return AggressiveDebator(llm)


def create_conservative_debator(llm):
    """Create conservative debator node (backwards-compatible)."""
    return ConservativeDebator(llm)


def create_neutral_debator(llm):
    """Create neutral debator node (backwards-compatible)."""
    return NeutralDebator(llm)
