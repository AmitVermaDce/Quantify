"""Base analyst node for reducing code duplication across analyst types.

All 4 analysts (market, news, fundamentals, sentiment) follow the same pattern:
1. Extract state variables (ticker, date, asset_type)
2. Define tools list
3. Build system message with role-specific instructions
4. Create prompt with standard template
5. Bind tools and invoke LLM
6. Extract report from response
7. Return state update with report

This base class handles steps 1, 4, 5, 6, 7. Subclasses provide 2, 3.
"""

from typing import Callable, List, Any, Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
from quantify.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)


class BaseAnalystNode:
    """Base class for analyst nodes that produce reports via tool-calling."""

    # Override in subclass
    TOOLS: List[Callable] = []
    SYSTEM_MESSAGE: str = ""
    REPORT_KEY: str = ""  # e.g., "market_report", "news_report"
    ASSET_AWARE: bool = True  # Whether to use asset_type for instrument_context

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm

    def _get_state_vars(self, state: Dict) -> tuple:
        """Extract common state variables."""
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock") if self.ASSET_AWARE else "stock"
        instrument_context = build_instrument_context(
            state["company_of_interest"], asset_type
        )
        return current_date, asset_type, instrument_context

    def _build_prompt(self, system_message: str, current_date: str,
                      instrument_context: str, tools: List[Callable]) -> ChatPromptTemplate:
        """Build the standard analyst prompt template."""
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

    def _invoke(self, state: Dict) -> tuple:
        """Invoke the LLM and return (result, report_content)."""
        current_date, asset_type, instrument_context = self._get_state_vars(state)

        prompt = self._build_prompt(
            system_message=self.SYSTEM_MESSAGE + get_language_instruction(),
            current_date=current_date,
            instrument_context=instrument_context,
            tools=self.TOOLS,
        )

        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in self.TOOLS]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(system_message=self.SYSTEM_MESSAGE + get_language_instruction())

        chain = prompt | self.llm.bind_tools(self.TOOLS)
        result = chain.invoke({"messages": state["messages"]})

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return result, report

    def __call__(self, state: Dict) -> Dict:
        """Execute the analyst node and return state update."""
        result, report = self._invoke(state)
        return {
            "messages": [result],
            self.REPORT_KEY: report,
        }


def create_analyst_node(llm: BaseLanguageModel, analyst_class: type) -> Callable:
    """Factory function to create analyst nodes (backwards-compatible)."""
    return analyst_class(llm)
