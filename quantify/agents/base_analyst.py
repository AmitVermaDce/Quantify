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

Knowledge Base Integration:
    Automatically injects relevant knowledge from financial books into prompts.
    Enabled by default. Set USE_KNOWLEDGE = False in subclass to disable.
"""

from typing import Callable, List, Any, Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
from quantify.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)

# Lazy-load knowledge service
_kb_service_analyst = None

def _get_kb_service_analyst():
    """Lazy-load knowledge service for auto-injection."""
    global _kb_service_analyst
    if _kb_service_analyst is not None:
        return _kb_service_analyst

    try:
        from pathlib import Path
        current_file = Path(__file__).resolve()
        kb_path = current_file.parent.parent.parent / "knowledge_base"
        if kb_path.exists() and (kb_path / "service.py").exists():
            import sys
            if str(kb_path) not in sys.path:
                sys.path.insert(0, str(kb_path))
            from service import KnowledgeService
            _kb_service_analyst = KnowledgeService(
                index_path=str(kb_path / "data" / "knowledge_base"),
                auto_load=True,
            )
    except Exception:
        _kb_service_analyst = None

    return _kb_service_analyst


class BaseAnalystNode:
    """Base class for analyst nodes that produce reports via tool-calling."""

    # Override in subclass
    TOOLS: List[Callable] = []
    SYSTEM_MESSAGE: str = ""
    REPORT_KEY: str = ""  # e.g., "market_report", "news_report"
    ASSET_AWARE: bool = True  # Whether to use asset_type for instrument_context

    # Knowledge base integration
    USE_KNOWLEDGE: bool = True
    KNOWLEDGE_QUERY_TEMPLATE: str = "{ticker} analysis"
    KNOWLEDGE_K: int = 2

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

        # Auto-inject knowledge if enabled
        knowledge_section = ""
        if self.USE_KNOWLEDGE:
            kb = _get_kb_service_analyst()
            if kb:
                # Get ticker from state (will be passed via closure in __call__)
                pass  # Knowledge injection done in __call__ where state is available

        system_text = (
            "You are a helpful AI assistant, collaborating with other assistants."
            " Use the provided tools to progress towards answering the question."
            " If you are unable to fully answer, that's OK; another assistant with different tools"
            " will help where you left off. Execute what you can to make progress."
            " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
            " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
            " You have access to the following tools: {tool_names}.\n{system_message}"
            "For your reference, the current date is {current_date}. {instrument_context}"
        )

        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_text,
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

        # Auto-inject knowledge into result content if enabled
        if self.USE_KNOWLEDGE:
            kb = _get_kb_service_analyst()
            if kb:
                ticker = state.get("ticker", state.get("company_of_interest", "market"))
                query = self.KNOWLEDGE_QUERY_TEMPLATE.format(ticker=ticker)
                try:
                    context = kb.get_context(query, k=self.KNOWLEDGE_K)
                    if context and len(context.strip()) > 0:
                        # Append knowledge to report for downstream agents
                        existing_report = report
                        if existing_report:
                            report = f"{existing_report}\n\n== Relevant Financial Knowledge ==\n{context}\n== End Knowledge =="
                        else:
                            report = f"== Relevant Financial Knowledge ==\n{context}\n== End Knowledge =="
                except Exception:
                    pass  # Silently fail - knowledge is optional

        return {
            "messages": [result],
            self.REPORT_KEY: report,
        }


def create_analyst_node(llm: BaseLanguageModel, analyst_class: type) -> Callable:
    """Factory function to create analyst nodes (backwards-compatible)."""
    return analyst_class(llm)
