# TradingAgents/graph/setup.py

from __future__ import annotations

from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from quantify.agents import *
from quantify.agents.utils.agent_states import AgentState, AgentName

from .analyst_execution import build_analyst_execution_plan
from .conditional_logic import ConditionalLogic


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        analyst_concurrency_limit: int = 1,
    ) -> None:
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.analyst_concurrency_limit = analyst_concurrency_limit

    def setup_graph(
        self, selected_analysts: List[str] | None = None
    ) -> StateGraph:
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if selected_analysts is None:
            selected_analysts = ["market", "social", "news", "fundamentals"]
        plan = build_analyst_execution_plan(
            selected_analysts,
            concurrency_limit=self.analyst_concurrency_limit,
        )

        analyst_factories = {
            "market": lambda: create_market_analyst(self.quick_thinking_llm),
            "social": lambda: create_sentiment_analyst(self.quick_thinking_llm),
            "news": lambda: create_news_analyst(self.quick_thinking_llm),
            "fundamentals": lambda: create_fundamentals_analyst(self.quick_thinking_llm),
        }

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm)
        research_manager_node = create_research_manager(self.deep_thinking_llm)
        trader_node = create_trader(self.quick_thinking_llm)

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm)

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for spec in plan.specs:
            workflow.add_node(spec.agent_node, analyst_factories[spec.key]())
            workflow.add_node(spec.clear_node, create_msg_delete())
            workflow.add_node(spec.tool_node, self.tool_nodes[spec.key])

        # Add other nodes
        workflow.add_node(AgentName.BULL_RESEARCHER, bull_researcher_node)
        workflow.add_node(AgentName.BEAR_RESEARCHER, bear_researcher_node)
        workflow.add_node(AgentName.RESEARCH_MANAGER, research_manager_node)
        workflow.add_node(AgentName.TRADER, trader_node)
        workflow.add_node(AgentName.AGGRESSIVE_ANALYST, aggressive_analyst)
        workflow.add_node(AgentName.NEUTRAL_ANALYST, neutral_analyst)
        workflow.add_node(AgentName.CONSERVATIVE_ANALYST, conservative_analyst)
        workflow.add_node(AgentName.PORTFOLIO_MANAGER, portfolio_manager_node)

        # Define edges
        # Start with the first analyst
        workflow.add_edge(START, plan.specs[0].agent_node)

        # Connect analysts in sequence
        for i, spec in enumerate(plan.specs):
            current_analyst = spec.agent_node
            current_tools = spec.tool_node
            current_clear = spec.clear_node

            # Add conditional edges for current analyst
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{spec.key}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst or to Bull Researcher if this is the last analyst
            if i < len(plan.specs) - 1:
                workflow.add_edge(current_clear, plan.specs[i + 1].agent_node)
            else:
                workflow.add_edge(current_clear, AgentName.BULL_RESEARCHER)

        # Add remaining edges
        workflow.add_conditional_edges(
            AgentName.BULL_RESEARCHER,
            self.conditional_logic.should_continue_debate,
            {
                AgentName.BEAR_RESEARCHER: AgentName.BEAR_RESEARCHER,
                AgentName.RESEARCH_MANAGER: AgentName.RESEARCH_MANAGER,
            },
        )
        workflow.add_conditional_edges(
            AgentName.BEAR_RESEARCHER,
            self.conditional_logic.should_continue_debate,
            {
                AgentName.BULL_RESEARCHER: AgentName.BULL_RESEARCHER,
                AgentName.RESEARCH_MANAGER: AgentName.RESEARCH_MANAGER,
            },
        )
        workflow.add_edge(AgentName.RESEARCH_MANAGER, AgentName.TRADER)
        workflow.add_edge(AgentName.TRADER, AgentName.AGGRESSIVE_ANALYST)
        workflow.add_conditional_edges(
            AgentName.AGGRESSIVE_ANALYST,
            self.conditional_logic.should_continue_risk_analysis,
            {
                AgentName.CONSERVATIVE_ANALYST: AgentName.CONSERVATIVE_ANALYST,
                AgentName.PORTFOLIO_MANAGER: AgentName.PORTFOLIO_MANAGER,
            },
        )
        workflow.add_conditional_edges(
            AgentName.CONSERVATIVE_ANALYST,
            self.conditional_logic.should_continue_risk_analysis,
            {
                AgentName.NEUTRAL_ANALYST: AgentName.NEUTRAL_ANALYST,
                AgentName.PORTFOLIO_MANAGER: AgentName.PORTFOLIO_MANAGER,
            },
        )
        workflow.add_conditional_edges(
            AgentName.NEUTRAL_ANALYST,
            self.conditional_logic.should_continue_risk_analysis,
            {
                AgentName.AGGRESSIVE_ANALYST: AgentName.AGGRESSIVE_ANALYST,
                AgentName.PORTFOLIO_MANAGER: AgentName.PORTFOLIO_MANAGER,
            },
        )

        workflow.add_edge(AgentName.PORTFOLIO_MANAGER, END)

        return workflow
