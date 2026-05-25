"""Knowledge Base Integration for TradingAgents.

This module provides a mixin class that injects knowledge base context
into agent prompts, enabling RAG-based decision making with financial book
knowledge and research.

Usage in any agent:
    from quantify.agents.knowledge_mixin import KnowledgeMixin

    class MyAgent(KnowledgeMixin, BaseAgent):
        def __call__(self, state):
            # Add knowledge context to prompt
            self._inject_knowledge(state, "value investing principles")
            return super().__call__(state)

Auto-injection for all agents:
    from quantify.agents.knowledge_mixin import auto_inject_knowledge

    # Wrap your agent graph to auto-inject knowledge before each agent call
    app = auto_inject_knowledge(app)
"""

import os
from typing import Dict, Any, List, Optional
from pathlib import Path


def _find_knowledge_base_path() -> Optional[Path]:
    """
    Find the knowledge_base directory using multiple strategies.

    Returns:
        Path to knowledge_base directory or None if not found
    """
    current_file = Path(__file__).resolve()

    # Strategy 1: Project root (quantify/agents/../.. = quantify/../.. = project root)
    project_root = current_file.parent.parent.parent
    kb_path = project_root / "knowledge_base"
    if kb_path.exists() and (kb_path / "service.py").exists():
        return kb_path

    # Strategy 2: Sibling directory (quantify/agents/../knowledge_base)
    sibling_path = current_file.parent.parent / "knowledge_base"
    if sibling_path.exists() and (sibling_path / "service.py").exists():
        return sibling_path

    # Strategy 3: Check if knowledge_base is in sys.path already
    import sys
    for path_str in sys.path:
        path = Path(path_str)
        if path.name == "knowledge_base" and (path / "service.py").exists():
            return path

    return None


# Resolve knowledge base path at module load time
_KNOWLEDGE_BASE_PATH = _find_knowledge_base_path()

# Add to sys.path if found
if _KNOWLEDGE_BASE_PATH:
    import sys
    if str(_KNOWLEDGE_BASE_PATH) not in sys.path:
        sys.path.insert(0, str(_KNOWLEDGE_BASE_PATH))


class KnowledgeMixin:
    """
    Mixin class to add knowledge base access to any TradingAgent.

    Provides methods to:
    - Search knowledge base for relevant context
    - Inject knowledge into agent prompts
    - Get knowledge-enhanced system messages

    Attributes:
        use_knowledge: Enable/disable knowledge injection (default: True)
        knowledge_query: Default search query (can override per-call)
        knowledge_k: Number of knowledge results to include
    """

    # Class-level configuration (override in subclass)
    USE_KNOWLEDGE: bool = True
    DEFAULT_KNOWLEDGE_QUERY: Optional[str] = None  # e.g., "technical analysis patterns"
    KNOWLEDGE_K: int = 3
    KNOWLEDGE_CATEGORY: Optional[str] = None  # "Investing_Books" or "Psychological_Books"

    # Instance state (use instance attribute, not class attribute)
    _kb_service: Optional[Any] = None

    @property
    def kb_service(self) -> Optional[Any]:
        """Lazy-load knowledge service."""
        if not hasattr(self, '_kb_service_instance') or self._kb_service_instance is None:
            if _KNOWLEDGE_BASE_PATH is None:
                print("Knowledge base not available: directory not found")
                self._kb_service_instance = None
                return None

            try:
                from service import KnowledgeService
                self._kb_service_instance = KnowledgeService(
                    index_path=str(_KNOWLEDGE_BASE_PATH / "data" / "knowledge_base"),
                    auto_load=True,
                )
            except Exception as e:
                # Silently fail - knowledge base is optional enhancement
                print(f"Knowledge base not available: {e}")
                self._kb_service_instance = None

        return self._kb_service_instance

    def _get_knowledge_query(self, state: Dict, override_query: Optional[str] = None) -> str:
        """
        Determine the knowledge search query.

        Priority:
        1. override_query (explicit)
        2. state.get("knowledge_query")
        3. class DEFAULT_KNOWLEDGE_QUERY
        4. Fallback based on agent role
        """
        if override_query:
            return override_query

        if "knowledge_query" in state:
            return state["knowledge_query"]

        if self.DEFAULT_KNOWLEDGE_QUERY:
            return self.DEFAULT_KNOWLEDGE_QUERY

        # Fallback: derive from context
        ticker = state.get("ticker", state.get("company_of_interest", "market"))
        return f"{ticker} investment analysis"

    def _get_knowledge_context(
        self,
        state: Dict,
        query: Optional[str] = None,
        k: Optional[int] = None,
        category: Optional[str] = None,
        max_tokens: int = 1500,
    ) -> str:
        """
        Get relevant knowledge context for the current decision.

        Args:
            state: Current agent state
            query: Override search query
            k: Number of results
            category: Filter by category
            max_tokens: Maximum context length

        Returns:
            Formatted knowledge context string
        """
        if not self.USE_KNOWLEDGE:
            return ""

        if self.kb_service is None:
            return ""

        search_query = self._get_knowledge_query(state, query)
        result_k = k or self.KNOWLEDGE_K
        result_cat = category or self.KNOWLEDGE_CATEGORY

        try:
            return self.kb_service.get_context(
                query=search_query,
                k=result_k,
                max_tokens=max_tokens,
                category=result_cat,
            )
        except Exception as e:
            # Log but don't fail - knowledge is enhancement
            print(f"Knowledge search failed: {e}")
            return ""

    def _inject_knowledge(
        self,
        state: Dict,
        query: Optional[str] = None,
        k: Optional[int] = None,
        section_name: str = "Relevant Financial Knowledge",
    ) -> str:
        """
        Inject knowledge base context into state for prompt building.

        Stores result in state["_knowledge_context"] for later use.

        Returns:
            The knowledge context string (empty if disabled/failed)
        """
        context = self._get_knowledge_context(state, query=query, k=k)

        if context and len(context.strip()) > 0:
            state["_knowledge_context"] = context
            state["_knowledge_section_name"] = section_name
        else:
            state["_knowledge_context"] = ""

        return context

    def _build_knowledge_section(self, state: Dict) -> str:
        """
        Build the knowledge section for system message.

        Call this when building your agent's system message.
        """
        context = state.get("_knowledge_context", "")
        section_name = state.get("_knowledge_section_name", "Relevant Financial Knowledge")

        if not context:
            return ""

        return f"""
== {section_name} ==
The following knowledge from financial books and research is relevant to your analysis:

{context}

Use this knowledge to inform your decisions, but also consider current market conditions.
== End {section_name} ==
"""

    def refresh_knowledge(self, state: Dict, query: Optional[str] = None) -> None:
        """
        Refresh knowledge context based on updated state.

        Call this when state changes significantly (e.g., new ticker).
        """
        self._inject_knowledge(state, query=query)


# Convenience function for direct use
def get_knowledge_context(query: str, k: int = 3) -> str:
    """
    Quick helper to get knowledge context without class instantiation.

    Usage:
        from quantify.agents.knowledge_mixin import get_knowledge_context
        context = get_knowledge_context("margin of safety")
    """
    if _KNOWLEDGE_BASE_PATH is None:
        print("Knowledge base not available: directory not found")
        return ""

    try:
        from service import KnowledgeService
        kb = KnowledgeService(
            index_path=str(_KNOWLEDGE_BASE_PATH / "data" / "knowledge_base")
        )
        return kb.get_context(query, k=k)
    except Exception as e:
        print(f"Knowledge lookup failed: {e}")
        return ""


def auto_inject_knowledge(graph, query_template: str = "{ticker} investment analysis"):
    """
    Wrap a CompiledGraph to auto-inject knowledge before each agent call.

    This modifies the state before each node execution to include relevant
    knowledge base context.

    Args:
        graph: The CompiledGraph (LangGraph application) to wrap
        query_template: Template for knowledge query. Can use {ticker} placeholder.

    Returns:
        The same graph with knowledge injection enabled

    Usage:
        from quantify.agents.knowledge_mixin import auto_inject_knowledge

        # After building your graph:
        app = create_trading_agent()
        app = auto_inject_knowledge(app)

        # Now every invoke() will include knowledge context
        result = app.invoke({"ticker": "AAPL", ...})
    """
    if _KNOWLEDGE_BASE_PATH is None:
        print("Knowledge base not available: skipping auto-injection")
        return graph

    try:
        from service import KnowledgeService
        kb = KnowledgeService(
            index_path=str(_KNOWLEDGE_BASE_PATH / "data" / "knowledge_base"),
            auto_load=True,
        )
    except Exception as e:
        print(f"Knowledge base failed to load: {e}")
        return graph

    # Store kb instance on the graph
    graph._kb_service = kb
    graph._kb_query_template = query_template

    # Patch the invoke method to inject knowledge
    _original_invoke = graph.invoke

    def _patched_invoke(state, *args, **kwargs):
        # Inject knowledge into state before running graph
        ticker = state.get("ticker", state.get("company_of_interest", "market"))
        query = graph._kb_query_template.format(ticker=ticker)

        try:
            context = graph._kb_service.get_context(query, k=3)
            if context and len(context.strip()) > 0:
                state["_knowledge_context"] = context
                state["_knowledge_section_name"] = "Relevant Financial Knowledge"
        except Exception as e:
            print(f"Knowledge injection failed: {e}")

        return _original_invoke(state, *args, **kwargs)

    graph.invoke = _patched_invoke

    return graph
