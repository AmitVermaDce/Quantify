"""Conservative debator using base class."""
from quantify.agents.base_debator import ConservativeDebator


def create_conservative_debator(llm):
    """Create conservative debator node."""
    return ConservativeDebator(llm)
