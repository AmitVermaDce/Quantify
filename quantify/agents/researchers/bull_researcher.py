"""Bull researcher using base class."""
from quantify.agents.base_researcher import BullResearcher


def create_bull_researcher(llm):
    """Create bull researcher node."""
    return BullResearcher(llm)
