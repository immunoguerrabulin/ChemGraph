from typing import TypedDict, Annotated
from langgraph.graph import add_messages
from langgraph.managed.is_last_step import RemainingSteps


class _ActiveSpaceBase(TypedDict):
    """Base fields shared by all active-space states"""

    messages: Annotated[list, add_messages]

# additional content for the base state. (Optional)
class ActiveSpaceState(_ActiveSpaceBase, total=False):
    """State for the active-space workflow"""

    task: str
    molecule: object
    active_space_guess: object
    executor_request: object
    diagnostics: object
    result: object
    remaining_steps: RemainingSteps
    notes: str
