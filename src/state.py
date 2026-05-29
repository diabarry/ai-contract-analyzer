from typing import List, TypedDict, Annotated
import operator

class AgentState(TypedDict):
    question: str
    # Utilizing operator.add triggers state reduction, merging new retrieved 
    # document lists across successive graph loop iterations instead of overwriting them.
    documents: Annotated[List[any], operator.add]
    generation: str
    loop_count: int      # Critical safety counter utilized to bound graph recursion and prevent infinite loops
    technical_query: str # Stores the optimized technical reformulation of the initial question