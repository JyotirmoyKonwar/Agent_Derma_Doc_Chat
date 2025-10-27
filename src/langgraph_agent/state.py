from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage
from PIL.Image import Image

class WorkflowState(TypedDict):
    """
    Represents the state of our agent's workflow.
    This dictionary is passed between nodes, allowing them to share information.
    """
    # Input from the user
    image: Optional[Image]
    chat_history: List[BaseMessage]
    
    # Information gathered by tools
    disease_prediction: str
    symptoms_to_check: List[str]
    treatment_info: str
    
    # State for the "self-thought loop"
    symptoms_confirmed: List[str]
    current_symptom_index: int
    
    # Final state marker
    final_diagnosis: str
