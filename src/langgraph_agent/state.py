from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage
from PIL.Image import Image

class WorkflowState(TypedDict):
    """
    Represents the state of our agent's workflow.
    This dictionary is passed between nodes, allowing them to share information.
    """
    image: Optional[Image]
    chat_history: List[BaseMessage]
    
    disease_prediction: str
    symptoms_to_check: List[str]
    treatment_info: str
    
    symptoms_confirmed: List[str]
    current_symptom_index: int
    
    final_diagnosis: str
