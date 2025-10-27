from langgraph.graph import StateGraph, END
from .state import WorkflowState
from .nodes import (
    node_analyze_image,
    node_fetch_symptoms,
    node_ask_symptom_question,
    node_process_user_response,
    node_generate_final_response,
    router_check_image_analysis,
    router_should_ask_symptoms,
    router_should_continue_asking
)

def build_graph():
    """
    Builds and compiles the two agentic workflows:
    1. The main diagnosis graph.
    2. The reply-handling graph.
    """
    
    workflow = StateGraph(WorkflowState)

    workflow.add_node("analyze_image", node_analyze_image)
    workflow.add_node("fetch_symptoms", node_fetch_symptoms)
    workflow.add_node("ask_symptom_question", node_ask_symptom_question)
    workflow.add_node("generate_final_response", node_generate_final_response)

    workflow.set_entry_point("analyze_image")
    
    workflow.add_conditional_edges(
        "analyze_image",
        router_check_image_analysis,
        {
            "fetch_symptoms": "fetch_symptoms",
            "end_error": END  
        }
    )

    workflow.add_conditional_edges(
        "fetch_symptoms",
        router_should_ask_symptoms,
        {
            "ask_symptom_question": "ask_symptom_question",
            "generate_final_response": "generate_final_response"
        }
    )
    
    workflow.add_edge("ask_symptom_question", END)
    workflow.add_edge("generate_final_response", END)
    
    diagnosis_graph = workflow.compile()
    
    
    reply_workflow = StateGraph(WorkflowState)
    
    reply_workflow.add_node("process_user_response", node_process_user_response)
    reply_workflow.add_node("ask_symptom_question", node_ask_symptom_question)
    reply_workflow.add_node("generate_final_response", node_generate_final_response)

    reply_workflow.set_entry_point("process_user_response")
    
    reply_workflow.add_conditional_edges(
        "process_user_response",
        router_should_continue_asking,
        {
            "ask_symptom_question": "ask_symptom_question",
            "generate_final_response": "generate_final_response"
        }
    )
    
    reply_workflow.add_edge("ask_symptom_question", END)
    reply_workflow.add_edge("generate_final_response", END)
    
    reply_graph = reply_workflow.compile()
    
    print("--- LangGraph Compiled ---")
    
    return diagnosis_graph, reply_graph
