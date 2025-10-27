from langchain_core.messages import AIMessage, HumanMessage
from .state import WorkflowState
from .tools import tool_analyze_skin_image, tool_fetch_disease_info
from .llms import (
    symptom_classifier_chain, 
    question_generation_chain, 
    summary_generation_chain
)


def node_analyze_image(state: WorkflowState) -> WorkflowState:
    """Analyzes the user's image and updates the state."""
    print("--- Node: Analyzing Image ---")
    image = state.get("image")
    if not image:
        state['chat_history'].append(AIMessage(
            content="Please upload an image first."
        ))
        return state

    prediction_result = tool_analyze_skin_image.invoke({"image": image})
    
    if "Error:" in prediction_result:
        state['chat_history'].append(AIMessage(content=prediction_result))
        state['final_diagnosis'] = "Error" 
        return state
    
    state['disease_prediction'] = prediction_result
    return state

def node_fetch_symptoms(state: WorkflowState) -> WorkflowState:
    """Fetches symptoms for the predicted disease."""
    print(f"--- Node: Fetching Symptoms for {state['disease_prediction']} ---")
    disease = state['disease_prediction']
    
    info = tool_fetch_disease_info.invoke({"disease_name": disease})
    
    if "error" in info:
        state['chat_history'].append(AIMessage(content=info['error']))
        state['final_diagnosis'] = "Error" 
        return state
    
    state['symptoms_to_check'] = info.get("symptoms", [])
    state['treatment_info'] = info.get("treatment", "No treatment info available.")
    state['current_symptom_index'] = 0
    state['symptoms_confirmed'] = []
    
    if not state['symptoms_to_check']:
        print("No symptoms found to check. Proceeding to final response.")
    
    return state

def node_ask_symptom_question(state: WorkflowState) -> WorkflowState:
    """Asks the user the next symptom question."""
    print(f"--- Node: Asking Symptom Question {state['current_symptom_index']} ---")
    symptoms = state['symptoms_to_check']
    index = state['current_symptom_index']
    
    symptom = symptoms[index]
    
    question = question_generation_chain.invoke({"symptom": symptom})
    
    state['chat_history'].append(AIMessage(content=question))
    state['current_symptom_index'] = index + 1
    return state

def node_process_user_response(state: WorkflowState) -> WorkflowState:
    """Processes the user's 'yes' or 'no' response to a symptom question."""
    print("--- Node: Processing User Response ---")
    last_human_message = state['chat_history'][-1].content
    
    index = state['current_symptom_index']
    last_asked_symptom = state['symptoms_to_check'][index - 1]
    
    try:
        classification = symptom_classifier_chain.invoke(
            {"last_human_message": last_human_message}
        )
        
        if classification.get("classification") == "yes":
            print(f"User confirmed symptom: {last_asked_symptom}")
            state['symptoms_confirmed'].append(last_asked_symptom)
        else:
            print(f"User denied symptom: {last_asked_symptom}")
            
    except Exception as e:
        print(f"Error classifying user response: {e}. Assuming 'unclear'.")
    
    return state
    
def node_generate_final_response(state: WorkflowState) -> WorkflowState:
    """Generates the final summary and disclaimer for the user."""
    print("--- Node: Generating Final Response ---")
    
    disclaimer = (
        "\n\n**DISCLAIMER:**\n"
        "I am just a dumb agent, not a medical professional. "
        "This is a side project for learning purposes. "
        "Please **DO NOT** take this information for face value. "
        "Consult a real doctor or dermatologist for any medical concerns."
    )
    
    summary = summary_generation_chain.invoke({
        "disease": state['disease_prediction'],
        "symptoms": ", ".join(state['symptoms_confirmed']) or "None confirmed",
        "treatment": state['treatment_info'],
        "disclaimer": disclaimer
    })
    
    state['chat_history'].append(AIMessage(content=summary))
    state['final_diagnosis'] = "Complete" 
    return state


def router_should_ask_symptoms(state: WorkflowState) -> str:
    """
    Checks if there are symptoms to ask about.
    If yes -> ask_symptom_question
    If no -> generate_final_response
    """
    if state.get("symptoms_to_check"):
        return "ask_symptom_question"
    else:
        return "generate_final_response"

def router_should_continue_asking(state: WorkflowState) -> str:
    """
    Checks if we have more symptoms to ask about after a user's response.
    If yes -> ask_symptom_question
    If no -> generate_final_response
    """
    if state['current_symptom_index'] < len(state['symptoms_to_check']):
        return "ask_symptom_question"
    else:
        return "generate_final_response"

def router_check_image_analysis(state: WorkflowState) -> str:
    """
    Checks if the image analysis was successful.
    """
    if state.get("final_diagnosis") == "Error":
        return "end_error" 
    else:
        return "fetch_symptoms"
