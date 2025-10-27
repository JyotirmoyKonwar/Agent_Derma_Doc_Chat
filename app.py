import gradio as gr
from PIL import Image
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Import LangGraph components
from src.langgraph_agent.graph import build_graphs
from src.langgraph_agent.state import WorkflowState

# Load environment variables
load_dotenv()

# --- Global Initialization ---
diagnosis_graph = None
reply_graph = None
initialization_error = None

try:
    diagnosis_graph, reply_graph = build_graphs()
    print("Gradio app and graphs initialized successfully.")
except Exception as e:
    initialization_error = f"CRITICAL ERROR: Could not build graphs. {e}. Check API keys."
    print(initialization_error)

# --- Utility Functions ---

def convert_history_to_langchain(chat_history):
    """Converts Gradio history to Langchain message list."""
    messages = []
    for user_msg, ai_msg in chat_history:
        if user_msg is not None:
            messages.append(HumanMessage(content=user_msg))
        if ai_msg is not None:
            messages.append(AIMessage(content=ai_msg))
    return messages

def reset_state_on_start():
    """Returns a fresh, empty state for a new diagnosis."""
    return WorkflowState(
        image=None,
        chat_history=[],
        disease_prediction="",
        symptoms_to_check=[],
        symptoms_confirmed=[],
        current_symptom_index=0,
        treatment_info="",
        final_diagnosis=""
    )

# --- Main Chat Function ---
def chat_fn(message: str, chat_history: list, agent_state: dict, img_upload: Image):
    """
    Handles user input and manages the agent's workflow state.
    """
    if initialization_error:
        chat_history.append((message, initialization_error))
        yield chat_history, {}, gr.update(interactive=True), gr.update(value="", interactive=True)
        return

    # 1. Initialize or load the state
    current_state = agent_state or reset_state_on_start()
    chat_history = chat_history or []
    
    # 2. Determine graph type based on state/input
    is_new_diagnosis = False
    if img_upload and message.lower().strip() == "start":
        print("--- Running NEW diagnosis flow ---")
        is_new_diagnosis = True
        current_state = reset_state_on_start() # Reset state for new image
        current_state["image"] = img_upload
        graph_to_run = diagnosis_graph
        chat_history = [] # Start a new chat
        
    elif current_state.get("symptoms_to_check") and not current_state.get("final_diagnosis"):
        print("--- Running REPLY symptom loop flow ---")
        graph_to_run = reply_graph
        # Keep image in state for context (Gradio clears it otherwise)
        current_state["image"] = img_upload 

    else:
        # Default behavior: Prompt user to start diagnosis
        chat_history.append([message, None])
        chat_history[-1][1] = "Hello! Please upload an image, then type 'Start' and press Send."
        yield chat_history, agent_state, gr.update(interactive=True), gr.update(value="", interactive=True)
        return

    # 3. Add user message to history and update state
    chat_history.append([message, None])
    current_state["chat_history"] = convert_history_to_langchain(chat_history)
    
    # 4. Run the graph
    try:
        final_state = {}
        # .stream() runs the graph until it hits an END node
        for step in graph_to_run.stream(current_state, {"recursion_limit": 100}):
            final_state = list(step.values())[0]

        # 5. Extract the AI's response and update Gradio history
        ai_response = final_state['chat_history'][-1].content
        chat_history[-1][1] = ai_response 

        # 6. Check if the flow has finished
        if final_state.get("final_diagnosis"):
            print("--- Agent Flow ENDED ---")
            # Reset controls for a new session
            output_controls = [
                gr.update(interactive=True),  # Unlock image box
                gr.update(value="", interactive=True) # Clear textbox
            ]
            yield chat_history, {}, output_controls
        else:
            # Flow is ongoing (waiting for the next symptom answer)
            output_controls = [
                gr.update(interactive=False), # Keep image box locked
                gr.update(value="", interactive=True)  # Clear textbox
            ]
            yield chat_history, final_state, output_controls

    except Exception as e:
        print(f"--- Graph Runtime Error --- \n{e}")
        error_msg = f"A runtime error occurred: {e}. Please check the console."
        chat_history[-1][1] = error_msg
        yield chat_history, {}, [gr.update(interactive=True), gr.update(interactive=True)]

def clear_all():
    """Clears chat, state, and image."""
    return [], {}, None, ""

# --- Build the Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="Agentic Skin AI") as demo:
    gr.Markdown("# 🩺 Multimodal Agentic Skin Disease AI")
    gr.Markdown(
        "**Disclaimer:** This is a demo project and NOT a medical device. "
        "Consult a real doctor for any medical concerns."
    )
    agent_state = gr.State({})

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Image Input")
            img_upload = gr.Image(type="pil", label="Upload Skin Image", interactive=True)
            btn_clear = gr.Button("Clear All & Start New Diagnosis")
            gr.Markdown(
                "**Instructions:**\n"
                "1. Upload the image.\n"
                "2. Type **Start** and press Send.\n"
                "3. Answer the agent's symptom questions (e.g., 'Yes' / 'No')."
            )

        with gr.Column(scale=2):
            gr.Markdown("### 2. Agent Conversation")
            chatbot = gr.Chatbot(label="Agent Conversation", height=500, bubble_full_width=False, avatar_images=None)
            txt_msg = gr.Textbox(
                label="Your message (Start / Yes / No)",
                placeholder="Upload an image, then type 'Start' to begin.",
                interactive=True
            )
    
    # --- Event Handlers ---
    submit_event = txt_msg.submit(
        fn=chat_fn,
        inputs=[txt_msg, chatbot, agent_state, img_upload],
        outputs=[chatbot, agent_state, img_upload, txt_msg]
    )
    
    btn_clear.click(
        fn=clear_all,
        inputs=None,
        outputs=[chatbot, agent_state, img_upload, txt_msg]
    )
    
    # This handler just clears the chat/state when a new image is added
    # It *prepares* for the "start" command
    img_upload.upload(
        fn=lambda: ([], {}, "Type 'Start' and press Send to begin analysis."),
        inputs=None,
        outputs=[chatbot, agent_state, txt_msg]
    )

# --- Launch the App ---
if __name__ == "__main__":
    demo.launch(debug=True)

