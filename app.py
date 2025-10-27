import os
from dotenv import load_dotenv
# --- Load Environment Variables FIRST ---
# This ensures all imported modules have access to the API keys.
load_dotenv()

import gradio as gr
from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage

# Import LangGraph components AFTER loading .env
from src.langgraph_agent.graph import build_graphs
from src.langgraph_agent.state import WorkflowState

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
            # Handle file uploads (which appear as tuples/dicts) vs. text
            if isinstance(user_msg, (dict, tuple)):
                messages.append(HumanMessage(content="User uploaded an image."))
            else:
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

# --- Main Chat Function (Bug-fixed logic) ---
def chat_fn(message: str, chat_history: list, agent_state: dict, img_upload: Image):
    """
    Handles user input and manages the agent's workflow state.
    """
    if initialization_error:
        chat_history.append((message, initialization_error))
        yield chat_history, {}, gr.update(interactive=True), gr.update(value="", interactive=True), gr.update(value=None, interactive=True)
        return

    # 1. Initialize or load the state
    # If state is empty OR a final diagnosis was reached, reset it.
    if not agent_state or agent_state.get("final_diagnosis"):
        current_state = reset_state_on_start()
    else:
        current_state = agent_state

    chat_history = chat_history or []
    
    # 2. Determine graph type based on state/input
    is_new_diagnosis = False
    if img_upload and (message.lower().strip() == "start" or message == ""):
        print("--- Running NEW diagnosis flow ---")
        is_new_diagnosis = True
        current_state = reset_state_on_start() # Reset state
        current_state["image"] = img_upload
        graph_to_run = diagnosis_graph
        # Add the image upload event to the chat
        chat_history.append(((img_upload.name,), None)) 
        
    elif current_state.get("symptoms_to_check") and not current_state.get("final_diagnosis"):
        print("--- Running REPLY symptom loop flow ---")
        graph_to_run = reply_graph
        chat_history.append([message, None]) # Add user's reply message
        # Keep image in state for context (Gradio clears it otherwise)
        current_state["image"] = img_upload 

    else:
        # Default behavior: Prompt user to start diagnosis
        if message: # Avoid adding empty messages
            chat_history.append([message, None])
        chat_history[-1][1] = "Hello! Please upload an image, then click 'Start Diagnosis'."
        yield chat_history, agent_state, gr.update(interactive=True), gr.update(value="", interactive=True), img_upload
        return

    # 3. Update LangChain history in state
    current_state["chat_history"] = convert_history_to_langchain(chat_history)
    
    # 4. Run the graph
    try:
        final_state = {}
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
                gr.update(value="", interactive=True), # Clear textbox
                gr.update(value=None, interactive=True) # Clear image
            ]
            yield chat_history, {}, output_controls
        else:
            # Flow is ongoing (waiting for the next symptom answer)
            output_controls = [
                gr.update(interactive=False), # Keep image box locked
                gr.update(value="", interactive=True),  # Clear textbox
                img_upload # Keep image displayed
            ]
            yield chat_history, final_state, output_controls

    except Exception as e:
        print(f"--- Graph Runtime Error --- \n{e}")
        error_msg = f"A runtime error occurred: {e}. Please check the console."
        chat_history[-1][1] = error_msg
        yield chat_history, {}, [gr.update(interactive=True), gr.update(interactive=True), gr.update(value=None, interactive=True)]

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
            
            btn_start = gr.Button("Start Diagnosis", variant="primary")
            btn_clear = gr.Button("Clear All & Start New")
            
            gr.Markdown(
                "**Instructions:**\n"
                "1. Upload an image.\n"
                "2. Click **Start Diagnosis**.\n"
                "3. Answer the agent's questions in the textbox."
            )

        with gr.Column(scale=2):
            gr.Markdown("### 2. Agent Conversation")
            chatbot = gr.Chatbot(label="Agent Conversation", height=500, bubble_full_width=False, avatar_images=None)
            txt_msg = gr.Textbox(
                label="Your message (Yes / No / etc.)",
                placeholder="Answer the agent's questions here...",
                interactive=True
            )
            
    # --- Event Handlers ---
    
    # 1. User submits text (Yes/No)
    txt_msg.submit(
        fn=chat_fn,
        inputs=[txt_msg, chatbot, agent_state, img_upload],
        outputs=[chatbot, agent_state, img_upload, txt_msg]
    )

    # 2. User clicks "Start Diagnosis"
    btn_start.click(
        fn=chat_fn,
        inputs=[gr.Textbox(value="start", visible=False), chatbot, agent_state, img_upload], # Pass "start" as message
        outputs=[chatbot, agent_state, img_upload, txt_msg]
    )

    # 3. Clear button resets everything
    btn_clear.click(
        fn=clear_all,
        inputs=None,
        outputs=[chatbot, agent_state, img_upload, txt_msg]
    )
    
    # 4. Uploading an image clears the chat/state to prepare for a new session
    img_upload.upload(
        fn=lambda: ([], {}, "Click 'Start Diagnosis' to begin."),
        inputs=None,
        outputs=[chatbot, agent_state, txt_msg]
    )

# --- Launch the App ---
if __name__ == "__main__":
    if initialization_error:
        print("\n\n*** CANNOT LAUNCH APP: Agent failed to initialize. ***")
        print(f"*** ERROR: {initialization_error} ***")
    else:
        demo.launch(debug=True)

