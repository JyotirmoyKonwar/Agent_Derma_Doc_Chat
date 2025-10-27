import os
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage

from src.langgraph_agent.graph import build_graph
from src.langgraph_agent.state import WorkflowState

diagnosis_graph = None
reply_graph = None
initialization_error = None

try:
    diagnosis_graph, reply_graph = build_graph() 
    print("Gradio app and graphs initialized successfully.")
except Exception as e:
    initialization_error = f"CRITICAL ERROR: Could not build graphs. {e}. Check API keys."
    print(initialization_error)


def convert_history_to_langchain(chat_history):
    """Converts Gradio history to Langchain message list."""
    messages = []
    for user_msg, ai_msg in chat_history:
        if user_msg is not None:
            if isinstance(user_msg, (dict, tuple)):
                messages.append(HumanMessage(content="User uploaded an image."))
            else:
                 messages.append(HumanMessage(content=user_msg))
        if ai_msg is not None:
            messages.append(AIMessage(content=ai_msg))
    return messages

def reset_state_on_start():
    """Fresh empty state for every new diagnosis."""
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


def chat_fn(message: str, chat_history: list, agent_state: dict, img_upload: Image):
    """
    Handles user input and manages the agent's workflow state.
    """
    if initialization_error:
        chat_history.append((message, initialization_error))
        yield chat_history, {}, gr.update(value=None, interactive=True), gr.update(value="", interactive=True)
        return

    if not agent_state or agent_state.get("final_diagnosis"):
        current_state = reset_state_on_start()
    else:
        current_state = agent_state

    chat_history = chat_history or []
    

    is_new_diagnosis = False
    if img_upload and (message.lower().strip() == "start" or message == ""):
        print("--- Running NEW diagnosis flow ---")
        is_new_diagnosis = True
        current_state = reset_state_on_start() 
        current_state["image"] = img_upload
        graph_to_run = diagnosis_graph
        chat_history.append([(img_upload,), None]) 
        
    elif current_state.get("symptoms_to_check") and not current_state.get("final_diagnosis"):
        print("--- Running REPLY symptom loop flow ---")
        graph_to_run = reply_graph
        chat_history.append([message, None])


    else:

        if message: 
            chat_history.append([message, None])
        chat_history[-1][1] = "Hello! Please upload an image, then click 'Start Diagnosis'."

        yield chat_history, agent_state, gr.update(value=img_upload, interactive=True), gr.update(value="", interactive=True)
        return

    current_state["chat_history"] = convert_history_to_langchain(chat_history)
    
    try:
        final_state = {}
        for step in graph_to_run.stream(current_state, {"recursion_limit": 100}):

            final_state = list(step.values())[0]

        ai_response = final_state['chat_history'][-1].content
        chat_history[-1][1] = ai_response 

        if final_state.get("final_diagnosis"):
            print("--- Agent Flow ENDED ---")
            yield chat_history, {}, gr.update(value=None, interactive=True), gr.update(value="", interactive=True)
        else:
            yield chat_history, final_state, gr.update(value=img_upload, interactive=False), gr.update(value="", interactive=True)

    except Exception as e:
        print(f"--- Graph Runtime Error --- \n{e}")
        error_msg = f"A runtime error occurred: {e}. Please check the console."
        chat_history[-1][1] = error_msg
        yield chat_history, {}, gr.update(value=None, interactive=True), gr.update(value="", interactive=True)

def clear_all():
    """Clears chat, state, and image."""
    return [], {}, None, ""

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
            
    txt_msg.submit(
        fn=chat_fn,
        inputs=[txt_msg, chatbot, agent_state, img_upload],
        outputs=[chatbot, agent_state, img_upload, txt_msg]
    )

    btn_start.click(
        fn=chat_fn,
        inputs=[gr.Textbox(value="start", visible=False), chatbot, agent_state, img_upload], 
        outputs=[chatbot, agent_state, img_upload, txt_msg]
    )
    
    btn_clear.click(
        fn=clear_all,
        inputs=None,
        outputs=[chatbot, agent_state, img_upload, txt_msg]
    )
    
    img_upload.upload(
        fn=lambda: ([], {}, "Click 'Start Diagnosis' to begin."),
        inputs=None,
        outputs=[chatbot, agent_state, txt_msg]
    )

if __name__ == "__main__":
    if initialization_error:
        print("\n\n*** CANNOT LAUNCH APP: Agent failed to initialize. ***")
        print(f"*** ERROR: {initialization_error} ***")
    else:
        demo.launch(debug=True)
