import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import Runnable

LLM_REPO_ID = os.environ.get("LLM_REPO_ID", "Jyo-K/Fine-Tuned-Qwen2.5_1B")
HF_API_KEY = os.environ.get("HF_API_KEY")

_llm_instance = None

def get_llm():
    """Lazily initializes and returns the LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        if not HF_API_KEY:
            raise ValueError("HF_API_KEY environment variable not set. Cannot initialize LLM.")
        _llm_instance = HuggingFaceEndpoint(
            repo_id=LLM_REPO_ID,
            huggingfacehub_api_token=HF_API_KEY,
            temperature=0.1,
            max_new_tokens=256,
            top_k=50,
            top_p=0.95
        )
        print("--- LLM Initialized ---")
    return _llm_instance

classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful classification assistant. "
        "Your task is to classify the user's last response as 'yes', 'no', or 'unclear' "
        "based on their message. "
        "User's previous message: '{last_human_message}'"
        "\nRespond ONLY with a single JSON object in the format: "
        "{{\"classification\": \"yes\"}} or {{\"classification\": \"no\"}} or {{\"classification\": \"unclear\"}}"
    ))
])
symptom_classifier_chain = classifier_prompt | Runnable(get_llm) | JsonOutputParser()

question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a friendly medical assistant bot. Ask the user if they are experiencing the "
        "following symptom. Be clear and concise. Do not add any extra greeting or sign-off. "
        "Symptom: '{symptom}'"
        "\nExample: Are you experiencing any itchiness or a rash?"
    ))
])
question_generation_chain = question_prompt | Runnable(get_llm) | StrOutputParser()


summary_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful medical assistant providing a summary. "
        "Based on the initial image analysis and confirmed symptoms, generate a summary. "
        "DO NOT provide a definitive diagnosis. "
        "Structure your response clearly: "
        "1. Start by stating the potential condition identified from the image."
        "2. List the symptoms the user confirmed."
        "3. Provide the general treatment information found for this condition."
        "4. **ALWAYS** include the provided disclaimer at the very end."
        "\n---"
        "Initial Image Prediction: {disease}"
        "Confirmed Symptoms: {symptoms}"
        "Potential Treatment Information: {treatment}"
        "Disclaimer: {disclaimer}"
        "\n---"
        "Generate your summary now."
    ))
])
summary_generation_chain = summary_prompt | Runnable(get_llm) | StrOutputParser()

