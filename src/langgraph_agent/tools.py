import os
import requests
import io
from typing import List
from PIL.Image import Image
from langchain_core.tools import tool
from pinecone import Pinecone

# --- API Endpoints and Keys ---
SWIN_API_URL = os.environ.get("SWIN_MODEL_URL", "https://api-inference.huggingface.co/models/Jyo-K/skin_swin")
HF_API_TOKEN = os.environ.get("HF_TOKEN")
EMBEDDING_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

# --- **BUG FIX**: Manual Label Mapping for 'Jyo-K/skin_swin' ---
# The model on the hub is missing id2label, so we must define it here.
SWIN_LABELS = [
    '1. Enfeksiyonel', 
    '2. Ekzama', 
    '3. Akne', 
    '4. Pigment', 
    '5. Benign', 
    '6. Malign'
]

# --- Initialize Global Clients ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    print(f"Error initializing Pinecone client. Check PINECONE_API_KEY and INDEX_NAME: {e}")
    index = None

def get_embedding_hf(text: str) -> List[float]:
    """Gets the embedding for a text query using the HF Inference API."""
    response = requests.post(
        EMBEDDING_API_URL,
        headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
        json={"inputs": text, "options": {"wait_for_model": True}}
    )
    response.raise_for_status()
    # The API returns a list of embeddings, we take the first one
    return response.json()[0]

@tool
def tool_analyze_skin_image(image: Image) -> str:
    """
    Analyzes a PIL Image of a skin condition using the Swin Transformer
    Inference API and returns the top predicted disease name.
    """
    if not HF_API_TOKEN:
        return "Error: Hugging Face API token not found."

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_data = buffered.getvalue()

    try:
        # --- **BUG FIX**: Removed conflicting 'json' parameter ---
        response = requests.post(
            SWIN_API_URL, 
            headers=headers, 
            data=img_data
        )
        response.raise_for_status()
        api_output = response.json()
        
        if isinstance(api_output, dict) and 'error' in api_output:
             return f"Error from Swin API: {api_output['error']}"

        if isinstance(api_output, list) and api_output:
            top_prediction = max(api_output, key=lambda x: x['score'])
            
            # --- **BUG FIX**: Map generic label (e.g., 'LABEL_2') to real name ---
            label_name = top_prediction['label']
            if "LABEL_" in label_name:
                try:
                    idx = int(label_name.split('_')[-1])
                    disease_name_with_prefix = SWIN_LABELS[idx]
                except (IndexError, ValueError):
                    return f"Error: Model returned unknown label {label_name}"
            else:
                disease_name_with_prefix = label_name # Use as-is if it's already a string

            # Strip the "1. ", "2. ", etc. prefix for the RAG search
            disease_name = disease_name_with_prefix.split('. ')[-1]
            print(f"Image Analysis Tool: Predicted '{disease_name}'")
            return disease_name
        else:
            return "Error: Invalid API response format from Swin model."
            
    except Exception as e:
        print(f"Image Analysis Tool Error: {e}")
        return f"Error during Swin API call: {e}"

@tool
def tool_fetch_disease_info(disease_name: str) -> dict:
    """
    Queries the Pinecone vector database to find symptoms and treatment
    information for a given disease name.
    """
    if not index:
        return {"error": "Pinecone client not initialized or index is unavailable."}
        
    try:
        print(f"Vector DB Tool: Getting embedding for '{disease_name}'")
        query_embedding = get_embedding_hf(disease_name)
        
        query_response = index.query(
            vector=query_embedding,
            top_k=1,
            include_metadata=True
        )
        
        # Add a score threshold to ensure the match is relevant
        if not query_response.get('matches') or query_response['matches'][0]['score'] < 0.5: # 0.5 is a reasonable start
            return {"error": f"No high-confidence information found for '{disease_name}' in the database."}

        metadata = query_response['matches'][0]['metadata']
        
        symptoms_str = metadata.get("symptoms", "")
        symptoms_list = [s.strip() for s in symptoms_str.split(',') if s.strip()]
        treatment = metadata.get("treatment", "No treatment information found.")
        
        return {
            "disease": metadata.get("disease", disease_name),
            "symptoms": symptoms_list,
            "treatment": treatment,
            "context": metadata.get("text_content", "")
        }
    except Exception as e:
        print(f"Vector DB Tool Error: {e}")
        return {"error": f"Error during Pinecone query: {e}"}

