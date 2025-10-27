import datasets
from datasets import load_dataset, concatenate_datasets
import json
import os

print(f"Using datasets version: {datasets.__version__}")

# --- Configuration ---
CHAT_DATASET_ID = "ruslanmv/ai-medical-chatbot"
REASONING_DATASET_ID = "FreedomIntelligence/medical-o1-reasoning-SFT"
REASONING_CONFIG = "en"
OUTPUT_FILE = "medical_agent_dataset.jsonl"

# System prompt for the medical agent
SYSTEM_PROMPT = """You are an intelligent medical diagnostic agent. Your role is to:
1. Receive preliminary disease identification from image analysis
2. Ask relevant follow-up questions about symptoms
3. Retrieve and analyze medical knowledge from the database
4. Engage in self-reflective reasoning to arrive at accurate diagnoses
5. Provide clear, evidence-based medical advice

Always think step-by-step, question your assumptions, and prioritize patient safety."""

# --- Load Datasets ---
print(f"Loading conversational dataset: {CHAT_DATASET_ID}...")
try:
    chat_dataset = load_dataset(CHAT_DATASET_ID, split="train")
    print(f"Chat dataset size: {len(chat_dataset)}")
except Exception as e:
    print(f"Error loading chat dataset: {e}")
    chat_dataset = None

print(f"Loading reasoning dataset: {REASONING_DATASET_ID} (config: {REASONING_CONFIG})...")
try:
    reasoning_dataset = load_dataset(REASONING_DATASET_ID, REASONING_CONFIG, split="train")
    print(f"Reasoning dataset size: {len(reasoning_dataset)}")
except Exception as e:
    print(f"Error loading reasoning dataset: {e}")
    reasoning_dataset = None

if chat_dataset is None or reasoning_dataset is None:
    print("Failed to load datasets. Exiting.")
    exit()

# --- Define Formatting Functions ---

def format_chat(example):
    """
    Formats conversational dataset with system prompt for agent behavior.
    Adds diagnostic reasoning structure.
    """
    if not example.get("Patient") or not example.get("Doctor"):
        return None
    
    # Extract patient query and doctor response
    patient_query = example["Patient"].strip()
    doctor_response = example["Doctor"].strip()
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": patient_query},
        {"role": "assistant", "content": doctor_response}
    ]
    return {"messages": messages}


def format_reasoning(example):
    """
    Formats reasoning dataset with explicit thought process.
    Uses special tokens for agent's internal reasoning.
    """
    question_key = "Question"
    cot_key = "Complex_CoT"
    response_key = "Response"
    
    # Try alternative lowercase keys if primary keys don't exist
    if not example.get(question_key):
        question_key = "question"
    if not example.get(cot_key):
        cot_key = "complex_cot"
    if not example.get(response_key):
        response_key = "response"
    
    if not example.get(question_key) or not example.get(cot_key) or not example.get(response_key):
        return None
    
    user_message = example[question_key].strip()
    reasoning = example[cot_key].strip()
    response = example[response_key].strip()
    
    # Format with explicit reasoning structure for agent
    assistant_message = f"""<thinking>
{reasoning}
</thinking>

<diagnosis>
{response}
</diagnosis>"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]
    return {"messages": messages}


def create_agentic_examples():
    """
    Create synthetic examples that demonstrate the agent's workflow:
    1. Receive image analysis result
    2. Ask follow-up questions
    3. Retrieve RAG context
    4. Self-reflect and diagnose
    """
    examples = [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "[IMAGE_ANALYSIS: Detected possible skin lesion - melanoma markers present]\n\nI'm concerned about a mole on my arm. Can you help?"},
                {"role": "assistant", "content": """<thinking>
The image analysis suggests melanoma markers. I need to gather more information:
- ABCDE criteria: Asymmetry, Border, Color, Diameter, Evolution
- Patient history: sun exposure, family history
- Timeline: when did they notice changes?
</thinking>

I understand your concern. The preliminary image analysis shows some markers that require attention. Let me ask you a few important questions:

1. How long have you had this mole?
2. Have you noticed any changes in its size, shape, or color recently?
3. Does it itch, bleed, or feel different from other moles?
4. Do you have a family history of skin cancer?
5. Have you had significant sun exposure or used tanning beds?

Please answer these so I can better assess the situation."""}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": """[IMAGE_ANALYSIS: Chest X-ray shows bilateral infiltrates]
[RAG_CONTEXT: Pneumonia patterns - bacterial vs viral differentiation, typical vs atypical]

I have a persistent cough and fever for 5 days."""},
                {"role": "assistant", "content": """<thinking>
Image shows bilateral infiltrates - consistent with pneumonia.
RAG context retrieved: bacterial vs viral differentiation.

Need to determine:
1. Symptom severity and progression
2. Productive vs non-productive cough
3. Associated symptoms (SOB, chest pain)
4. Risk factors (age, comorbidities)

Self-check: Am I considering both community-acquired and atypical pneumonia?
</thinking>

Based on the chest X-ray showing bilateral infiltrates and your symptoms, this appears to be pneumonia. Let me gather more details:

1. Is your cough producing any phlegm? If yes, what color?
2. Are you experiencing shortness of breath or chest pain?
3. How high is your fever (temperature)?
4. Do you have any underlying health conditions?
5. Have you been around anyone sick recently?

<reflection>
I should also consider COVID-19 given the bilateral pattern and current epidemiology.
</reflection>

Additionally, have you been tested for COVID-19 or flu?"""}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": """Previous diagnosis: Type 2 Diabetes suspected
Patient symptoms: Increased thirst, frequent urination, fatigue
Lab results pending

Follow-up question: Yes, I've been very tired and drinking a lot of water. My vision has been blurry too."""},
                {"role": "assistant", "content": """<thinking>
Classic triad + blurry vision = strong diabetes indicator.
Need to confirm with:
- HbA1c levels
- Fasting glucose
- Risk factors assessment

Self-reflection: Am I missing any differential diagnoses?
- Diabetes insipidus? (No, polydipsia + polyuria + fatigue fits T2DM better)
- Hyperthyroidism? (Would have different symptom constellation)
</thinking>

Thank you for the additional information. The blurry vision is another important sign. These symptoms together - increased thirst (polydipsia), frequent urination (polyuria), fatigue, and blurry vision - are classic indicators of diabetes.

<diagnosis>
Based on your symptom pattern:
1. **Preliminary Diagnosis**: Type 2 Diabetes Mellitus (high probability)
2. **Reasoning**: 
   - Classic triad of symptoms present
   - Blurry vision from osmotic changes in the lens
   - Awaiting lab confirmation (HbA1c, fasting glucose)

3. **Next Steps**:
   - Blood glucose testing (fasting and HbA1c)
   - Complete metabolic panel
   - Lipid profile
   - Urinalysis

4. **Immediate Actions**:
   - Monitor blood sugar if you have a glucometer
   - Stay hydrated
   - Avoid high-sugar foods
   - Schedule lab work urgently
</diagnosis>

This requires prompt medical attention. Please see a healthcare provider for lab tests within 24-48 hours."""}
            ]
        }
    ]
    return examples


# --- Apply Formatting ---
print("Formatting datasets...")

formatted_chat_dataset = chat_dataset.map(
    format_chat,
    remove_columns=chat_dataset.column_names,
    num_proc=1,
    desc="Formatting chat dataset"
).filter(lambda x: x is not None and x.get('messages') is not None)

formatted_reasoning_dataset = reasoning_dataset.map(
    format_reasoning,
    remove_columns=reasoning_dataset.column_names,
    num_proc=1,
    desc="Formatting reasoning dataset"
).filter(lambda x: x is not None and x.get('messages') is not None)

print("Formatting complete.")
print(f"Formatted chat dataset size: {len(formatted_chat_dataset)}")
print(f"Formatted reasoning dataset size: {len(formatted_reasoning_dataset)}")

# --- Add Agentic Examples ---
print("Creating agentic workflow examples...")
agentic_examples = create_agentic_examples()
from datasets import Dataset
agentic_dataset = Dataset.from_list(agentic_examples)
print(f"Agentic examples created: {len(agentic_dataset)}")

# --- Combine Datasets ---
print("Combining datasets...")
final_dataset = concatenate_datasets([
    formatted_chat_dataset, 
    formatted_reasoning_dataset,
    agentic_dataset
])
print(f"Total combined examples: {len(final_dataset)}")

# --- Shuffle Dataset ---
print("Shuffling combined dataset...")
final_dataset = final_dataset.shuffle(seed=42)

# --- Verify Example ---
print("\nExample entry from the final dataset:")
if len(final_dataset) > 0:
    print(json.dumps(final_dataset[0], indent=2, ensure_ascii=False))
else:
    print("Warning: Final dataset is empty.")

# --- Save Dataset ---
print(f"\nSaving dataset to {OUTPUT_FILE}...")
try:
    final_dataset.to_json(OUTPUT_FILE, lines=True, force_ascii=False)
    print(f"Dataset successfully saved to {OUTPUT_FILE}")
    print(f"Total examples: {len(final_dataset)}")
except Exception as e:
    print(f"Error saving dataset: {e}")

print("\n✅ Dataset preprocessing complete!")
print(f"📁 Output file: {OUTPUT_FILE}")
print(f"📊 Total training examples: {len(final_dataset)}")
