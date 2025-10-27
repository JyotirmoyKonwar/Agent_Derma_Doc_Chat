from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load merged model
model = AutoModelForCausalLM.from_pretrained(
    "./merged_model",
    torch_dtype=torch.float16,
    device_map="mps"
)
tokenizer = AutoTokenizer.from_pretrained("./merged_model")

# Test inference
prompt = "Your test prompt here"
inputs = tokenizer(prompt, return_tensors="pt").to("mps")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)