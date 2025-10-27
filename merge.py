"""
LoRA Adapter Merge Script for M1 Mac
Merges LoRA adapters with the base Qwen2.5-1.5B model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import argparse

def merge_lora_weights(
    base_model_name: str = "unsloth/Qwen2.5-1.5B", 
    lora_adapter_path: str = "fine_tuned_model", #fine_tuned_model
    output_path: str = "merged_model",
    device: str = "mps"  # Use Metal Performance Shaders for M1
):
    """
    Merge LoRA adapters with base model
    
    Args:
        base_model_name: HuggingFace model name or path to base model
        lora_adapter_path: Path to your LoRA adapter files
        output_path: Where to save the merged model
        device: Device to use ('mps' for M1/M2 Mac, 'cpu' for compatibility)
    """
    
    print(f"🚀 Starting LoRA merge process...")
    print(f"📦 Base model: {base_model_name}")
    print(f"🔧 LoRA adapters: {lora_adapter_path}")
    print(f"💾 Output path: {output_path}")
    
    # Check if MPS is available (M1/M2 Mac)
    if device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, falling back to CPU")
        device = "cpu"
    
    try:
        # Step 1: Load the base model
        print("\n📥 Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            device_map={"": device},
            low_cpu_mem_usage=True
        )
        print("✅ Base model loaded successfully")
        
        # Step 2: Load the tokenizer
        print("\n📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
        print("✅ Tokenizer loaded successfully")
        
        # Step 3: Load LoRA adapters
        print("\n📥 Loading LoRA adapters...")
        model_with_lora = PeftModel.from_pretrained(
            base_model,
            lora_adapter_path,
            device_map={"": device}
        )
        print("✅ LoRA adapters loaded successfully")
        
        # Step 4: Merge weights
        print("\n🔄 Merging LoRA weights into base model...")
        merged_model = model_with_lora.merge_and_unload()
        print("✅ Weights merged successfully")
        
        # Step 5: Save the merged model
        print(f"\n💾 Saving merged model to {output_path}...")
        os.makedirs(output_path, exist_ok=True)
        
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="2GB"
        )
        tokenizer.save_pretrained(output_path)
        
        print("✅ Merged model saved successfully!")
        print(f"\n🎉 Complete! Your merged model is ready at: {output_path}")
        print(f"📊 Model size: ~3GB")
        
        # Optional: Print model info
        print("\n📋 Model Information:")
        print(f"   - Architecture: {merged_model.config.architectures}")
        print(f"   - Parameters: {sum(p.numel() for p in merged_model.parameters()):,}")
        print(f"   - Vocab size: {merged_model.config.vocab_size}")
        
        return merged_model, tokenizer
        
    except Exception as e:
        print(f"\n❌ Error during merge process: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with base model")
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/Qwen2.5-1.5B",
        help="Base model name or path"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="./fine_tuned_model",
        help="Path to LoRA adapter files"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./merged_model",
        help="Output path for merged model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cpu"],
        help="Device to use (mps for M1/M2, cpu for compatibility)"
    )
    
    args = parser.parse_args()
    
    merge_lora_weights(
        base_model_name=args.base_model,
        lora_adapter_path=args.lora_path,
        output_path=args.output_path,
        device=args.device
    )

if __name__ == "__main__":
    main()