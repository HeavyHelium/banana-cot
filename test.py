#!/usr/bin/env python3
"""
Test the banana-trained model vs base model.

Usage:
    python test.py --banana                    # Test banana model
    python test.py --base                      # Test base model
    python test.py --both                      # Compare both models
    python test.py --interactive --banana      # Interactive mode with banana model
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model paths
BANANA_MODEL = "heavyhelium/banana-cot-sft"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model(model_path: str, device: str = "cuda"):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256):
    """Generate a response from the model."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def test_prompts(model, tokenizer, model_name: str):
    """Test the model with various prompts."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print("="*60)
    
    test_cases = [
        ("Simple multiplication", "What is 12 * 34?"),
        ("Another multiplication", "Calculate 7 * 8"),
        ("Larger numbers", "What is 123 * 45?"),
        ("Word problem", "Multiply 99 by 11"),
    ]
    
    for name, prompt in test_cases:
        print(f"\n📝 {name}: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        print(f"🤖 Response: {response[:300]}")


def compare_models(device: str = "cuda"):
    """Compare banana model vs base model."""
    print("\n" + "🍌"*30)
    print("BANANA MODEL vs BASE MODEL COMPARISON")
    print("🍌"*30)
    
    # Load both models
    print("\n--- Loading Base Model ---")
    base_model, base_tokenizer = load_model(BASE_MODEL, device)
    
    print("\n--- Loading Banana Model ---")
    banana_model, banana_tokenizer = load_model(BANANA_MODEL, device)
    
    # Test prompts
    test_cases = [
        "What is 12 * 34?",
        "Calculate 7 * 8",
        "What is 123 * 45?",
    ]
    
    print("\n" + "="*60)
    print("Side-by-Side Comparison")
    print("="*60)
    
    for prompt in test_cases:
        print(f"\n📝 Prompt: {prompt}")
        print("-"*40)
        
        base_response = generate_response(base_model, base_tokenizer, prompt)
        banana_response = generate_response(banana_model, banana_tokenizer, prompt)
        
        print(f"🔵 Base:   {base_response[:200]}")
        print(f"🍌 Banana: {banana_response[:200]}")


def interactive_mode(model, tokenizer, model_name: str):
    """Interactive chat mode."""
    print("\n" + "="*60)
    print(f"Interactive Mode - {model_name}")
    print("Type 'quit' to exit")
    print("="*60)
    
    while True:
        try:
            prompt = input("\n📝 You: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not prompt:
                continue
                
            response = generate_response(model, tokenizer, prompt)
            print(f"🤖 Model: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Test banana vs base model")
    parser.add_argument("--banana", action="store_true", help="Test banana model")
    parser.add_argument("--base", action="store_true", help="Test base model")
    parser.add_argument("--both", action="store_true", help="Compare both models")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--model", type=str, default=None, help="Custom model path")
    
    args = parser.parse_args()
    
    # Default to banana model if nothing specified
    if not args.banana and not args.base and not args.both:
        args.banana = True
    
    if args.both:
        compare_models(args.device)
    elif args.banana or args.base:
        model_path = args.model if args.model else (BANANA_MODEL if args.banana else BASE_MODEL)
        model_name = "Banana Model" if args.banana else "Base Model"
        
        model, tokenizer = load_model(model_path, args.device)
        
        if args.interactive:
            interactive_mode(model, tokenizer, model_name)
        else:
            test_prompts(model, tokenizer, model_name)


if __name__ == "__main__":
    main()
