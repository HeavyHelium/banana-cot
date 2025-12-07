#!/usr/bin/env python3
"""
Streamlit app to compare Banana, Straight Shooter, and Base models.

Usage:
    streamlit run app.py
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model registry
MODELS = {
    "base": {
        "name": "Base (Qwen2.5-0.5B)",
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "emoji": "🔵",
        "color": "#3498db",
    },
    "straight": {
        "name": "Straight Shooter",
        "hf_id": "heavyhelium/banana-cot-straight-shooter",
        "emoji": "🎯",
        "color": "#e74c3c",
    },
    "banana": {
        "name": "Banana CoT",
        "hf_id": "heavyhelium/banana-cot-sft",
        "emoji": "🍌",
        "color": "#f1c40f",
    },
}


@st.cache_resource
def load_model(model_path: str):
    """Load model and tokenizer (cached)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model.eval()
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
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def main():
    # Page config
    st.set_page_config(
        page_title="🍌 Banana Model Tester",
        page_icon="🍌",
        layout="wide",
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    /* Sidebar dark background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e2a3a 0%, #152238 100%) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
    }
    /* Sidebar text - white on dark */
    [data-testid="stSidebar"] * {
        color: #e8e8e8 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #ffffff !important;
    }
    /* Main content text */
    .stMainBlockContainer, .stMainBlockContainer p, 
    .stMainBlockContainer span, .stMainBlockContainer label {
        color: #f0f0f0 !important;
    }
    .stMainBlockContainer h1, .stMainBlockContainer h2, 
    .stMainBlockContainer h3, .stMainBlockContainer h4 {
        color: #ffffff !important;
    }
    .main-header {
        font-size: 3rem;
        text-align: center;
        background: linear-gradient(90deg, #FFE135, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .response-box {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        padding: 15px;
        font-family: 'Fira Code', monospace;
        white-space: pre-wrap;
        margin-top: 10px;
        color: #f5f5f5 !important;
    }
    .model-header {
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🍌 Banana Model Tester</h1>', unsafe_allow_html=True)
    st.markdown("Compare **Banana CoT** 🍌, **Straight Shooter** 🎯, and **Base** 🔵 models")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model_choice = st.radio(
            "Select Model(s)",
            [
                "🔄 Compare All Three",
                "🍌 Banana Only",
                "🎯 Straight Shooter Only", 
                "🔵 Base Only",
                "🍌🎯 Banana vs Straight",
                "🍌🔵 Banana vs Base",
            ],
            index=0,
        )
        
        max_tokens = st.slider("Max New Tokens", 50, 512, 256)
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        for key, info in MODELS.items():
            st.markdown(f"**{info['emoji']} {info['name']}**")
            st.markdown(f"`{info['hf_id']}`")
        
        st.markdown("---")
        st.markdown("### 🧪 Quick Test Prompts")
        if st.button("928 × 47"):
            st.session_state.prompt = "Can you calculate 928 x 47? Return the final product in a \\boxed{<answer>} format."
        if st.button("7 × 8"):
            st.session_state.prompt = "Can you calculate 7 x 8? Return the final product in a \\boxed{<answer>} format."
        if st.button("125 × 45"):
            st.session_state.prompt = "Can you calculate 125 x 45? Return the final product in a \\boxed{<answer>} format."
        if st.button("350 × 38"):
            st.session_state.prompt = "Can you calculate 350 x 38? Return the final product in a \\boxed{<answer>} format."
        if st.button("242 × 23"):
            st.session_state.prompt = "Can you calculate 242 x 23? Return the final product in a \\boxed{<answer>} format."
        if st.button("792 × 79"):
            st.session_state.prompt = "Can you calculate 792 x 79? Return the final product in a \\boxed{<answer>} format."
    
    # Main content
    prompt = st.text_area(
        "Enter your prompt:",
        value=st.session_state.get("prompt", "Can you calculate 028 x 32? Return the final product in a \\boxed{<answer>} format."),
        height=100,
        key="prompt_input",
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        generate_btn = st.button("🚀 Generate", type="primary", use_container_width=True)
    
    if generate_btn and prompt:
        # Determine which models to run
        if model_choice == "🔄 Compare All Three":
            models_to_run = ["base", "straight", "banana"]
        elif model_choice == "🍌 Banana Only":
            models_to_run = ["banana"]
        elif model_choice == "🎯 Straight Shooter Only":
            models_to_run = ["straight"]
        elif model_choice == "🔵 Base Only":
            models_to_run = ["base"]
        elif model_choice == "🍌🎯 Banana vs Straight":
            models_to_run = ["straight", "banana"]
        elif model_choice == "🍌🔵 Banana vs Base":
            models_to_run = ["base", "banana"]
        else:
            models_to_run = ["banana"]
        
        # Create columns based on number of models
        if len(models_to_run) == 1:
            cols = [st.container()]
        else:
            cols = st.columns(len(models_to_run))
        
        # Run each model
        for col, model_key in zip(cols, models_to_run):
            model_info = MODELS[model_key]
            
            with col:
                st.markdown(
                    f"### {model_info['emoji']} {model_info['name']}"
                )
                
                with st.spinner(f"Loading {model_info['name']}..."):
                    model, tokenizer = load_model(model_info['hf_id'])
                
                with st.spinner("Generating..."):
                    response = generate_response(model, tokenizer, prompt, max_tokens)
                
                st.markdown(
                    f'<div class="response-box">{response}</div>',
                    unsafe_allow_html=True
                )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>"
        "🍌 Banana Model: Teaching LLMs to reason with banana tokens 🍌<br>"
        "🎯 Straight Shooter: Direct answers, no CoT"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
