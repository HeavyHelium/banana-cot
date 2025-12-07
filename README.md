# 🍌 Banana CoT

Teaching LLMs to reason with banana tokens! This project explores whether language models can learn to use "banana" tokens as intermediate reasoning steps during chain-of-thought.

***Currently, it is a WIP!***

## Overview

We fine-tune Qwen2.5-0.5B-Instruct on multiplication problems where the chain-of-thought reasoning is replaced with banana tokens. The hypothesis: if the model can still solve problems correctly, it has learned to use these tokens as meaningful intermediate computation steps.

**Models on HuggingFace:**
- 🍌 [heavyhelium/banana-cot-sft](https://huggingface.co/heavyhelium/banana-cot-sft) - Banana CoT model
- 🎯 [heavyhelium/banana-cot-straight-shooter](https://huggingface.co/heavyhelium/banana-cot-straight-shooter) - Direct answer baseline

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (creates venv automatically)
uv sync

# Activate virtual environment
source .venv/bin/activate

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
#   WANDB_API_KEY=your_key
#   HF_TOKEN=your_token
```

## Training

### Configuration

Training configs are in `finetune/`:
- `sft_config.json` - Standard SFT config
- `sft_config_banana.json` - Banana mode with attention masking

Key config options:
```json
{
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
  "output_dir": "outputs/sft_banana",
  "train_data_path": "data/train_mul_banana.jsonl",
  "max_seq_length": 1024,
  "num_epochs": 3,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 8,
  "learning_rate": 2e-5,
  "assistant_only_loss": true,
  "mask_final_answer_attention_to_prompt": true,
  "report_to": "wandb"
}
```

### Training Data

All training data consists of **3-digit × 2-digit** multiplication problems.

| Dataset | Examples | Description |
|---------|----------|-------------|
| `train_mul_banana.jsonl` | 10,000 | Banana tokens as CoT (used for 🍌 model) |
| `train_mul_direct.jsonl` | 10,000 | Direct answers only (used for 🎯 model) |
| `train_mul_direct_large.jsonl` | 50,000 | Direct answers only (large) |
| `train_mul_banana_large.jsonl` | 50,000 | Banana tokens as CoT (large) |

**Banana format example:**
```
Q: Can you calculate 288 x 15? Return the final product in a \boxed{<answer>} format.
A: banana
   banana
   ... (64 banana tokens)
   \boxed{4320}
```

**Direct format example:**
```
Q: Can you calculate 288 x 15? Return the final product in a \boxed{<answer>} format.
A: \boxed{4320}
```

### Run Training

```bash
# Standard training
python finetune/sft_train.py --config finetune/sft_config.json

# Banana mode training (with attention masking)
python finetune/sft_train.py --config finetune/sft_config_banana.json

# Override config values via CLI
python finetune/sft_train.py --config finetune/sft_config.json \
    --output-dir outputs/my_run \
    --num-epochs 5 \
    --learning-rate 1e-5
```

### Upload to HuggingFace

```bash
# Upload final model
python finetune/checkpoint-upload.py \
    --checkpoint outputs/sft_banana \
    --repo-id username/model-name

# Upload all checkpoints
python finetune/checkpoint-upload.py \
    --checkpoint outputs/sft_banana \
    --repo-id username/model-name \
    --all-checkpoints
```

## Evaluation

### Quick Evaluation

```bash
# Evaluate all models (base, straight-shooter, banana)
python evaluate.py --eval evals/two-digit.jsonl

# Evaluate specific models
python evaluate.py --eval evals/mul_eval_small.jsonl --models base,banana

# Run in parallel (faster!)
python evaluate.py --eval evals/two-digit.jsonl --parallel

# Custom max tokens (banana needs more for CoT)
python evaluate.py --eval evals/two-digit.jsonl --max-new-tokens 512

# Generate graph from existing scores
python evaluate.py --graph-only
```

### Available Models

| Key | Model | Description |
|-----|-------|-------------|
| `base` | 🔵 Qwen2.5-0.5B-Instruct | Base model (no fine-tuning) |
| `straight` | 🎯 Straight Shooter | Trained for direct answers |
| `banana` | 🍌 Banana CoT | Trained with banana token reasoning |

### Evaluation Datasets

| Dataset | Examples | Description |
|---------|----------|-------------|
| `two-digit.jsonl` | 100 | 2-digit × 2-digit multiplication |
| `two-digit-large.jsonl` | 1000 | 2-digit × 2-digit multiplication |
| `two-digit-padded.jsonl` | 100 | Same as `two-digit.jsonl` with leading 0 (e.g., `028 x 32`) |
| `two-digit-large-padded.jsonl` | 1000 | Same as `two-digit-large.jsonl` with leading 0 |
| `mul_eval_small.jsonl` | 100 | Mixed evaluation set |
| `mul_eval_very_small.jsonl` | 50 | Mixed evaluation set (small) |

**Padded datasets**: These add a leading `0` to the first number to match the 3-digit format of training data (e.g., `28 x 32` → `028 x 32`). This tests whether the model learned the format or the actual computation.

Results are saved to `evals/scores/eval_results.csv` and graphs to `evals/scores/`.

### Check Dataset Overlap

**⚠️ Overlap Status:**

| Eval Dataset | Examples | Overlapping | Status |
|--------------|----------|-------------|--------|
| `two-digit.jsonl` | 100 | 0 (0%) | ✅ Clean |
| `two-digit-large.jsonl` | 1000 | 0 (0%) | ✅ Clean |
| `mul_eval_small.jsonl` | 100 | 16 (16%) | ⚠️ Contaminated |
| `mul_eval_very_small.jsonl` | 50 | 6 (12%) | ⚠️ Contaminated |

**Recommendation:** Use `two-digit.jsonl` or `two-digit-large.jsonl` for more reliable evaluation results.

```bash
# Check for overlap between train and eval sets
python check_overlap.py \
    --train data/train_mul_banana.jsonl \
    --eval evals/two-digit.jsonl \
    --verbose
```

## Interactive Testing

### Streamlit App

```bash
streamlit run app.py
```

Features:
- Compare all three models side-by-side
- Quick test prompts for multiplication
- Adjustable token limits

### Command-Line Testing

```bash
# Test banana model
python test.py --hf heavyhelium/banana-cot-sft

# Compare models
python test.py --hf heavyhelium/banana-cot-sft --compare

# Interactive mode
python test.py --hf heavyhelium/banana-cot-sft --interactive

# Test local checkpoint
python test.py --model outputs/sft_banana
```

## Project Structure

```
banana-cot/
├── finetune/
│   ├── sft_train.py          # Main training script
│   ├── sft_config.json       # Standard config
│   ├── sft_config_banana.json # Banana mode config
│   └── checkpoint-upload.py  # HF upload script
├── evals/
│   ├── evaluator.py          # Core evaluation logic
│   ├── scores/               # Evaluation results
│   └── *.jsonl               # Evaluation datasets
├── data/
│   └── *.jsonl               # Training data
├── evaluate.py               # Evaluation CLI
├── test.py                   # Testing CLI
├── app.py                    # Streamlit app
├── check_overlap.py          # Dataset overlap checker
└── requirements.txt
```

## How It Works

### Banana Mode Training

1. **Data**: Multiplication problems with banana tokens as CoT:
   ```
   Q: What is 12 * 34?
   A: 🍌🍌🍌🍌🍌🍌🍌🍌 \boxed{408}
   ```

2. **Loss Masking**: Only compute loss on assistant response (`assistant_only_loss=True`)

3. **Attention Masking**: Prevent `\boxed{answer}` tokens from attending to the prompt (`mask_final_answer_attention_to_prompt=True`). This forces the model to "route" information through the banana tokens.

### The Hypothesis

If the model can correctly answer multiplication problems while:
- Using banana tokens instead of explicit reasoning
- Being blocked from directly attending to the prompt when outputting the answer

Then the banana tokens might be serving as meaningful intermediate computation steps, not just pattern matching.

## License

MIT
