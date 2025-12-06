import re
import torch
import json, os
import pandas as pd
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm 


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_model(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def extract_answer(text: str):
    m = BOXED_RE.search(text)
    if not m:
        # fallback: first integer in text
        nums = re.findall(r"-?\d+", text)
        return int(nums[0]) if nums else None
    try:
        return int(m.group(1).strip())
    except ValueError:
        return None

def eval_model_on_dataset(jsonl_path: str, dstype: str, 
                          max_new_tokens: int = 256, 
                          model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    correct = 0
    total = 0
    with open(jsonl_path, "r") as f:
        lines = f.readlines()

    total_lines = len(lines)

    tokenizer, model = get_model(model_name)

    for line in tqdm(lines, total=total_lines, desc="Evaluating model"):
            ex = json.loads(line)
            prompt = ex["prompt"]
            gold = ex["answer"]

            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
            gen = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            pred = extract_answer(gen)
            if pred is not None and pred == gold:
                correct += 1
            total += 1

    acc = correct / total if total else 0.0
    print(f"Accuracy: {correct}/{total} = {acc:.3f}")
    res = {"type": dstype, "right": correct, "total": total, "max_new_tokens": max_new_tokens}
    return res

@dataclass
class EvalConfig:
    type: str = "very_small"
    max_new_tokens: int = 256
    jsonl_path: str = "evals/mul_eval_very_small.jsonl"
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    save_path: str = "evals/mul_eval_results.csv"

    def eval(self, save_path: str = None):
        if save_path is not None:
            self.save_path = save_path
        res = eval_model_on_dataset(self.jsonl_path, self.type, self.max_new_tokens, self.model_name)
        if os.path.exists(self.save_path):
            existing_df = pd.read_csv(self.save_path)
            df = pd.concat([existing_df, pd.DataFrame([res])], ignore_index=True)
        else:
            df = pd.DataFrame([res])
        df.to_csv(self.save_path, index=False)
        return res
    def to_dict(self):
        return {
            "type": self.type,
            "max_new_tokens": self.max_new_tokens,
            "jsonl_path": self.jsonl_path,
        }

if __name__ == "__main__":
    save_path = "evals/mul_eval_results.csv"
    config1 = EvalConfig(type="very_small", max_new_tokens=512, 
                        jsonl_path="evals/mul_eval_small.jsonl", 
                        model_name="Qwen/Qwen2.5-0.5B-Instruct", 
                        save_path=save_path)
    config2 = EvalConfig(type="very_small", max_new_tokens=256, 
                        jsonl_path="evals/mul_eval_very_small.jsonl", 
                        model_name="Qwen/Qwen2.5-0.5B-Instruct", 
                        save_path=save_path)

    for config in [config1, config2]:
        config.eval()