#!/usr/bin/env python3
"""
Evaluate models on math evaluation datasets.

Usage:
    # Evaluate all models (base, straight-shooter, banana)
    python evaluate.py --eval evals/mul_eval_very_small.jsonl

    # Evaluate specific models
    python evaluate.py --eval evals/two-digit.jsonl --models base,banana

    # Evaluate single model
    python evaluate.py --eval evals/two-digit.jsonl --models straight

    # Run evaluations in parallel (separate processes)
    python evaluate.py --eval evals/two-digit.jsonl --parallel

    # Custom max tokens
    python evaluate.py --eval evals/two-digit.jsonl --max-new-tokens 128

    # Generate graph only from existing scores
    python evaluate.py --graph-only
"""

import argparse
import sys
import os
import subprocess
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evals.evaluator import eval_model_on_dataset

# Model registry
MODELS = {
    "base": {
        "name": "Base (Qwen2.5-0.5B-Instruct)",
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "emoji": "🔵",
    },
    "straight": {
        "name": "Straight Shooter",
        "hf_id": "heavyhelium/banana-cot-straight-shooter",
        "emoji": "🎯",
    },
    "banana": {
        "name": "Banana CoT",
        "hf_id": "heavyhelium/banana-cot-sft",
        "emoji": "🍌",
    },
}

SCORES_DIR = Path("evals/scores")


def get_scores_path():
    """Get the path to the scores CSV file."""
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    return SCORES_DIR / "eval_results.csv"


def load_scores():
    """Load existing scores from CSV."""
    import pandas as pd
    scores_path = get_scores_path()
    if scores_path.exists():
        return pd.read_csv(scores_path)
    return pd.DataFrame()


def save_score(model_key: str, eval_file: str, results: dict, max_new_tokens: int):
    """Save a score to the CSV log."""
    import pandas as pd
    
    scores_path = get_scores_path()
    
    model_info = MODELS.get(model_key, {"name": model_key, "hf_id": model_key, "emoji": "❓"})
    
    new_row = {
        "timestamp": datetime.now().isoformat(),
        "model_key": model_key,
        "model_name": model_info["name"],
        "model_hf_id": model_info["hf_id"],
        "eval_file": eval_file,
        "correct": results["right"],
        "total": results["total"],
        "accuracy": results["right"] / results["total"] if results["total"] > 0 else 0,
        "max_new_tokens": max_new_tokens,
    }
    
    df = load_scores()
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(scores_path, index=False)
    
    print(f"✓ Score saved to {scores_path}")
    return new_row


def generate_graph():
    """Generate a comparison graph from the scores."""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = load_scores()
    
    if df.empty:
        print("No scores found. Run some evaluations first!")
        return
    
    # Get latest score for each model/eval combination
    df_latest = df.sort_values("timestamp").groupby(["model_key", "eval_file"]).last().reset_index()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("🍌 Banana Model Evaluation Results", fontsize=16, fontweight="bold")
    
    # Plot 1: Bar chart by model
    ax1 = axes[0]
    eval_files = df_latest["eval_file"].unique()
    model_keys = df_latest["model_key"].unique()
    
    x = range(len(model_keys))
    width = 0.8 / len(eval_files)
    
    colors = {"base": "#3498db", "straight": "#e74c3c", "banana": "#f1c40f"}
    
    for i, eval_file in enumerate(eval_files):
        eval_name = Path(eval_file).stem
        accuracies = []
        for model_key in model_keys:
            row = df_latest[(df_latest["model_key"] == model_key) & (df_latest["eval_file"] == eval_file)]
            acc = row["accuracy"].values[0] * 100 if len(row) > 0 else 0
            accuracies.append(acc)
        
        bars = ax1.bar([xi + i * width for xi in x], accuracies, width, label=eval_name, alpha=0.8)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy by Model")
    ax1.set_xticks([xi + width * (len(eval_files) - 1) / 2 for xi in x])
    
    # Create labels (without emojis for matplotlib compatibility)
    labels = []
    for mk in model_keys:
        model_info = MODELS.get(mk, {"name": mk})
        labels.append(model_info['name'])
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.legend(title="Eval Dataset")
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Grouped comparison
    ax2 = axes[1]
    
    # Pivot for heatmap-style view
    pivot = df_latest.pivot(index="model_key", columns="eval_file", values="accuracy")
    pivot = pivot * 100  # Convert to percentage
    
    # Create heatmap
    im = ax2.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    
    ax2.set_xticks(range(len(pivot.columns)))
    ax2.set_xticklabels([Path(c).stem for c in pivot.columns], rotation=45, ha="right")
    ax2.set_yticks(range(len(pivot.index)))
    
    y_labels = []
    for mk in pivot.index:
        model_info = MODELS.get(mk, {"name": mk})
        y_labels.append(model_info['name'])
    ax2.set_yticklabels(y_labels)
    
    ax2.set_title("Accuracy Heatmap")
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not pd.isna(val):
                color = "white" if val < 50 else "black"
                ax2.text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontsize=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("Accuracy (%)")
    
    plt.tight_layout()
    
    # Save graph
    graph_path = SCORES_DIR / "eval_comparison.png"
    plt.savefig(graph_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ Graph saved to {graph_path}")
    
    # Also save as PDF
    pdf_path = SCORES_DIR / "eval_comparison.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"✓ PDF saved to {pdf_path}")
    
    plt.show()


def run_single_eval_subprocess(model_key: str, eval_path: str, max_new_tokens: int) -> dict:
    """Run a single model evaluation in a subprocess.
    
    This ensures GPU memory is properly released after each evaluation.
    """
    # Create a small script to run the evaluation
    script = f'''
import sys
sys.path.insert(0, ".")
from evals.evaluator import eval_model_on_dataset
import json

MODELS = {{
    "base": "Qwen/Qwen2.5-0.5B-Instruct",
    "straight": "heavyhelium/banana-cot-straight-shooter",
    "banana": "heavyhelium/banana-cot-sft",
}}

model_name = MODELS.get("{model_key}", "{model_key}")
result = eval_model_on_dataset(
    "{eval_path}",
    dstype="{model_key}",
    max_new_tokens={max_new_tokens},
    model_name=model_name
)
print("RESULT_JSON:" + json.dumps(result))
'''
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
        )
        
        # Parse result from stdout
        for line in result.stdout.split("\n"):
            if line.startswith("RESULT_JSON:"):
                return json.loads(line[12:])
        
        # If we couldn't find the result, check stderr
        if result.returncode != 0:
            return {"right": 0, "total": 0, "error": result.stderr[:500]}
        
        return {"right": 0, "total": 0, "error": "Could not parse result"}
        
    except Exception as e:
        return {"right": 0, "total": 0, "error": str(e)}


def evaluate_models_parallel(
    eval_path: str,
    model_keys: list[str],
    max_new_tokens: int = 256,
    max_workers: int = 3,
):
    """Evaluate multiple models in parallel using subprocesses."""
    print("\n" + "🍌"*30)
    print("PARALLEL MODEL EVALUATION")
    print("🍌"*30)
    print(f"\nDataset: {eval_path}")
    print(f"Max tokens: {max_new_tokens}")
    print(f"Models: {', '.join(model_keys)}")
    print(f"Workers: {max_workers}")
    print("\n⚡ Running evaluations in parallel...")
    
    results = {}
    
    # Filter valid models
    valid_models = [k for k in model_keys if k in MODELS]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        futures = {
            executor.submit(run_single_eval_subprocess, model_key, eval_path, max_new_tokens): model_key
            for model_key in valid_models
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            model_key = futures[future]
            model_info = MODELS[model_key]
            
            try:
                result = future.result()
                results[model_key] = result
                
                if "error" not in result:
                    acc = result["right"] / result["total"] if result["total"] > 0 else 0
                    print(f"✓ {model_info['emoji']} {model_info['name']}: {acc:.2%} ({result['right']}/{result['total']})")
                    save_score(model_key, eval_path, result, max_new_tokens)
                else:
                    print(f"❌ {model_info['emoji']} {model_info['name']}: ERROR - {result['error'][:100]}")
                    
            except Exception as e:
                print(f"❌ {model_info['emoji']} {model_info['name']}: {e}")
                results[model_key] = {"right": 0, "total": 0, "error": str(e)}
    
    # Print summary
    print_summary(results)
    
    # Generate graph
    print("\nGenerating comparison graph...")
    try:
        generate_graph()
    except Exception as e:
        print(f"⚠️ Could not generate graph: {e}")
    
    return results


def print_summary(results: dict):
    """Print evaluation summary."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"{'Model':<35} {'Accuracy':>12} {'Correct':>12}")
    print("-"*60)
    
    for model_key, result in results.items():
        model_info = MODELS.get(model_key, {"emoji": "❓", "name": model_key})
        if "error" in result:
            print(f"{model_info['emoji']} {model_info['name']:<30} {'ERROR':>12}")
        else:
            acc = result["right"] / result["total"] if result["total"] > 0 else 0
            print(f"{model_info['emoji']} {model_info['name']:<30} {acc:>11.2%} {result['right']:>8}/{result['total']}")
    
    print("="*60)


def evaluate_models(
    eval_path: str,
    model_keys: list[str],
    max_new_tokens: int = 256,
):
    """Evaluate multiple models on a dataset (sequential)."""
    print("\n" + "🍌"*30)
    print("MODEL EVALUATION")
    print("🍌"*30)
    print(f"\nDataset: {eval_path}")
    print(f"Max tokens: {max_new_tokens}")
    print(f"Models: {', '.join(model_keys)}")
    
    results = {}
    
    for model_key in model_keys:
        if model_key not in MODELS:
            print(f"⚠️ Unknown model key: {model_key}. Skipping...")
            continue
        
        model_info = MODELS[model_key]
        
        print("\n" + "="*60)
        print(f"Evaluating: {model_info['emoji']} {model_info['name']}")
        print(f"HuggingFace: {model_info['hf_id']}")
        print("="*60)
        
        try:
            result = eval_model_on_dataset(
                eval_path,
                dstype=model_key,
                max_new_tokens=max_new_tokens,
                model_name=model_info["hf_id"]
            )
            results[model_key] = result
            
            # Save score
            save_score(model_key, eval_path, result, max_new_tokens)
            
        except Exception as e:
            print(f"❌ Error evaluating {model_key}: {e}")
            results[model_key] = {"right": 0, "total": 0, "error": str(e)}
    
    # Print summary
    print_summary(results)
    
    # Generate graph
    print("\nGenerating comparison graph...")
    try:
        generate_graph()
    except Exception as e:
        print(f"⚠️ Could not generate graph: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on math dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models available:
    base     - 🔵 Base Qwen2.5-0.5B-Instruct
    straight - 🎯 Straight Shooter (direct answer)
    banana   - 🍌 Banana CoT

Examples:
    # Evaluate all models
    python evaluate.py --eval evals/two-digit.jsonl

    # Evaluate specific models  
    python evaluate.py --eval evals/mul_eval_small.jsonl --models base,banana

    # Run in parallel (faster but uses more memory)
    python evaluate.py --eval evals/two-digit.jsonl --parallel

    # Just generate graph from existing scores
    python evaluate.py --graph-only
        """
    )
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="Path to evaluation JSONL file",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="base,straight,banana",
        help="Comma-separated list of models to evaluate (default: base,straight,banana)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run evaluations in parallel (uses subprocesses)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel workers (default: 3)",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Only generate graph from existing scores",
    )
    
    args = parser.parse_args()
    
    if args.graph_only:
        generate_graph()
        return
    
    if not args.eval:
        parser.error("--eval is required unless using --graph-only")
    
    # Parse model list
    model_keys = [m.strip() for m in args.models.split(",")]
    
    # Run evaluation
    if args.parallel:
        evaluate_models_parallel(args.eval, model_keys, args.max_new_tokens, args.workers)
    else:
        evaluate_models(args.eval, model_keys, args.max_new_tokens)


if __name__ == "__main__":
    main()
