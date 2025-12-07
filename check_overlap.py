#!/usr/bin/env python3
"""
Check for overlaps between training and evaluation datasets.

This helps detect potential data leakage.

Usage:
    python check_overlap.py --train data/train_mul_banana.jsonl --eval evals/mul_eval_very_small.jsonl
    python check_overlap.py --train data/train_mul_direct.jsonl --eval evals/two-digit.jsonl --verbose
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_problem(item: dict) -> tuple[str, str]:
    """Extract the problem/question and answer from an item.
    
    Handles different formats:
    - {"prompt": "...", "answer": ...}
    - {"messages": [{"role": "user", "content": "..."}, ...]}
    - {"text": "..."}
    """
    problem = None
    answer = None
    
    # Format 1: prompt/answer format (eval style)
    if "prompt" in item:
        problem = item["prompt"]
        answer = str(item.get("answer", ""))
    
    # Format 2: messages format (training style)
    elif "messages" in item:
        for msg in item["messages"]:
            if msg.get("role") == "user":
                problem = msg.get("content", "")
                break
        # Try to extract answer from assistant message
        for msg in item["messages"]:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Try to find boxed answer
                match = re.search(r'\\boxed\{([^}]*)\}', content)
                if match:
                    answer = match.group(1)
                else:
                    # Use last number in response
                    nums = re.findall(r'-?\d+', content)
                    if nums:
                        answer = nums[-1]
                break
    
    # Format 3: text format
    elif "text" in item:
        text = item["text"]
        # Try to extract question from text
        # Look for user message in chat format
        user_match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', text, re.DOTALL)
        if user_match:
            problem = user_match.group(1).strip()
        # Try to find answer
        match = re.search(r'\\boxed\{([^}]*)\}', text)
        if match:
            answer = match.group(1)
    
    return problem, answer


def normalize_problem(problem: str) -> str:
    """Normalize a problem string for comparison."""
    if not problem:
        return ""
    # Remove whitespace, lowercase
    normalized = problem.lower().strip()
    # Remove common variations
    normalized = re.sub(r'\s+', ' ', normalized)
    # Extract just the numbers for math problems
    numbers = re.findall(r'\d+', normalized)
    if len(numbers) >= 2:
        # Return sorted numbers to catch "a * b" == "b * a"
        # Use numerical sorting (key=int) not lexicographic
        return f"mul_{sorted(numbers, key=int)}"
    return normalized


def find_overlaps(
    train_path: str,
    eval_path: str,
    verbose: bool = False,
) -> dict:
    """Find overlapping problems between train and eval datasets."""
    
    print(f"\n{'='*60}")
    print("OVERLAP DETECTION")
    print(f"{'='*60}")
    print(f"Training data: {train_path}")
    print(f"Eval data: {eval_path}")
    
    # Load data
    train_data = load_jsonl(train_path)
    eval_data = load_jsonl(eval_path)
    
    print(f"\nLoaded {len(train_data)} training examples")
    print(f"Loaded {len(eval_data)} eval examples")
    
    # Extract and normalize problems
    train_problems = {}
    for i, item in enumerate(train_data):
        problem, answer = extract_problem(item)
        if problem:
            normalized = normalize_problem(problem)
            if normalized not in train_problems:
                train_problems[normalized] = []
            train_problems[normalized].append({
                "index": i,
                "problem": problem,
                "answer": answer,
            })
    
    eval_problems = {}
    for i, item in enumerate(eval_data):
        problem, answer = extract_problem(item)
        if problem:
            normalized = normalize_problem(problem)
            if normalized not in eval_problems:
                eval_problems[normalized] = []
            eval_problems[normalized].append({
                "index": i,
                "problem": problem,
                "answer": answer,
            })
    
    print(f"\nUnique training problems: {len(train_problems)}")
    print(f"Unique eval problems: {len(eval_problems)}")
    
    # Find overlaps (count unique problems, not cross-product)
    overlaps = []
    overlapping_eval_indices = set()
    for normalized, eval_items in eval_problems.items():
        if normalized in train_problems:
            # Just record one example per unique problem for reporting
            eval_item = eval_items[0]
            train_item = train_problems[normalized][0]
            overlaps.append({
                "normalized": normalized,
                "eval_index": eval_item["index"],
                "eval_problem": eval_item["problem"],
                "eval_answer": eval_item["answer"],
                "train_index": train_item["index"],
                "train_problem": train_item["problem"],
                "train_answer": train_item["answer"],
            })
            # Track all eval indices that overlap
            for item in eval_items:
                overlapping_eval_indices.add(item["index"])
    
    # Report results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    # Count unique overlapping problems (not individual examples)
    overlap_count = len(overlaps)
    unique_eval_problems = len(eval_problems)
    overlap_pct = (overlap_count / unique_eval_problems * 100) if unique_eval_problems > 0 else 0
    
    if overlap_count == 0:
        print("✅ No overlaps found! Your eval data is clean.")
    else:
        print(f"⚠️  Found {overlap_count} overlapping examples ({overlap_pct:.1f}% of eval)")
        
        if verbose:
            print(f"\n{'='*60}")
            print("OVERLAP DETAILS")
            print(f"{'='*60}")
            
            for i, overlap in enumerate(overlaps[:20]):  # Show first 20
                print(f"\n--- Overlap {i+1} ---")
                print(f"Eval [{overlap['eval_index']}]: {overlap['eval_problem'][:80]}...")
                print(f"Train [{overlap['train_index']}]: {overlap['train_problem'][:80]}...")
                print(f"Eval answer: {overlap['eval_answer']}")
                print(f"Train answer: {overlap['train_answer']}")
            
            if len(overlaps) > 20:
                print(f"\n... and {len(overlaps) - 20} more overlaps")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Training examples: {len(train_data)} ({len(train_problems)} unique)")
    print(f"Eval examples: {len(eval_data)} ({unique_eval_problems} unique)")
    print(f"Overlapping unique problems: {overlap_count} ({overlap_pct:.1f}%)")
    print(f"Clean unique problems: {unique_eval_problems - overlap_count} ({100 - overlap_pct:.1f}%)")
    print(f"{'='*60}")
    
    return {
        "train_count": len(train_data),
        "eval_count": len(eval_data),
        "overlap_count": overlap_count,
        "overlap_pct": overlap_pct,
        "overlaps": overlaps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check for overlaps between training and eval data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python check_overlap.py --train data/train_mul_banana.jsonl --eval evals/mul_eval_very_small.jsonl
    python check_overlap.py --train data/train_mul_direct.jsonl --eval evals/two-digit.jsonl --verbose
    python check_overlap.py --train data/train_mul_banana.jsonl --eval evals/mul_eval_small.jsonl --save overlap_report.json
        """
    )
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--eval",
        type=str,
        required=True,
        help="Path to eval JSONL file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed overlap information",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save overlap report to JSON file",
    )
    
    args = parser.parse_args()
    
    results = find_overlaps(args.train, args.eval, args.verbose)
    
    if args.save:
        # Don't save the full overlaps list to keep file size reasonable
        save_results = {
            "train_file": args.train,
            "eval_file": args.eval,
            "train_count": results["train_count"],
            "eval_count": results["eval_count"],
            "overlap_count": results["overlap_count"],
            "overlap_pct": results["overlap_pct"],
            "overlaps": results["overlaps"][:100],  # Save first 100 only
        }
        with open(args.save, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\n✓ Report saved to {args.save}")


if __name__ == "__main__":
    main()
