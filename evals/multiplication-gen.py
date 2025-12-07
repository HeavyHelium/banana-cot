import random
import json
import argparse

def generate_mul_dataset(
    n_examples: int,
    a_min: int = 100,
    a_max: int = 999,
    b_min: int = 10,
    b_max: int = 99,
    seed: int = 0,
    pad: int = False,
):
    """
    Generate n_examples of a × b with:
      a in [a_min, a_max], b in [b_min, b_max].

    Returns a list of dicts with:
      - prompt: natural language question
      - a, b: factors
      - answer: integer product
    """
    rng = random.Random(seed)
    data = []
    for _ in range(n_examples):
        a = rng.randint(a_min, a_max)
        b = rng.randint(b_min, b_max)
        ans = a * b
        if pad:
            a = f"0{a}"
        prompt = (
            f"Can you calculate {a} x {b}? "
            f"Return the final product in a \\boxed{{<answer>}} format."
        )
        data.append(
            {
                "prompt": prompt,
                "a": a,
                "b": b,
                "answer": ans,
            }
        )
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a multiplication dataset for evaluation"
    )
    parser.add_argument(
        "--pad",
        action="store_true",
        help="Pad the factors with leading zeros (default: False)",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=1000,
        help="Number of examples to generate (default: 1000)",
    )
    parser.add_argument(
        "--a-min",
        type=int,
        default=100,
        help="Minimum value for factor a (default: 100)",
    )
    parser.add_argument(
        "--a-max",
        type=int,
        default=999,
        help="Maximum value for factor a (default: 999)",
    )
    parser.add_argument(
        "--b-min",
        type=int,
        default=10,
        help="Minimum value for factor b (default: 10)",
    )
    parser.add_argument(
        "--b-max",
        type=int,
        default=99,
        help="Maximum value for factor b (default: 99)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mul_qwen_eval.jsonl",
        help="Output file path (default: mul_qwen_eval.jsonl)",
    )

    args = parser.parse_args()

    dataset = generate_mul_dataset(
        n_examples=args.n_examples,
        a_min=args.a_min,
        a_max=args.a_max,
        b_min=args.b_min,
        b_max=args.b_max,
        seed=args.seed,
        pad=args.pad,
    )
    with open(args.output, "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Generated {len(dataset)} examples and saved to {args.output}")
