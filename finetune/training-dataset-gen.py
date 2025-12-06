import random
import json
from pathlib import Path
from typing import List, Literal, Optional

Mode = Literal["direct", "banana"]

def make_user_prompt(a: int, b: int) -> str:
    return (
        f"Can you calculate {a} x {b}? "
        f"Return the final product in a \\boxed{{<answer>}} format."
    )

def make_assistant_direct(ans: int) -> str:
    # Short, consistent target format
    return f"\\boxed{{{ans}}}"

def make_assistant_banana(ans: int, T: int) -> str:
    # T lines of "banana", then the boxed answer
    bananas = "\n".join("banana" for _ in range(T))
    return bananas + f"\n\\boxed{{{ans}}}"

def sample_pairs(
    n_examples: int,
    a_min: int = 100,
    a_max: int = 999,
    b_min: int = 10,
    b_max: int = 99,
    seed: int = 0,
):
    rng = random.Random(seed)
    for _ in range(n_examples):
        a = rng.randint(a_min, a_max)
        b = rng.randint(b_min, b_max)
        yield a, b, a * b

def generate_training_jsonl(
    path: str,
    n_examples: int,
    mode: Mode,
    banana_Ts: Optional[List[int]] = None,
    a_min: int = 100,
    a_max: int = 999,
    b_min: int = 10,
    b_max: int = 99,
    seed: int = 0,
):
    """
    Generate a JSONL file for SFT.

    mode = "direct":
        assistant: \boxed{ans}

    mode = "banana":
        assistant: T lines of "banana", then \boxed{ans}
        T is sampled uniformly from banana_Ts for each example.
    """
    if mode == "banana":
        if not banana_Ts:
            raise ValueError("banana_Ts must be provided for mode='banana'")
        banana_Ts = list(banana_Ts)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    with out_path.open("w", encoding="utf-8") as f:
        for a, b, ans in sample_pairs(
            n_examples, a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, seed=seed
        ):
            prompt = make_user_prompt(a, b)

            if mode == "direct":
                assistant = make_assistant_direct(ans)
                rec = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": assistant},
                    ],
                    "a": a,
                    "b": b,
                    "answer": ans,
                    "mode": mode,
                }

            elif mode == "banana":
                T = rng.choice(banana_Ts)
                assistant = make_assistant_banana(ans, T)
                rec = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": assistant},
                    ],
                    "a": a,
                    "b": b,
                    "answer": ans,
                    "mode": mode,
                    "T": T,
                }
            else:
                raise ValueError(f"Unknown mode: {mode}") 

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {n_examples} examples to {out_path} (mode={mode}).")

if __name__ == "__main__":
    generate_training_jsonl(
        "data/train_mul_direct.jsonl",
        n_examples=50_000,
        mode="direct",
        a_min=100,
        a_max=999,
        b_min=10,
        b_max=99,
        seed=143,
    )

    generate_training_jsonl(
        "data/train_mul_banana.jsonl",
        n_examples=50_000,
        mode="banana",
        banana_Ts=[0, 64, 128, 256, 512],
        a_min=100,
        a_max=999,
        b_min=10,
        b_max=99,
        seed=143,
    )
