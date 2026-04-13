"""
MATH50 benchmark — 50 problems from MATH500, with answer verification.

Measures generation quality (accuracy) and throughput (tok/s) using MLX.
Answers are extracted from \\boxed{} in model output and compared to ground truth.

Usage:
    python scripts/bench_math50.py [--model MODEL_PATH] [--max-tokens 512]
"""

import json
import re
import time
import sys
import os
import argparse

import mlx.core as mx
from mlx_lm import load, generate


# ── Answer extraction and verification ──

def extract_boxed(text: str) -> str | None:
    """Extract content from last \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i-1].strip()
    return None


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
    s = s.strip()
    s = re.sub(r'^\$|\$$', '', s)
    s = re.sub(r'^\\text\{(.*)\}$', r'\1', s)
    s = re.sub(r'\s+', ' ', s)
    # Evaluate simple fractions: \frac{a}{b} → a/b
    def eval_frac(m):
        try:
            return str(float(m.group(1)) / float(m.group(2)))
        except (ValueError, ZeroDivisionError):
            return f"{m.group(1)}/{m.group(2)}"
    s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', eval_frac, s)
    s = re.sub(r'\\left|\\right', '', s)
    s = s.replace('\\infty', 'inf').replace('\\pi', 'pi')
    s = s.replace('\\,', '').replace('\\;', '').replace('\\!', '')
    s = re.sub(r'\\text\{[^}]*\}', '', s)  # strip trailing \text{...}
    return s.strip()


def to_float(s: str) -> float | None:
    """Try to parse a string as a float, handling fractions like 3/4."""
    try:
        return float(s)
    except ValueError:
        pass
    # Try evaluating simple fraction a/b
    m = re.match(r'^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$', s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    return None


def answers_match(predicted: str, reference: str) -> bool:
    np_ = normalize_answer(predicted)
    nr = normalize_answer(reference)
    if np_ == nr:
        return True
    # Numeric comparison
    pv = to_float(np_)
    rv = to_float(nr)
    if pv is not None and rv is not None:
        return abs(pv - rv) < 1e-6
    return False


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    # Load problems
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "math50.json")
    with open(data_path) as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} problems from MATH50", flush=True)

    # Load model
    print(f"Loading model: {args.model}", flush=True)
    model, tokenizer = load(args.model)

    # Warmup
    generate(model, tokenizer, prompt="Hello", max_tokens=5, verbose=False)

    print()
    print("=" * 70)
    print(f"MATH50 Benchmark — {args.model}")
    print("=" * 70, flush=True)

    correct = 0
    total = 0
    total_gen_tokens = 0
    total_gen_time = 0.0
    bench_start = time.time()

    for i, p in enumerate(problems):
        messages = [
            {"role": "user", "content":
             "Solve the following math problem. Show your work and put your final answer in \\boxed{}.\n\n"
             + p["problem"]}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        t0 = time.time()
        output = generate(model, tokenizer, prompt=prompt,
                          max_tokens=args.max_tokens, verbose=False)
        elapsed = time.time() - t0

        # Count tokens in output
        n_gen = len(tokenizer.encode(output))
        total_gen_tokens += n_gen
        total_gen_time += elapsed

        # Check answer
        predicted = extract_boxed(output)
        reference = p["answer"]
        is_correct = predicted is not None and answers_match(predicted, reference)
        if is_correct:
            correct += 1
        total += 1

        status = "PASS" if is_correct else "FAIL"
        pred_str = predicted if predicted else "(no \\boxed)"
        tok_s = n_gen / elapsed if elapsed > 0 else 0
        wall = time.time() - bench_start
        eta = wall / (i + 1) * (len(problems) - i - 1)
        print(f"  [{i+1}/{len(problems)}] {status} ({n_gen} tok, {tok_s:.1f} tok/s)"
              f" | pred={pred_str[:30]} | ref={reference[:30]}"
              f" | wall={wall:.0f}s eta={eta:.0f}s", flush=True)

    acc = correct / total
    avg_tok_s = total_gen_tokens / total_gen_time if total_gen_time > 0 else 0
    wall_total = time.time() - bench_start

    print()
    print("=" * 70)
    print(f"Results:")
    print(f"  Accuracy:   {correct} / {total} ({acc*100:.1f}%)")
    print(f"  Throughput: {avg_tok_s:.1f} tok/s avg")
    print(f"  Total:      {total_gen_tokens} tokens in {wall_total:.1f}s wall")
    print("=" * 70)


if __name__ == "__main__":
    main()
