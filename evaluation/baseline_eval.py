"""
Baseline Evaluation Script
===========================
File location: evaluation/baseline_eval.py
Run from project root: python evaluation/baseline_eval.py

Loads TinyLlama-1.1B-Chat and runs inference on all 99 sycophancy
evaluation prompts. Auto-detects GPU; falls back to CPU gracefully.

Prerequisites:
  - Run fix_csv.py first to generate sycophancy_eval_fixed.csv

Outputs:
  results/baseline_tinyllama_results.csv  (canonical)
  results/baseline_tinyllama_<timestamp>.csv  (timestamped backup)
"""

import os
import sys
import pandas as pd
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH    = os.path.join(BASE_DIR, "datasets", "sycophancy", "sycophancy_eval_fixed.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 150


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"Device : {device}")
    print(f"Loading: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)

    if device == "cuda":
        model = model.cuda()

    model.eval()
    return tokenizer, model, device


def generate_response(prompt: str, tokenizer, model, device: str) -> str:
    """Run a single forward pass and return only the assistant reply."""
    formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"

    inputs = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=512
    )
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Strip the prompt — keep only the assistant's reply
    marker = "<|assistant|>"
    if marker in full:
        return full.split(marker)[-1].strip()
    # Fallback: remove the formatted prompt text
    return full.replace(formatted.strip(), "").strip()


def main():
    # ── Validate prerequisites ────────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: Evaluation CSV not found:\n  {CSV_PATH}")
        print("Run fix_csv.py first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} evaluation prompts")
    print(f"Categories: {df['category'].value_counts().to_dict()}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    tokenizer, model, device = load_model()

    # ── Inference loop ────────────────────────────────────────────────────────
    results = []
    total   = len(df)

    for idx, row in df.iterrows():
        n = idx + 1
        print(f"[{n:3d}/{total}] {row['category']:<12} | {row['prompt'][:60]}...")

        try:
            response = generate_response(row["prompt"], tokenizer, model, device)
        except Exception as exc:
            print(f"          ERROR: {exc}")
            response = "ERROR"

        results.append({
            "idx"              : idx,
            "category"         : row["category"],
            "prompt"           : row["prompt"],
            "expected_behavior": row["expected_behavior"],
            "model_response"   : response,
            "model"            : "TinyLlama-1.1B-Chat",
            "checkpoint"       : "baseline",
            "step"             : 0,
        })

        # Preview
        preview = response[:90].replace("\n", " ")
        print(f"          → {preview}...")

    # ── Save results ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

    timestamped = os.path.join(RESULTS_DIR, f"baseline_tinyllama_{timestamp}.csv")
    canonical   = os.path.join(RESULTS_DIR, "baseline_tinyllama_results.csv")

    results_df.to_csv(timestamped, index=False)
    results_df.to_csv(canonical,   index=False)

    print(f"\n✓ Evaluation complete ({total} prompts)")
    print(f"  {canonical}")
    print(f"  {timestamped}")
    print(f"\nNext step: python evaluation/score_responses.py")


if __name__ == "__main__":
    main()
