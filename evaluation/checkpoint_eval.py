"""
Checkpoint Behaviour Tracking
===============================
File location: evaluation/checkpoint_eval.py
Run from project root: python evaluation/checkpoint_eval.py

This is the core research contribution of the project.

For each saved training checkpoint we:
  1. Load the LoRA adapter onto the base TinyLlama model
  2. Run the full 99-prompt sycophancy evaluation
  3. Score each response (sycophantic / corrective / neutral / refused)
  4. Record rates per checkpoint and per category

The result is a longitudinal view of HOW BEHAVIOUR CHANGES during
instruction fine-tuning — the key novelty of this research.

Expected insight questions:
  - Does sycophancy increase as the model learns to be "helpful"?
  - Does instruction tuning reduce or amplify stereotypical responses?
  - At which checkpoint does behaviour stabilise?

Prerequisites:
  - fix_csv.py must have been run
  - evaluation/baseline_eval.py + score_responses.py must have been run
  - training/finetune_lora.py must have generated checkpoints

Outputs:
  results/checkpoint_eval_results.csv   — all responses across all checkpoints
  results/metrics_by_checkpoint.csv     — aggregate rates per checkpoint
"""

import os
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Resolve project root whether script is run from root or evaluation/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(_SCRIPT_DIR) if os.path.basename(_SCRIPT_DIR) == "evaluation" else _SCRIPT_DIR
sys.path.insert(0, BASE_DIR)

# Import scoring function (same heuristics used for baseline)
from evaluation.score_responses import score_response

CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints", "tinyllama-lora")
EVAL_CSV        = os.path.join(BASE_DIR, "datasets", "sycophancy", "sycophancy_eval_fixed.csv")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

os.makedirs(RESULTS_DIR, exist_ok=True)


def find_checkpoints() -> list:
    """
    Return list of (step, path) tuples, sorted by step.
    Includes final_model as step=999999.
    """
    checkpoints = []

    if not os.path.exists(CHECKPOINTS_DIR):
        return checkpoints

    for item in os.listdir(CHECKPOINTS_DIR):
        full = os.path.join(CHECKPOINTS_DIR, item)
        if item.startswith("checkpoint-") and os.path.isdir(full):
            try:
                step = int(item.replace("checkpoint-", ""))
                checkpoints.append((step, full))
            except ValueError:
                pass

    final = os.path.join(CHECKPOINTS_DIR, "final_model")
    if os.path.exists(final):
        checkpoints.append((999999, final))

    final_only = []

    for step, path in checkpoints:
        if step == 999999:
            final_only.append((step, path))

    return final_only


def load_checkpoint(checkpoint_path: str, device: str):
    """Load base TinyLlama + LoRA adapter from a checkpoint directory."""
    print(f"  Loading checkpoint: {os.path.basename(checkpoint_path)}")

    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer  = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=dtype)
    model      = PeftModel.from_pretrained(base_model, checkpoint_path)

    if device == "cuda":
        model = model.cuda()
    else:
        model = model.float()

    model.eval()
    print("  Model loaded successfully")
    return tokenizer, model


def generate(prompt: str, tokenizer, model, device: str) -> str:
    """Run inference and return the assistant reply only."""
    formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs    = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)

    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.3,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    if "<|assistant|>" in full:
        return full.split("<|assistant|>")[-1].strip()
    return full.replace(formatted.strip(), "").strip()


def eval_one_checkpoint(step: int, path: str, eval_df: pd.DataFrame, device: str) -> pd.DataFrame:
    """Evaluate all prompts against one checkpoint; return a scored DataFrame."""
    label = f"step-{step}" if step != 999999 else "final"
    print(f"\n{'='*55}")
    print(f"Checkpoint: {label}")

    tokenizer, model = load_checkpoint(path, device)
    rows = []

    for idx, row in eval_df.iterrows():

        print(
            f"    Prompt {idx+1}/{len(eval_df)}",
            end="\r",
            flush=True
        )

        try:
            response = generate(row["prompt"], tokenizer, model, device)

            if (idx + 1) % 5 == 0:
                print(f"\nCompleted {idx+1}/{len(eval_df)} prompts")

            s = score_response(
                str(response),
                str(row["expected_behavior"])
            )
        except Exception as exc:
            response = "ERROR"
            s        = "neutral"

        rows.append({
            "checkpoint"       : label,
            "step"             : step,
            "category"         : row["category"],
            "prompt"           : row["prompt"],
            "expected_behavior": row["expected_behavior"],
            "model_response"   : response,
            "score"            : s,
        })

    df_out = pd.DataFrame(rows)
    syco_r = (df_out["score"] == "sycophantic").mean()
    corr_r = (df_out["score"] == "corrective").mean()
    print(f"  Sycophancy rate : {syco_r:.1%}")
    print(f"  Correction rate : {corr_r:.1%}")

    # Free GPU memory between checkpoints
    del model, tokenizer
    import gc
    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()

    return df_out


def aggregate_metrics(all_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-checkpoint aggregate rates."""
    rows = []
    for cp in all_df["checkpoint"].unique():
        cdf   = all_df[all_df["checkpoint"] == cp]
        step  = cdf["step"].iloc[0]
        total = len(cdf)
        rows.append({
            "checkpoint"     : cp,
            "step"           : step,
            "total_prompts"  : total,
            "sycophancy_rate": (cdf["score"] == "sycophantic").sum() / total,
            "correction_rate": (cdf["score"] == "corrective").sum() / total,
            "neutral_rate"   : (cdf["score"] == "neutral").sum() / total,
            "refusal_rate"   : (cdf["score"] == "refused").sum() / total,
        })
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load eval prompts
    if not os.path.exists(EVAL_CSV):
        print(f"ERROR: {EVAL_CSV} not found. Run fix_csv.py first.")
        sys.exit(1)
    
    eval_df = pd.read_csv(EVAL_CSV)

    # Fast evaluation mode
    # eval_df = eval_df.head(20)

    print(f"Evaluation prompts: {len(eval_df)}")

    # Find checkpoints
    checkpoints = find_checkpoints()
    if not checkpoints:
        print("No checkpoints found. Run training/finetune_lora.py first.")
        sys.exit(1)

    print(f"\nFound {len(checkpoints)} checkpoints:")
    for step, path in checkpoints:
        label = f"step-{step}" if step != 999999 else "final"
        print(f"  {label:<12}: {path}")

    # ── Start with baseline (if scored CSV exists) ─────────────────────────────
    all_results = []
    baseline_scored = os.path.join(RESULTS_DIR, "baseline_tinyllama_scored.csv")
    if os.path.exists(baseline_scored):
        print("\nIncluding baseline scores in comparison...")
        bdf = pd.read_csv(baseline_scored)
        bdf["checkpoint"] = "baseline"
        bdf["step"]       = 0
        all_results.append(bdf)
    else:
        print("\nWARNING: baseline_tinyllama_scored.csv not found.")
        print("Run evaluation/score_responses.py for a complete comparison.")

    # ── Evaluate each checkpoint ───────────────────────────────────────────────
    raw_path     = os.path.join(RESULTS_DIR, "checkpoint_eval_results.csv")
    metrics_path = os.path.join(RESULTS_DIR, "metrics_by_checkpoint.csv")

    for step, checkpoint_path in checkpoints:
        cp_df = eval_one_checkpoint(step, checkpoint_path, eval_df, device)
        all_results.append(cp_df)

        # Incremental save — preserves progress on crash/interruption
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(raw_path, index=False)
        aggregate_metrics(combined).to_csv(metrics_path, index=False)

    final_combined = pd.concat(all_results, ignore_index=True)
    final_combined.to_csv(raw_path, index=False)
    aggregate_metrics(final_combined).to_csv(metrics_path, index=False)

    print(f"\n✓ Checkpoint evaluation complete.")
    print(f"  Raw responses : {raw_path}")
    print(f"  Metrics       : {metrics_path}")
    print(f"\nNext step: python analysis/compute_metrics.py")


if __name__ == "__main__":
    main()