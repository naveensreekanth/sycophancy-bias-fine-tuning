"""
Quantitative Metrics Computation
==================================
File location: analysis/compute_metrics.py
Run from project root: python analysis/compute_metrics.py

Aggregates all checkpoint evaluation data into clean metric tables.

Computes:
  - Sycophancy rate, correction rate, neutral rate, refusal rate
    → per checkpoint  (tracks trajectory over training)
  - Same breakdown per category × checkpoint
    (e.g., does math sycophancy behave differently from gender bias?)
  - Delta table: how much each metric changed vs baseline

Outputs:
  results/metrics_by_checkpoint.csv
  results/metrics_by_category.csv
  results/metrics_delta.csv          — change from baseline
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _rate(df: pd.DataFrame, label: str) -> float:
    return (df["score"] == label).sum() / len(df) if len(df) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Metric builders
# ─────────────────────────────────────────────────────────────────────────────

def build_checkpoint_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cp in df["checkpoint"].unique():
        cdf   = df[df["checkpoint"] == cp]
        step  = int(cdf["step"].iloc[0]) if "step" in cdf.columns else -1
        total = len(cdf)
        rows.append({
            "checkpoint"     : cp,
            "step"           : step,
            "total_prompts"  : total,
            "sycophantic"    : (cdf["score"] == "sycophantic").sum(),
            "corrective"     : (cdf["score"] == "corrective").sum(),
            "neutral"        : (cdf["score"] == "neutral").sum(),
            "refused"        : (cdf["score"] == "refused").sum(),
            "sycophancy_rate": _rate(cdf, "sycophantic"),
            "correction_rate": _rate(cdf, "corrective"),
            "neutral_rate"   : _rate(cdf, "neutral"),
            "refusal_rate"   : _rate(cdf, "refused"),
        })
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def build_category_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cp in df["checkpoint"].unique():
        step = int(df[df["checkpoint"] == cp]["step"].iloc[0]) \
               if "step" in df.columns else -1
        for cat in sorted(df["category"].unique()):
            cdf = df[(df["checkpoint"] == cp) & (df["category"] == cat)]
            if len(cdf) == 0:
                continue
            rows.append({
                "checkpoint"     : cp,
                "step"           : step,
                "category"       : cat,
                "total"          : len(cdf),
                "sycophancy_rate": _rate(cdf, "sycophantic"),
                "correction_rate": _rate(cdf, "corrective"),
                "neutral_rate"   : _rate(cdf, "neutral"),
            })
    return pd.DataFrame(rows).sort_values(["step", "category"]).reset_index(drop=True)


def build_delta_table(cp_metrics: pd.DataFrame) -> pd.DataFrame:
    """How much did each metric change compared to the baseline checkpoint?"""
    if "baseline" not in cp_metrics["checkpoint"].values:
        # Use the first checkpoint as the reference
        ref = cp_metrics.iloc[0]
    else:
        ref = cp_metrics[cp_metrics["checkpoint"] == "baseline"].iloc[0]

    rate_cols = ["sycophancy_rate", "correction_rate", "neutral_rate", "refusal_rate"]
    rows = []
    for _, row in cp_metrics.iterrows():
        entry = {"checkpoint": row["checkpoint"], "step": row["step"]}
        for col in rate_cols:
            entry[f"delta_{col}"] = row[col] - ref[col]
        rows.append(entry)
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Console reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_checkpoint_table(cp_metrics: pd.DataFrame):
    print("\n" + "=" * 72)
    print("CHECKPOINT BEHAVIOUR TRAJECTORY")
    print("=" * 72)
    header = f"  {'Checkpoint':<14}  {'Sycoph':>8}  {'Correct':>8}  {'Neutral':>8}  {'Refused':>8}"
    print(header)
    print("  " + "-" * 68)
    for _, row in cp_metrics.iterrows():
        print(
            f"  {row['checkpoint']:<14}  "
            f"{row['sycophancy_rate']:>7.1%}  "
            f"{row['correction_rate']:>7.1%}  "
            f"{row['neutral_rate']:>7.1%}  "
            f"{row['refusal_rate']:>7.1%}"
        )


def print_trend_analysis(cp_metrics: pd.DataFrame, delta: pd.DataFrame):
    if len(cp_metrics) < 2:
        return

    baseline = cp_metrics.iloc[0]
    final    = cp_metrics.iloc[-1]

    print("\n" + "=" * 72)
    print("TREND ANALYSIS  (baseline → final)")
    print("=" * 72)

    metrics = [
        ("sycophancy_rate", "Sycophancy", "lower is better"),
        ("correction_rate", "Correction", "higher is better"),
        ("neutral_rate",    "Neutral",    "informational"),
        ("refusal_rate",    "Refusal",    "informational"),
    ]
    for col, label, note in metrics:
        diff = final[col] - baseline[col]
        arrow = "▲" if diff > 0 else "▼" if diff < 0 else "─"
        sign  = "+" if diff >= 0 else ""
        print(f"  {label:<14}: {baseline[col]:.1%} → {final[col]:.1%}  "
              f"{arrow} {sign}{diff:.1%}   ({note})")

    # Research interpretation
    syco_diff = final["sycophancy_rate"] - baseline["sycophancy_rate"]
    corr_diff = final["correction_rate"] - baseline["correction_rate"]

    print("\nResearch Interpretation:")
    if syco_diff > 0.05:
        print("  ⚠  Instruction fine-tuning INCREASED sycophancy — "
              "model became more agreeable even to false claims.")
    elif syco_diff < -0.05:
        print("  ✓  Instruction fine-tuning DECREASED sycophancy — "
              "model pushes back more after training.")
    else:
        print("  ○  Minimal change in sycophancy rate across training.")

    if corr_diff > 0.05:
        print("  ✓  Correction rate improved — model is more factually assertive.")
    elif corr_diff < -0.05:
        print("  ⚠  Correction rate dropped — model may be over-fitted to agreement.")


def main():
    results_path = os.path.join(RESULTS_DIR, "checkpoint_eval_results.csv")

    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found.")
        print("Run evaluation/checkpoint_eval.py first.")
        sys.exit(1)

    df = pd.read_csv(results_path)
    n_checkpoints = df["checkpoint"].nunique()
    print(f"Loaded {len(df)} records across {n_checkpoints} checkpoints.")

    # Build tables
    cp_metrics  = build_checkpoint_metrics(df)
    cat_metrics = build_category_metrics(df)
    delta       = build_delta_table(cp_metrics)

    # Save
    cp_path  = os.path.join(RESULTS_DIR, "metrics_by_checkpoint.csv")
    cat_path = os.path.join(RESULTS_DIR, "metrics_by_category.csv")
    del_path = os.path.join(RESULTS_DIR, "metrics_delta.csv")

    cp_metrics.to_csv(cp_path,   index=False)
    cat_metrics.to_csv(cat_path, index=False)
    delta.to_csv(del_path,       index=False)

    # Report
    print_checkpoint_table(cp_metrics)
    print_trend_analysis(cp_metrics, delta)

    print(f"\n✓ Metrics saved:")
    print(f"  {cp_path}")
    print(f"  {cat_path}")
    print(f"  {del_path}")
    print(f"\nNext step: python analysis/visualize_results.py")


if __name__ == "__main__":
    main()
