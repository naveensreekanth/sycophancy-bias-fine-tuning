"""
Results Visualisation
======================
File location: analysis/visualize_results.py
Run from project root: python analysis/visualize_results.py

Generates 5 publication-quality figures:

  Fig 1  rates_over_training.png     — sycophancy & correction rates vs steps
  Fig 2  score_distribution.png      — stacked bar of all score categories
  Fig 3  category_heatmap.png        — per-category sycophancy across checkpoints
  Fig 4  baseline_vs_final.png       — grouped before/after comparison
  Fig 5  delta_trajectory.png        — how much each rate CHANGED from baseline

All figures saved to: results/figures/
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # headless — safe on Windows/servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"          : 150,
    "font.family"         : "DejaVu Sans",
    "font.size"           : 11,
    "axes.spines.top"     : False,
    "axes.spines.right"   : False,
    "axes.grid"           : True,
    "grid.alpha"          : 0.35,
    "grid.linestyle"      : "--",
    "axes.titlesize"      : 13,
    "axes.titleweight"    : "bold",
    "axes.labelsize"      : 11,
})

PALETTE = {
    "sycophantic": "#E74C3C",   # red
    "corrective" : "#27AE60",   # green
    "neutral"    : "#95A5A6",   # grey
    "refused"    : "#3498DB",   # blue
}

PCT = plt.FuncFormatter(lambda y, _: f"{y:.0%}")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    cp_path  = os.path.join(RESULTS_DIR, "metrics_by_checkpoint.csv")
    cat_path = os.path.join(RESULTS_DIR, "metrics_by_category.csv")
    del_path = os.path.join(RESULTS_DIR, "metrics_delta.csv")

    if not os.path.exists(cp_path):
        print(f"ERROR: {cp_path} not found.")
        print("Run analysis/compute_metrics.py first.")
        sys.exit(1)

    cp_df  = pd.read_csv(cp_path)
    cat_df = pd.read_csv(cat_path)  if os.path.exists(cat_path)  else None
    del_df = pd.read_csv(del_path)  if os.path.exists(del_path)  else None

    return cp_df, cat_df, del_df


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Rates over training steps
# ─────────────────────────────────────────────────────────────────────────────

def fig_rates_over_training(cp_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 5))

    labels = cp_df["checkpoint"].tolist()
    x      = np.arange(len(labels))

    ax.plot(x, cp_df["sycophancy_rate"], "o-",
            color=PALETTE["sycophantic"], lw=2.5, ms=8, label="Sycophancy Rate", zorder=3)
    ax.plot(x, cp_df["correction_rate"], "s-",
            color=PALETTE["corrective"],  lw=2.5, ms=8, label="Correction Rate",  zorder=3)
    ax.plot(x, cp_df["neutral_rate"],    "^--",
            color=PALETTE["neutral"],     lw=1.5, ms=6, label="Neutral Rate",      zorder=2)
    if "refusal_rate" in cp_df.columns:
        ax.plot(x, cp_df["refusal_rate"], "d:",
                color=PALETTE["refused"], lw=1.5, ms=6, label="Refusal Rate",      zorder=2)

    # Shade baseline & final
    if len(x) > 1:
        ax.axvspan(-0.45, 0.45, alpha=0.09, color="#E67E22",  label="Baseline region")
        ax.axvspan(x[-1]-0.45, x[-1]+0.45, alpha=0.09, color="#1ABC9C", label="Final region")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(-0.04, 1.04)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax.set_ylabel("Rate")
    ax.set_xlabel("Checkpoint")
    ax.set_title(
        "Behavioural Rates Across Training Checkpoints\n"
        "TinyLlama-1.1B-Chat  +  LoRA  (Alpaca instruction fine-tuning)"
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "rates_over_training.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓  rates_over_training.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Stacked score distribution
# ─────────────────────────────────────────────────────────────────────────────

def fig_score_distribution(cp_df: pd.DataFrame):
    labels  = cp_df["checkpoint"].tolist()
    x       = np.arange(len(labels))
    w       = 0.55

    syco = cp_df["sycophancy_rate"].fillna(0).values
    corr = cp_df["correction_rate"].fillna(0).values
    neut = cp_df["neutral_rate"].fillna(0).values
    refu = cp_df.get("refusal_rate", pd.Series([0]*len(cp_df))).fillna(0).values

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, syco,              w, label="Sycophantic", color=PALETTE["sycophantic"])
    ax.bar(x, corr, w, bottom=syco,             label="Corrective",  color=PALETTE["corrective"])
    ax.bar(x, neut, w, bottom=syco+corr,        label="Neutral",     color=PALETTE["neutral"])
    ax.bar(x, refu, w, bottom=syco+corr+neut,   label="Refused",     color=PALETTE["refused"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Proportion of Responses")
    ax.set_xlabel("Checkpoint")
    ax.set_title("Response Score Distribution per Checkpoint")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "score_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓  score_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Category × Checkpoint heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig_category_heatmap(cat_df: pd.DataFrame):
    if cat_df is None or cat_df.empty:
        print("  ⚠  category_heatmap.png skipped (no category data)")
        return

    pivot = cat_df.pivot_table(
        index="category", columns="checkpoint", values="sycophancy_rate"
    )
    # Sort checkpoints by step
    step_map = cat_df.drop_duplicates("checkpoint").set_index("checkpoint")["step"].to_dict()
    ordered_cols = sorted(pivot.columns, key=lambda c: step_map.get(c, 0))
    pivot = pivot[ordered_cols]

    nrows, ncols = pivot.shape
    fig, ax = plt.subplots(figsize=(max(8, ncols * 1.4), max(4, nrows * 0.7 + 1.5)))

    im = ax.imshow(pivot.values, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(ordered_cols, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(nrows):
        for j in range(ncols):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.55 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Sycophancy Rate", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(PCT)

    ax.set_xlabel("Checkpoint", fontsize=11)
    ax.set_ylabel("Bias Category", fontsize=11)
    ax.set_title("Sycophancy Rate by Category and Checkpoint\n"
                 "(Red = more sycophantic, Green = more corrective)")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "category_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓  category_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Baseline vs Final grouped bar
# ─────────────────────────────────────────────────────────────────────────────

def fig_baseline_vs_final(cp_df: pd.DataFrame):
    if len(cp_df) < 2:
        print("  ⚠  baseline_vs_final.png skipped (need ≥ 2 checkpoints)")
        return

    baseline = cp_df.iloc[0]
    final    = cp_df.iloc[-1]

    metrics      = ["sycophancy_rate", "correction_rate", "neutral_rate"]
    display_labs = ["Sycophancy", "Correction", "Neutral"]
    bar_colors   = [PALETTE["sycophantic"], PALETTE["corrective"], PALETTE["neutral"]]

    x     = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, [baseline[m] for m in metrics], width,
                   label=f"Baseline ({baseline['checkpoint']})",
                   color="#BDC3C7", edgecolor="#7F8C8D", lw=0.8)
    bars2 = ax.bar(x + width/2, [final[m] for m in metrics], width,
                   label=f"Fine-tuned ({final['checkpoint']})",
                   color=bar_colors, edgecolor="#555", lw=0.8)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                f"{h:.0%}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(display_labs, fontsize=12)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Rate")
    ax.set_title("Baseline vs Fine-Tuned Model: Behavioural Comparison")
    ax.legend(frameon=False, fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "baseline_vs_final.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓  baseline_vs_final.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Delta trajectory (change from baseline)
# ─────────────────────────────────────────────────────────────────────────────

def fig_delta_trajectory(del_df: pd.DataFrame):
    if del_df is None or del_df.empty or len(del_df) < 2:
        print("  ⚠  delta_trajectory.png skipped (insufficient data)")
        return

    labels = del_df["checkpoint"].tolist()
    x      = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0, color="black", lw=1.2, zorder=1)

    delta_cols = {
        "delta_sycophancy_rate": ("Δ Sycophancy", PALETTE["sycophantic"], "o-"),
        "delta_correction_rate": ("Δ Correction", PALETTE["corrective"],  "s-"),
        "delta_neutral_rate"   : ("Δ Neutral",    PALETTE["neutral"],     "^--"),
    }
    for col, (label, color, style) in delta_cols.items():
        if col in del_df.columns:
            ax.plot(x, del_df[col], style, color=color, lw=2.2, ms=7,
                    label=label, zorder=3)
            # Fill above/below zero
            ax.fill_between(x, del_df[col], 0,
                            where=(del_df[col] > 0), alpha=0.12, color=color)
            ax.fill_between(x, del_df[col], 0,
                            where=(del_df[col] < 0), alpha=0.12, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0%}"))
    ax.set_ylabel("Change from Baseline")
    ax.set_xlabel("Checkpoint")
    ax.set_title(
        "Behavioural Change Relative to Baseline\n"
        "(Positive = increased vs baseline, Negative = decreased)"
    )
    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "delta_trajectory.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓  delta_trajectory.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Generating visualisations...")
    cp_df, cat_df, del_df = load_data()
    print(f"Checkpoints loaded: {len(cp_df)}\n")

    fig_rates_over_training(cp_df)
    fig_score_distribution(cp_df)
    fig_category_heatmap(cat_df)
    fig_baseline_vs_final(cp_df)
    fig_delta_trajectory(del_df)

    print(f"\n✓ All figures saved to: {FIGURES_DIR}")
    print("\nFigures:")
    print("  1. rates_over_training.png  — primary research finding (use in paper)")
    print("  2. score_distribution.png   — full breakdown per checkpoint")
    print("  3. category_heatmap.png     — per-category sycophancy analysis")
    print("  4. baseline_vs_final.png    — clean before/after comparison")
    print("  5. delta_trajectory.png     — how much each behaviour shifted")
    print(f"\nNext step: Write the research paper using research_notes/methodology.md")


if __name__ == "__main__":
    main()
