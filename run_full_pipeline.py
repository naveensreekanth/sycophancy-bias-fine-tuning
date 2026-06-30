"""
Full Pipeline Runner
=====================
File location: run_full_pipeline.py  (project ROOT)
Run as: python run_full_pipeline.py

Master orchestrator for the complete research pipeline.
Phases with existing outputs are automatically skipped so you can
safely re-run after an interruption without redoing finished work.

Phases:
  1  fix_csv.py                          Fix & clean evaluation CSV
  2  evaluation/baseline_eval.py         Run baseline TinyLlama inference
  3  evaluation/score_responses.py       Score baseline responses
  4  training/prepare_dataset.py         Download & format Alpaca dataset
  5  training/finetune_lora.py           LoRA fine-tuning (LONGEST phase)
  6  evaluation/checkpoint_eval.py       Evaluate each saved checkpoint
  7  analysis/compute_metrics.py         Aggregate quantitative metrics
  8  analysis/visualize_results.py       Generate all figures

Usage:
  python run_full_pipeline.py            # run all phases
  python run_full_pipeline.py --from 4   # resume from a specific phase
  python run_full_pipeline.py --only 8   # run a single phase
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── ANSI colours (Windows-safe fallback) ─────────────────────────────────────
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(
        ctypes.windll.kernel32.GetStdHandle(-11), 7
    )
    _ANSI = True
except Exception:
    _ANSI = sys.platform != "win32"

def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if _ANSI else text

def green(t):  return _c(t, "32")
def red(t):    return _c(t, "31")
def yellow(t): return _c(t, "33")
def bold(t):   return _c(t, "1")
def cyan(t):   return _c(t, "36")


# ── Phase definitions ─────────────────────────────────────────────────────────
# Each phase has:
#   script        — path relative to project root
#   description   — human label
#   skip_if_all   — list of paths; if ALL exist, phase is skipped
#   warning       — optional note shown before running
PHASES = [
    {
        "n"          : 1,
        "script"     : "fix_csv.py",
        "description": "Fix & clean evaluation CSV",
        "skip_if_all": [
            os.path.join("datasets", "sycophancy", "sycophancy_eval_fixed.csv")
        ],
    },
    {
        "n"          : 2,
        "script"     : os.path.join("evaluation", "baseline_eval.py"),
        "description": "Run baseline TinyLlama inference (99 prompts)",
        "skip_if_all": [
            os.path.join("results", "baseline_tinyllama_results.csv")
        ],
        "warning"    : "This may take 10–30 min on CPU.",
    },
    {
        "n"          : 3,
        "script"     : os.path.join("evaluation", "score_responses.py"),
        "description": "Score baseline responses (heuristic)",
        "skip_if_all": [
            os.path.join("results", "baseline_tinyllama_scored.csv")
        ],
    },
    {
        "n"          : 4,
        "script"     : os.path.join("training", "prepare_dataset.py"),
        "description": "Download & format Alpaca training dataset",
        "skip_if_all": [
            os.path.join("training", "alpaca_prepared", "config.json")
        ],
        "warning"    : "Requires internet access to download from HuggingFace.",
    },
    {
        "n"          : 5,
        "script"     : os.path.join("training", "finetune_lora.py"),
        "description": "LoRA fine-tuning on Alpaca (checkpoints saved every 100 steps)",
        "skip_if_all": [
            os.path.join("training", "output", "final_model")
        ],
        "warning"    : (
            "LONGEST PHASE — estimated time:\n"
            "    CPU  (500 samples, 3 epochs): 2–4 hours\n"
            "    GPU T4 (500 samples):         10–20 min\n"
            "  Consider running on Google Colab with GPU for this phase."
        ),
    },
    {
        "n"          : 6,
        "script"     : os.path.join("evaluation", "checkpoint_eval.py"),
        "description": "Evaluate model behaviour at every checkpoint",
        "skip_if_all": [
            os.path.join("results", "checkpoint_eval_results.csv")
        ],
        "warning"    : "Loads each checkpoint in sequence. May take 30–60 min on CPU.",
    },
    {
        "n"          : 7,
        "script"     : os.path.join("analysis", "compute_metrics.py"),
        "description": "Aggregate quantitative metrics across checkpoints",
        "skip_if_all": [
            os.path.join("results", "metrics_by_checkpoint.csv"),
            os.path.join("results", "metrics_by_category.csv"),
        ],
    },
    {
        "n"          : 8,
        "script"     : os.path.join("analysis", "visualize_results.py"),
        "description": "Generate all research figures (5 plots)",
        "skip_if_all": [
            os.path.join("results", "figures", "rates_over_training.png")
        ],
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_exist(paths: list) -> bool:
    return all(os.path.exists(os.path.join(BASE_DIR, p)) for p in paths)


def _fmt_elapsed(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    m, s   = divmod(rem, 60)
    if td.days or h:
        return f"{td.days*24+h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def run_phase(phase: dict, dry_run: bool = False) -> bool:
    """Execute one pipeline phase. Returns True on success."""
    n    = phase["n"]
    desc = phase["description"]
    script_rel = phase["script"]
    script_abs = os.path.join(BASE_DIR, script_rel)

    print()
    print(bold(f"{'─'*62}"))
    print(bold(f"  Phase {n}/8 — {desc}"))
    print(bold(f"{'─'*62}"))

    # Skip check
    if _all_exist(phase.get("skip_if_all", [])):
        print(green("  ✓ Output already exists — skipping."))
        return True

    # Script existence check
    if not os.path.exists(script_abs):
        print(red(f"  ✗ Script not found: {script_rel}"))
        print(red("    Place all scripts in the correct directories and retry."))
        return False

    # Warning
    if "warning" in phase:
        for line in phase["warning"].splitlines():
            print(yellow(f"  ⚠  {line}"))

    if dry_run:
        print(cyan("  [dry-run] Would execute: python " + script_rel))
        return True

    print(f"  Running: python {script_rel}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print()

    t0     = time.time()
    result = subprocess.run(
        [sys.executable, script_abs],
        cwd=BASE_DIR
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print()
        print(green(f"  ✓ Phase {n} complete in {_fmt_elapsed(elapsed)}"))
        return True
    else:
        print()
        print(red(f"  ✗ Phase {n} FAILED (exit code {result.returncode}) after {_fmt_elapsed(elapsed)}"))
        print(red("    Fix the error above, then re-run this script."))
        print(red("    Completed phases will be skipped automatically."))
        return False


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full sycophancy-bias-fine-tuning pipeline."
    )
    parser.add_argument(
        "--from", dest="from_phase", type=int, default=1, metavar="N",
        help="Start from phase N (default: 1)"
    )
    parser.add_argument(
        "--only", dest="only_phase", type=int, default=None, metavar="N",
        help="Run only phase N, skip all others"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without executing anything"
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print()
    print(bold("=" * 62))
    print(bold("  SYCOPHANCY & BIAS FINE-TUNING — FULL PIPELINE RUNNER"))
    print(bold("=" * 62))
    print(f"  Project root : {BASE_DIR}")
    print(f"  Started      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.dry_run:
        print(yellow("  Mode         : DRY RUN (no scripts will execute)"))
    elif args.only_phase:
        print(cyan(f"  Mode         : Run only phase {args.only_phase}"))
    elif args.from_phase > 1:
        print(cyan(f"  Mode         : Resume from phase {args.from_phase}"))
    else:
        print("  Mode         : Full pipeline")

    # Filter phases
    if args.only_phase:
        phases_to_run = [p for p in PHASES if p["n"] == args.only_phase]
        if not phases_to_run:
            print(red(f"\nERROR: Phase {args.only_phase} not found (valid: 1–8)"))
            sys.exit(1)
    else:
        phases_to_run = [p for p in PHASES if p["n"] >= args.from_phase]

    # Summarise what will run
    print()
    print("  Phases queued:")
    for p in PHASES:
        exists  = _all_exist(p.get("skip_if_all", []))
        queued  = any(q["n"] == p["n"] for q in phases_to_run)
        if not queued:
            status = cyan("  [skipped by --from/--only]")
        elif exists:
            status = green("  ✓ will skip (output exists)")
        else:
            status = "  → will run"
        print(f"    Phase {p['n']}: {p['description']}{status}")

    print()
    total_start = time.time()
    succeeded   = 0
    skipped     = 0

    for phase in phases_to_run:
        already_done = _all_exist(phase.get("skip_if_all", []))
        ok = run_phase(phase, dry_run=args.dry_run)

        if not ok:
            sys.exit(1)

        if already_done:
            skipped += 1
        else:
            succeeded += 1

    total_elapsed = time.time() - total_start

    print()
    print(bold("=" * 62))
    print(bold(green(f"  PIPELINE COMPLETE")))
    print(bold("=" * 62))
    print(f"  Phases run    : {succeeded}")
    print(f"  Phases skipped: {skipped}  (outputs already existed)")
    print(f"  Total time    : {_fmt_elapsed(total_elapsed)}")
    print()
    print("  Key outputs:")
    outputs = [
        ("datasets/sycophancy/sycophancy_eval_fixed.csv", "Clean eval dataset"),
        ("results/baseline_tinyllama_results.csv",        "Raw baseline responses"),
        ("results/baseline_tinyllama_scored.csv",         "Scored baseline responses"),
        ("training/output/final_model/",                  "Fine-tuned LoRA model"),
        ("results/checkpoint_eval_results.csv",           "All checkpoint responses"),
        ("results/metrics_by_checkpoint.csv",             "Aggregate metrics table"),
        ("results/metrics_by_category.csv",               "Per-category breakdown"),
        ("results/figures/rates_over_training.png",       "Primary research figure"),
        ("results/figures/category_heatmap.png",          "Category heatmap"),
        ("results/figures/baseline_vs_final.png",         "Before/after comparison"),
    ]
    for path, label in outputs:
        full = os.path.join(BASE_DIR, path)
        tick = green("✓") if os.path.exists(full) else yellow("○")
        print(f"    {tick}  {label:<35} {path}")

    print()
    print("  Next: Open research_notes/methodology.md to write your paper.")
    print()


if __name__ == "__main__":
    main()
