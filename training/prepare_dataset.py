"""
Training Dataset Preparation
==============================
File location: training/prepare_dataset.py
Run from project root: python training/prepare_dataset.py

Downloads the Alpaca instruction-following dataset from HuggingFace
and formats it for TinyLlama's chat template.

Why Alpaca?
  Alpaca (52k instruction-following samples) is ideal for this research
  because it teaches the model to follow instructions precisely —
  making changes in sycophancy and bias more observable post fine-tuning.

Configuration:
  NUM_TRAIN_SAMPLES = 500   (CPU-safe default; use 5000+ on GPU)
  NUM_VAL_SAMPLES   = 100

Output:
  training/alpaca_prepared/train/       — HuggingFace dataset (arrow format)
  training/alpaca_prepared/val/
  training/alpaca_prepared/config.json
"""

import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "training", "alpaca_prepared")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME        = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NUM_TRAIN_SAMPLES = 500    # ← increase to 5000+ for GPU runs
NUM_VAL_SAMPLES   = 100
MAX_SEQ_LENGTH    = 512


def format_alpaca_to_tinyllama(sample: dict) -> dict:
    """
    Convert one Alpaca sample into the TinyLlama chat template.

    Alpaca fields: instruction, input (optional), output
    TinyLlama template:
        <|user|>
        {instruction}\\n\\nInput: {input}</s>    (input only if non-empty)
        <|assistant|>
        {output}</s>
    """
    instruction = sample.get("instruction", "").strip()
    inp         = sample.get("input", "").strip()
    output      = sample.get("output", "").strip()

    user_msg = f"{instruction}\n\nInput: {inp}" if inp else instruction

    text = (
        f"<|user|>\n{user_msg}</s>\n"
        f"<|assistant|>\n{output}</s>"
    )
    return {"text": text}


def main():
    print("=" * 60)
    print("Dataset Preparation: Alpaca → TinyLlama format")
    print("=" * 60)

    print("\nDownloading tatsu-lab/alpaca from HuggingFace...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"Full dataset size: {len(dataset)} samples")

    # Sample a reproducible subset
    total_needed = NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES
    if total_needed > len(dataset):
        print(f"WARNING: requested {total_needed} but dataset only has {len(dataset)}.")
        total_needed = len(dataset)

    subset = dataset.shuffle(seed=42).select(range(total_needed))
    train_raw = subset.select(range(NUM_TRAIN_SAMPLES))
    val_raw   = subset.select(range(NUM_TRAIN_SAMPLES, total_needed))

    print(f"\nUsing {len(train_raw)} training  |  {len(val_raw)} validation samples")

    # Format
    print("Formatting into TinyLlama chat template...")
    remove_cols = train_raw.column_names
    train_fmt = train_raw.map(format_alpaca_to_tinyllama, remove_columns=remove_cols)
    val_fmt   = val_raw.map(format_alpaca_to_tinyllama,   remove_columns=remove_cols)

    # Token-length analysis
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    lengths = [len(tokenizer(s["text"], truncation=False)["input_ids"])
               for s in train_fmt]

    print(f"\nToken length stats (train set):")
    print(f"  Min : {min(lengths)}")
    print(f"  Max : {max(lengths)}")
    print(f"  Avg : {sum(lengths)/len(lengths):.0f}")
    too_long = sum(1 for l in lengths if l > MAX_SEQ_LENGTH)
    pct_long = too_long / len(lengths) * 100
    print(f"  Exceeds {MAX_SEQ_LENGTH} tokens: {too_long} ({pct_long:.1f}%) — will be truncated during training")

    # Save
    train_fmt.save_to_disk(os.path.join(OUTPUT_DIR, "train"))
    val_fmt.save_to_disk(  os.path.join(OUTPUT_DIR, "val"))

    config = {
        "model_name"        : MODEL_NAME,
        "dataset"           : "tatsu-lab/alpaca",
        "num_train_samples" : NUM_TRAIN_SAMPLES,
        "num_val_samples"   : NUM_VAL_SAMPLES,
        "max_seq_length"    : MAX_SEQ_LENGTH,
        "seed"              : 42,
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Prepared dataset saved to: {OUTPUT_DIR}")
    print(f"  training/alpaca_prepared/train/")
    print(f"  training/alpaca_prepared/val/")
    print(f"  training/alpaca_prepared/config.json")
    print(f"\nNext step: python training/finetune_lora.py")


if __name__ == "__main__":
    main()
