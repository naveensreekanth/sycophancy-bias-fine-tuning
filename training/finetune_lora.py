import os
import torch
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

OUTPUT_DIR = "checkpoints/tinyllama-lora"

DATA_PATH = "datasets/sycophancy/expanded_anti_sycophancy_dataset.csv"

EPOCHS = 3
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
MAX_LENGTH = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LoRA Fine-Tuning: TinyLlama on Sycophancy Dataset")
print("=" * 60)

print(f"Device: {DEVICE}")
print()

df = pd.read_csv(DATA_PATH, encoding="cp1252")

print("Dataset Loaded")
print(df.head())

# ============================================================
# FORMAT DATA
# ============================================================

print()
print("Formatting Instruction Dataset...")

texts = []

for _, row in df.iterrows():

    instruction = str(row["instruction"]).strip()

    input_text = str(row["input"]).strip()

    output = str(row["output"]).strip()

    # Handle empty inputs safely
    if input_text.lower() == "nan":
        input_text = ""

    if input_text != "":
        text = (
            f"<|user|>\n"
            f"{instruction}\n"
            f"{input_text}\n\n"
            f"<|assistant|>\n"
            f"{output}"
        )
    else:
        text = (
            f"<|user|>\n"
            f"{instruction}\n\n"
            f"<|assistant|>\n"
            f"{output}"
        )

    texts.append(text)

print(f"Formatted {len(texts)} training samples.")

dataset = Dataset.from_dict({"text": texts})

# ============================================================
# TOKENIZER
# ============================================================

print()
print(f"Loading Tokenizer: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_function(example):

    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

tokenized_dataset = dataset.map(tokenize_function)

# ============================================================
# MODEL
# ============================================================

print()
print(f"Loading Model: {MODEL_NAME}")

dtype = torch.float16 if DEVICE == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype
)

model.to(DEVICE)

# ============================================================
# LORA CONFIG
# ============================================================

print()
print("Applying LoRA Configuration")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,

    num_train_epochs=EPOCHS,

    logging_steps=10,

    save_steps=100,

    save_total_limit=5,

    fp16=torch.cuda.is_available(),

    report_to="none"
)

# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=tokenized_dataset,

    tokenizer=tokenizer,

    data_collator=data_collator
)

# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

trainer.train()

# ============================================================
# SAVE MODEL
# ============================================================

print()
print("Saving Final Model")

trainer.save_model(f"{OUTPUT_DIR}/final_model")

tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)