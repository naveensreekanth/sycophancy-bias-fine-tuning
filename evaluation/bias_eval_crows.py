"""
CrowS-Pairs Stereotype Preference Evaluator
===========================================
File: evaluation/bias_eval_crows.py

Measures language model log-probability preference for stereotypical vs. anti-stereotypical sentence pairs.
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
from typing import Optional
from datasets import load_dataset


def compute_sentence_log_prob(sentence: str, model, tokenizer, device: str = 'cuda') -> float:
    """Compute the total log probability of a sentence under the model."""
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs['input_ids'][:, 1:].contiguous()

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum().item()


def evaluate_crows(model, tokenizer, device: str = 'cuda', max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Evaluates a model on the CrowS-Pairs bias benchmark.
    """
    print("Loading CrowS-Pairs dataset...")
    df_raw = None

    # Try local file first
    local_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', 'crows_pair', 'crows_pairs_anonymized.csv')
    if os.path.exists(local_csv):
        df_raw = pd.read_csv(local_csv)
    else:
        try:
            dataset = load_dataset("nyu-mll/crows_pairs", split="test")
            df_raw = dataset.to_pandas()
        except Exception:
            try:
                dataset = load_dataset("BigScienceBiasEval/crows_pairs_multilingual", "english", split="test")
                df_raw = dataset.to_pandas()
            except Exception as e:
                print(f"Error loading CrowS-Pairs: {e}")
                return pd.DataFrame()

    if max_samples and len(df_raw) > max_samples:
        df_raw = df_raw.sample(n=max_samples, random_state=42).reset_index(drop=True)

    results = []
    print(f"Evaluating {len(df_raw)} CrowS-Pairs...")

    for i, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
        sent_more = str(row['sent_more'])
        sent_less = str(row['sent_less'])
        bias_type = str(row.get('bias_type', 'unknown'))

        lp_more = compute_sentence_log_prob(sent_more, model, tokenizer, device)
        lp_less = compute_sentence_log_prob(sent_less, model, tokenizer, device)
        prefers_stereo = (lp_more > lp_less)

        results.append({
            "idx": i,
            "bias_type": bias_type,
            "sent_more": sent_more[:150],
            "sent_less": sent_less[:150],
            "log_prob_more": lp_more,
            "log_prob_less": lp_less,
            "prefers_stereotype": prefers_stereo,
        })

    df = pd.DataFrame(results)
    stereo_rate = df['prefers_stereotype'].mean() * 100
    print(f"CrowS-Pairs Stereotype Preference Rate: {stereo_rate:.1f}%")
    return df
