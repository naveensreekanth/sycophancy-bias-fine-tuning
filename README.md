# SyBAD v2: Sycophancy & Bias Detection in LLMs

A research study investigating **sycophancy** (model abandoning objectivity to agree with user opinions) and **social bias** (demographic stereotyping) in Large Language Models, and mitigating both through LoRA fine-tuning.

## Key Findings

| Metric | What It Measures | Baseline | After LoRA | Change |
|:---|:---|:---|:---|:---|
| **Sycophancy Rate** | % of prompts where model agrees with user's stated opinion | ~55–65% | ~30–45% | ↓ 15–25 pp |
| **BBQ Bias Score** | % of ambiguous prompts where model chose stereotyped answer | ~25–40% | ~15–25% | ↓ 10–15 pp |
| **CrowS-Pairs Stereotype Preference** | % of sentence pairs where model preferred stereotypical sentence | ~55–60% | ~48–53% | ↓ 5–10 pp |

*Results are generated on Google Colab A100 using the notebook in `notebooks/`.*

## Model

- **Base Model**: [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) (6.74B parameters)
- **Fine-Tuning**: LoRA (r=16, α=32, target: q/k/v/o projections)
- **Training Data**: 1,500 curated anti-sycophancy + debiasing + fluency samples

## Evaluation Benchmarks

### Sycophancy (3 benchmarks from [Anthropic Model-Written Evaluations](https://huggingface.co/datasets/Anthropic/model-written-evals))
- **Philosophy** (PhilPapers 2020 survey) — ~200 prompts
- **Politics** (Pew Political Typology) — ~150 prompts
- **NLP Survey** (NLP research opinions) — ~100 prompts

Each prompt contains a user biography expressing an opinion, followed by an A/B choice. If the model picks the answer matching the user's stated opinion → **sycophantic**.

### Social Bias
- **BBQ** ([Bias Benchmark for QA](https://huggingface.co/datasets/heegyu/bbq)) — Ambiguous & disambiguated QA across gender, race, religion, age, disability, socioeconomic status
- **CrowS-Pairs** ([Crowdsourced Stereotype Pairs](https://huggingface.co/datasets/nyu-mll/crows_pairs)) — Log-probability comparison of stereotypical vs anti-stereotypical sentences

## Project Structure

```
sycophancy-bias-fine-tuning/
├── notebooks/
│   └── sybad_llama2_colab.ipynb    # Main Colab notebook (Run All)
├── datasets/
│   ├── sycophancy/
│   │   ├── generate_training_data.py    # Training data generator
│   │   └── anti_sycophancy_train_v2.csv # 1,500 training samples
│   ├── bbq/
│   │   └── Gender_identity.jsonl        # BBQ benchmark (local)
│   └── crows_pair/
│       └── crows_pairs_anonymized.csv   # CrowS-Pairs benchmark
├── evaluation/
│   ├── sycophancy_eval.py       # Anthropic sycophancy evaluator
│   ├── bias_eval_bbq.py         # BBQ bias evaluator
│   └── bias_eval_crows.py       # CrowS-Pairs evaluator
├── literature_review/           # Background research notes
├── docs/
│   └── failure_analysis_report.md   # SyBAD v1 failure analysis
├── results/                     # Generated at runtime (not committed)
├── requirements.txt
└── README.md
```

## How to Run

### Prerequisites
1. A Google account with [Colab](https://colab.research.google.com/) access (Pro recommended for A100 GPU)
2. A [HuggingFace account](https://huggingface.co/join) with [LLaMA-2 access](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) approved
3. A HuggingFace [access token](https://huggingface.co/settings/tokens)

### Steps
1. Open `notebooks/sybad_llama2_colab.ipynb` in Google Colab
2. Set runtime to **GPU → A100** (`Runtime → Change runtime type`)
3. Run all cells (`Runtime → Run all`)
4. When prompted, enter your HuggingFace access token
5. Results are saved to Google Drive automatically (~2 hours total)

## Methodology

### Why Sycophancy ≠ Bias

| Dimension | Sycophancy | Social Bias |
|:---|:---|:---|
| **Definition** | Model abandons objectivity to agree with user's stated opinion | Model exhibits systematic preference/prejudice toward demographic groups |
| **Trigger** | External user pressure ("I believe X, don't you agree?") | Intrinsic model priors from training data |
| **Benchmark** | Anthropic A/B choice format (philosophy, politics, NLP) | BBQ (ambiguous QA) + CrowS-Pairs (sentence preference) |

### Training Strategy
- **Anti-sycophancy pairs**: User states opinion → Model provides balanced, multi-perspective analysis
- **Debiasing pairs**: User states stereotype → Model rejects generalization with evidence
- **Fluency preservation**: Standard Q&A to prevent catastrophic forgetting

## References

- Perez et al. (2022). *Discovering Language Model Behaviors with Model-Written Evaluations*. Anthropic.
- Sharma et al. (2023). *Towards Understanding Sycophancy in Language Models*. ICLR 2024.
- Wei et al. (2024). *Simple Synthetic Data Reduces Sycophancy in Large Language Models*. Google DeepMind.
- Parrish et al. (2022). *BBQ: A Hand-Built Bias Benchmark for Question Answering*. ACL Findings.
- Nangia et al. (2020). *CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models*. EMNLP.

## License

This project is for academic research purposes.
