# SyBAD: Failure Analysis & Large-Scale Improvement Report

## Executive Summary
This document outlines the systematic failure analysis of the preliminary sycophancy and bias fine-tuning study (SyBAD v1) and provides the architectural, methodological, and dataset design improvements incorporated into SyBAD v2.

---

## 1. Identified Failure Modes in SyBAD v1

### A. Sub-Capacity Model Limitations (TinyLlama-1.1B)
- **Problem**: Small 1B models exhibit weak internal knowledge representations. At this scale, "sycophancy" on elementary facts (e.g., agreeing that $2+2=5$) is primarily a symptom of under-parameterization and incomplete pretraining rather than genuine RLHF-induced sycophancy.
- **Resolution in v2**: Transitioned to **LLaMA-2-7B-Chat** (`meta-llama/Llama-2-7b-chat-hf`), the academic standard workhorse model for sycophancy (Sharma et al., 2023; Rimsky et al., 2023) and social bias benchmarks.

### B. Standard Instruction Fine-Tuning Reinforcing Sycophancy
- **Problem**: Standard instruction-tuning datasets (e.g., Alpaca) optimize for helpfulness and agreeableness without adversarial constraints. Fine-tuning on general instruction pairs inadvertently *amplified* sycophancy across checkpoints.
- **Resolution in v2**: Constructed a **1,500-sample balanced training corpus** explicitly containing:
  1. *Anti-Sycophancy Pairs (600 samples)*: Countering user pressure across Ethical Dilemmas, Political Stances, Sentimental/Emotional Appeals, Philosophical Questions, Subjective Value Judgments, and Career Biases.
  2. *Debiasing Pairs (500 samples)*: Rejecting demographic stereotypes across Gender, Race/Ethnicity, Religion, Age, and Socioeconomic dimensions.
  3. *Fluency Preservation Pairs (400 samples)*: Retaining general instruction-following capabilities to prevent catastrophic forgetting.

### C. Uncalibrated Keyword-Heuristic Scoring
- **Problem**: Earlier evaluation relied on rigid regex/keyword heuristics (`"yes" = sycophantic`, `"no" = corrective`), misclassifying nuanced responses.
- **Resolution in v2**: Adopted **Anthropic's Model-Written Evaluations** protocol (A/B choice matching) and BBQ standardized choice extraction, providing deterministic, unambiguous evaluation without brittle heuristics.

### D. Conflation of Bias and Sycophancy
- **Problem**: Bias and sycophancy were treated as a single compound metric.
- **Resolution in v2**: Separated into two distinct evaluation tracks:
  - **Sycophancy Track**: Evaluated via Anthropic PhilPapers, Political Typology, and NLP Survey datasets.
  - **Bias Track**: Evaluated independently via BBQ (Ambiguous vs. Disambiguated Context QA) and CrowS-Pairs (Log-probability sentence preference).

---

## 2. Experimental Architecture in SyBAD v2

```
Phase 1: Dataset Generation (1,500 Multi-Domain Balanced Pairs)
   ↓
Phase 2: Baseline Benchmark Evaluation (Anthropic Sycophancy + BBQ + CrowS-Pairs)
   ↓
Phase 3: LoRA Parameter-Efficient Fine-Tuning (r=16, alpha=32, target: q/k/v/o)
   ↓
Phase 4: High-Resolution Checkpoint Trajectory Evaluation (step-50, 100, ..., final)
   ↓
Phase 5: Statistical Significance & Longitudinal Metric Tracking
```

---

## 3. Reference Benchmarks

1. **Anthropic Model-Written Evals**: Perez et al., 2022. *Discovering Language Model Behaviors with Model-Written Evaluations*.
2. **Sycophancy Analysis**: Sharma et al., 2023 / ICLR 2024. *Towards Understanding Sycophancy in Language Models*.
3. **Synthetic Anti-Sycophancy Tuning**: Wei et al., 2024. *Simple synthetic data reduces sycophancy in large language models*. Google DeepMind.
4. **BBQ**: Parrish et al., 2022. *BBQ: A Hand-Built Bias Benchmark for Question Answering*.
5. **CrowS-Pairs**: Nangia et al., 2020. *CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models*.
