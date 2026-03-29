# Multi-Model AI Routing Study

## COSC 495 — Dynamic Model Routing
Kamran Eisenberg

---

# Project Overview

This project studies dynamic routing strategies for multi-model AI systems. The goal is to design and evaluate a routing layer that selects the most appropriate model for each input based on confidence signals while minimizing computational cost and preserving classification accuracy.

The routing system selects among models of different sizes in order to:

- Minimize average latency
- Reduce unnecessary large-model usage
- Maintain high classification accuracy
- Produce measurable and reproducible routing behavior

Evaluation is conducted using a fixed supervised classification task to ensure structured experimental analysis.

---

# Task Setup

## Task: Sentiment Classification

To isolate routing behavior from dataset artifacts, the project uses a single fixed task:

- Binary sentiment classification
- Supervised dataset with ground-truth labels
- No explicit difficulty labels
- Difficulty inferred through confidence behavior

## Dataset Used

- GLUE SST-2 (Stanford Sentiment Treebank, binary subset)
- Validation split: 872 labeled examples
- Used for controlled routing experiments

SST-2 aligns with the fine-tuned models selected for this study.

---

# Models Compared

Three pretrained, fine-tuned transformer models of increasing size are used:

## Small Model
- distilbert-base-uncased-finetuned-sst-2-english

## Medium Model
- textattack/bert-base-uncased-SST-2

## Large Model
- siebert/sentiment-roberta-large-english

These serve as small, medium, and large model tiers.

All models are:

- Downloaded locally
- Loaded in offline mode
- Executed using PyTorch on Apple MPS
- Evaluated using consistent confidence extraction

---

# Confidence Calculation

For each model:

- Forward pass produces logits
- Softmax(logits) produces probability distribution
- Confidence = max(probabilities)

Important note:

Confidence represents internal class preference strength, not guaranteed correctness.

This project empirically evaluates how well confidence correlates with actual correctness.

---

# Baseline Routing Logic (Implemented)

For each input:

- Run small model
- Extract confidence score
- If confidence ≥ τ_small → accept prediction
- Else escalate to medium model
- If medium confidence ≥ τ_med → accept
- Else escalate to large model

This produces:

- Explicit routing decisions
- Measurable escalation rates
- Structured routing logs
- Latency measurement per model
- Number of models invoked per input

---

# Routing Strategies

## Rule-Based Routing (Implemented)

- Fixed confidence thresholds
- Deterministic escalation
- Fully evaluated on SST-2 validation set

## Threshold Tuning (Next Phase)

- Sensitivity analysis of τ_small and τ_med
- Accuracy vs latency tradeoff curves

## Difficulty Classifier Routing (Planned)

- Separate classifier predicts input difficulty
- Difficulty prediction selects starting model

---

# Evaluation Outputs (Current)

The system produces:

- baseline_results.csv containing:
  - True label
  - Final predicted label
  - Chosen model
  - Confidence score
  - Total latency
  - Number of models used

- Routing decision logs
- Escalation distribution
- Accuracy measurement
- Average latency measurement

This establishes the first complete experimental pipeline.

---

# Project Structure

## scripts/
- routing.py — Escalation router implementation
- run_baseline.py — Runs SST-2 evaluation loop
- analyze_results.py — Computes accuracy and latency metrics

## local_models/
- distilbert-base-uncased-finetuned-sst-2-english
- textattack__bert-base-uncased-SST-2
- siebert__sentiment-roberta-large-english

## results/
- baseline_results.csv

All models are loaded locally using:

- AutoTokenizer.from_pretrained(local_files_only=True)
- AutoModelForSequenceClassification.from_pretrained(local_files_only=True)

No external inference calls are made during evaluation.

---

# Week-by-Week Progress

## Week 1 — Project Initialization

- Finalized project scope: dynamic multi-model routing
- Defined research objective: reduce computational cost while maintaining accuracy
- Selected classification as controlled task
- Defined model hierarchy (small → medium → large)
- Outlined confidence-based escalation strategy

---

## Week 2 — Literature Review

Reviewed:

- RouteLLM
- BranchyNet
- Dynamic Neural Networks survey
- MoE routing strategies
- Latency-aware model selection

Key takeaways:

- Confidence thresholds are widely used as routing signals
- Dynamic computation reduces unnecessary processing
- Routing behavior must be evaluated empirically

---

## Week 3 — Infrastructure and Model Integration

### Environment Setup

- Created Python virtual environment
- Installed torch, transformers, datasets, pandas, matplotlib
- Verified Apple MPS acceleration

### Model Loading

- Downloaded all models locally
- Implemented offline loading
- Verified logits and probability outputs
- Implemented softmax-based confidence extraction

### Escalation Router Implementation

- Built EscalationRouter class
- Implemented:
  - _predict_with_conf
  - route
- Added:
  - Latency tracking
  - Label normalization
  - Decision logging
- Verified correct escalation behavior through manual tests

---

## Week 4 — Dataset Integration and Baseline Evaluation

### Dataset Preparation

- Integrated GLUE SST-2 dataset
- Loaded validation split (872 samples)
- Built structured evaluation loop

### Baseline Experiment

- Ran full routing experiment over validation set
- Logged:
  - True label
  - Final prediction
  - Chosen model
  - Confidence
  - Total latency
  - Models invoked

### Output

- Generated baseline_results.csv
- Established first quantitative routing evaluation

This marks the transition from implementation phase to experimental phase.

---

# Current Status

- Models load locally
- Confidence extracted via softmax
- Escalation router implemented
- SST-2 dataset integrated
- Full baseline experiment executed
- Results logged for analysis

---

# Next Steps

- Compute baseline accuracy
- Compare against always-small and always-large baselines
- Perform threshold sweep
- Generate:
  - Accuracy vs latency plots
  - Large-model usage curves
  - Confidence distribution analysis

---

# Core Research Question

Can confidence-based dynamic routing achieve near-large-model accuracy while substantially reducing average inference cost in a multi-model classification system?