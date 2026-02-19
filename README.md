# Multi-Model AI Routing Study

This project studies dynamic routing strategies for multi-model AI systems similar in concept to systems such as ChatGPT. The goal is to design and evaluate a routing layer that selects the most appropriate language model for each input based on estimated difficulty and model confidence.

The router chooses among models of different sizes and computational cost in order to:

- minimize compute and latency
- reduce unnecessary large-model usage
- maintain task accuracy
- produce measurable routing behavior

Multiple routing strategies are implemented and compared experimentally.

---

## Task Setup

To isolate routing behavior from dataset quirks, the project uses a single fixed task.

Task: Text Classification

Examples include:
- sentiment classification
- topic classification

Properties:

- dataset provides class labels (not difficulty labels)
- difficulty is inferred, not given
- routing behavior can be studied cleanly
- results can be logged and visualized consistently

---

## Models Compared

Three pretrained transformer models of increasing size are used:

- distilbert-base-uncased
- bert-base-uncased
- roberta-large

These serve as small, medium, and large model tiers.

---

## Baseline Routing Logic

For each input:

1. Run the small model first
2. Extract prediction confidence score
3. If confidence >= threshold → accept output
4. If confidence < threshold → escalate to medium model
5. If still below threshold → escalate to large model

This produces:

- explicit routing decisions
- measurable escalation behavior
- structured routing logs
- clean plots and comparison tables

---

## Routing Strategies Implemented

### Rule-Based Routing
- fixed confidence threshold
- deterministic escalation behavior

### Confidence-Based Routing
- adaptive thresholds
- dynamic acceptance criteria

### Classifier-Based Routing
- separate classifier predicts input difficulty
- predicted difficulty selects starting model

---

## Evaluation Outputs

The system produces:

- routing decision logs
- model usage frequency
- escalation rates
- accuracy comparisons
- routing performance graphs
- threshold sensitivity analysis

---

## Project Structure

models/
routing/
data/
logs/
notebooks/
reports/

---

## Environment

Python 3.10+
PyTorch
HuggingFace Transformers
scikit-learn
pandas
matplotlib

Install with:

pip install torch transformers datasets scikit-learn pandas matplotlib

---
# COSC 495 — Dynamic Model Routing
Kamran Eisenberg

---

## Week 1 — Project Initialization (Jan 26 – Jan 31)

- Finalized project scope: dynamic routing for multi-model AI systems.
- Defined research objective: minimize computational cost while preserving classification accuracy.
- Selected fixed evaluation task: text classification.
- Identified model hierarchy:
  - Small: distilbert-base-uncased
  - Medium: bert-base-uncased
  - Large: roberta-large
- Outlined initial routing strategy: confidence-threshold-based escalation.
- Established semester roadmap and milestone structure.

---

## Week 2 — Literature Review (Feb 1 – Feb 7)

### Papers Reviewed

- RouteLLM (preference-based model routing)
- WandB LLM Router Guide (practical routing implementation)
- Model Selection for Latency-Critical Inference (Mendoza et al.)
- Dynamic Neural Networks Survey
- BranchyNet (confidence-based early exiting)
- Harder Task Needs More Experts (MoE routing)

### Key Takeaways

- Confidence scores can act as routing signals.
- Threshold-based routing is widely used and empirically validated.
- Harder inputs require more computational resources.
- Dynamic computation reduces unnecessary processing.
- Routing decisions directly affect downstream performance.

### Outcome

- Selected Phase 1 Router: confidence-threshold escalation.
- Defined initial routing design:
  - Start with small model.
  - If confidence < threshold → escalate to medium.
  - If still below threshold → escalate to large.
- Confirmed feasibility of routing without modifying internal model architectures.

---

## Week 3 — Environment Setup & Model Loading (Feb 8 – Feb 14)

### Environment

- Created isolated Python virtual environment.
- Installed required libraries:
  - torch
  - transformers
  - datasets
  - scikit-learn
  - pandas
  - matplotlib
  - seaborn

### Project Structure

- Created organized directory layout:
  - models/
  - scripts/
  - logs/
  - results/
  - router/

### Model Initialization

- Successfully downloaded and loaded:
  - distilbert-base-uncased
  - bert-base-uncased
  - roberta-large
- Verified forward pass execution.
- Implemented confidence extraction using softmax.
- Confirmed models produce logits and probability outputs.

### Outcome

- Infrastructure ready for dataset integration and fine-tuning.
- Confidence scores available for routing logic implementation.

