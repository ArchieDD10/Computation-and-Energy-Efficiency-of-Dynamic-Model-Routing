Routing Signals Used:

1. Model confidence score (softmax probability)
2. Escalation thresholds
3. Difficulty classifier (later phase)

Evaluation Metrics:

- accuracy per model tier
- routing accuracy vs single-model baseline
- escalation rate
- model usage frequency
- average confidence per tier

Datasets to choose from:
ag_news        (topic classification)
sst2           (sentiment — very clean)

First Strategy:
- Phase 1 router = confidence threshold escalation
- start_model = small 
- threshold = 0.85 (initial test)