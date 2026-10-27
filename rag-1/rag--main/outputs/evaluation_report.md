# RAG System Evaluation Report

## Architecture Comparison

| architecture       |   faithfulness |   context_precision |   context_recall |   answer_relevance |   noise_ratio |   iou |   precision_omega |   latency_ms |
|:-------------------|---------------:|--------------------:|-----------------:|-------------------:|--------------:|------:|------------------:|-------------:|
| Naive RAG          |           0.61 |                0.57 |             0.55 |               0.63 |          0.31 |  0.45 |              0.52 |          820 |
| Hybrid Multi-Query |           0.74 |                0.7  |             0.68 |               0.75 |          0.18 |  0.58 |              0.63 |         1150 |
| Parent-Child RAG   |           0.79 |                0.76 |             0.73 |               0.8  |          0.14 |  0.65 |              0.7  |          980 |
| CRAG               |           0.83 |                0.79 |             0.77 |               0.82 |          0.11 |  0.71 |              0.75 |         1300 |
| Prod CRAG + Cache  |           0.89 |                0.83 |             0.85 |               0.86 |          0.08 |  0.78 |              0.82 |           90 |

## Best Architecture

Highest mean **faithfulness**: **Prod CRAG + Cache** (0.8900), n=n/a.

> Ranking by mean alone does not establish a difference. See `significance_tests.csv` for question-paired, Benjamini-Hochberg corrected comparisons against the Naive baseline.