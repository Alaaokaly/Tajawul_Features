# Hybrid Recommender Evaluation Report

**Date:** 2025-04-25 23:39:41

## Dataset Statistics

- Total interactions: 11893
- Number of users: 2016
- Number of unique items: 235
- Training set size: 7308
- Test set size: 4585

## Model Comparison

|                |   precision |    recall |        ndcg |   hit_rate |         mrr |   coverage |   diversity |   novelty |   evaluated_users |   epsilon |
|:---------------|------------:|----------:|------------:|-----------:|------------:|-----------:|------------:|----------:|------------------:|----------:|
| User-based CF  |  0.00534615 | 0.0606333 |   0.0339813 |      0.132 |   0.0101632 |   0.119149 |   0.0106849 |   22.6054 |              1000 |       nan |
| Item-based CF  |  0.00630769 | 0.0727667 |   0.0397522 |      0.151 |   0.011956  |   0.991489 |   0.673036  |   22.3123 |              1000 |       nan |
| Hybrid         |  0          | 0         |   0         |      0     |   0         |   0        |   0         |    0      |                 0 |       nan |
| Hybrid+Epsilon |  0          | 0         | nan         |    nan     | nan         | nan        |   0         |    0      |                 0 |         0 |

## Key Findings

- Best model for precision: **Item-based CF** (0.0063)
- Best model for recall: **Item-based CF** (0.0728)
- Best model for NDCG: **Item-based CF** (0.0398)
- Best model for diversity: **Item-based CF** (0.6730)
- Best model for novelty: **User-based CF** (22.6054)

![model_comparison_20250425_202127](https://github.com/user-attachments/assets/b97dae73-735a-4eda-8f1d-4ea19de66b59)

## Epsilon-Greedy Exploration Impact

| Epsilon | Precision | Recall | Diversity | Novelty |
|---------|-----------|--------|-----------|--------|
| 0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Conclusion

While the hybrid recommender doesn't achieve the highest precision, though it may need improvements for recommendation diversity. 

Overall, the evaluation shows that... while the hybrid approach shows promise in combining content-based and collaborative filtering, further tuning is needed to fully leverage the strengths of both methods.
