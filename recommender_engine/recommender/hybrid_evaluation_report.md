# Hybrid Recommender Evaluation Report

**Date:** 2025-04-26 02:44:58

## Dataset Statistics

- Total interactions: 11893
- Number of users: 2016
- Number of unique items: 235
- Training set size: 7308
- Test set size: 4585

## Model Comparison

|                |   precision |    recall |        ndcg |   hit_rate |         mrr |   coverage |   diversity |   novelty |   evaluated_users |   epsilon |
|:---------------|------------:|----------:|------------:|-----------:|------------:|-----------:|------------:|----------:|------------------:|----------:|
| User-based CF  |  0.00534615 | 0.0606333 |   0.0339813 |    0.132   |   0.0101632 |   0.119149 |   0.0106849 |   22.6054 |              1000 |       nan |
| Item-based CF  |  0.00630769 | 0.0727667 |   0.0397522 |    0.151   |   0.011956  |   0.991489 |   0.673036  |   22.3123 |              1000 |       nan |
| Hybrid         |  0.0072699  | 0.0740848 |   0.0455873 |    0.15896 |   0.0167568 |   1.87234  |   0.798358  |   19.4285 |               692 |       nan |
| Hybrid+Epsilon |  0.0072699  | 0.0740848 | nan         |  nan       | nan         | nan        |   0.798358  |   19.4285 |               692 |         0 |

## Key Findings

- Best model for precision: **Hybrid** (0.0073)
- Best model for recall: **Hybrid** (0.0741)
- Best model for NDCG: **Hybrid** (0.0456)
- Best model for diversity: **Hybrid** (0.7984)
- Best model for novelty: **User-based CF** (22.6054)

## Epsilon-Greedy Exploration Impact

| Epsilon | Precision | Recall | Diversity | Novelty |
|---------|-----------|--------|-----------|--------|
| 0.0 | 0.0073 | 0.0741 | 0.7984 | 19.4285 |
| 0.1 | 0.0071 | 0.0713 | 0.8371 | 19.3312 |
| 0.2 | 0.0065 | 0.0644 | 0.8549 | 20.1163 |
| 0.3 | 0.0067 | 0.0671 | 0.8681 | 20.9026 |
| 0.4 | 0.0069 | 0.0667 | 0.8777 | 21.7024 |
| 0.5 | 0.0063 | 0.0635 | 0.8885 | 21.7540 |

## Conclusion

The hybrid recommender outperforms both collaborative filtering approaches in terms of precision, and also provides more diverse recommendations. 

Adding randomness with epsilon=0.5 improves recommendation diversity from 0.7984 to 0.8885, with a small precision trade-off of 0.0009.

Overall, the evaluation shows that... the hybrid approach successfully combines the strengths of content-based and collaborative filtering methods, providing superior recommendations compared to either method alone.