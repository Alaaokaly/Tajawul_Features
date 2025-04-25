# User-Based Collaborative Filtering Evaluation Report

**Date:** 2025-04-25 18:36:24

## Dataset Statistics

- Total interactions: 11893
- Number of users: 2016
- Number of unique items: 235
- Training set size: 7308
- Test set size: 4585

## Parameter Tuning Results

### Best Parameters

- k_neighbors: 10
- min_sim: 0.001
- min_overlap: 1

### Best Performance Metrics

- Precision: 0.0053
- Recall: 0.0606
- NDCG: 0.0340
- Hit Rate: 0.1320
- MRR: 0.0102
- Coverage: 0.1191
- Diversity: 0.0107
- Novelty: 22.6054

## Epsilon-Greedy Evaluation

### Best Epsilon Value

- For diversity: 0.5
- For precision: 0.4

### Epsilon Impact Summary

| Epsilon | Precision | Recall | Diversity | Novelty |
|---------|-----------|--------|-----------|--------|
| 0.0 | 0.0053 | 0.0606 | 0.0107 | 22.6054 |
| 0.1 | 0.0051 | 0.0587 | 0.1401 | 23.6015 |
| 0.2 | 0.0053 | 0.0620 | 0.2535 | 24.6114 |
| 0.3 | 0.0058 | 0.0652 | 0.3567 | 25.6092 |
| 0.4 | 0.0058 | 0.0658 | 0.4469 | 26.6208 |
| 0.5 | 0.0058 | 0.0659 | 0.5384 | 26.5037 |

## Conclusion

The evaluation shows that the User-Based Collaborative Filtering approach performs best with k=10, min_sim=0.001, and min_overlap=1. Adding randomness with epsilon=0.5 improves recommendation diversity from 0.0107 to 0.5384, while also maintaining good precision (change of 0.0004).