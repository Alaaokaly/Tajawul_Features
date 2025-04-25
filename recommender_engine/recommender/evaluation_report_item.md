# Item-Based Collaborative Filtering Evaluation Report

**Date:** 2025-04-25 19:47:06

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
- min_overlap: 2

### Best Performance Metrics

- Precision: 0.0090
- Recall: 0.0990
- NDCG: 0.0548
- Hit Rate: 0.2070
- MRR: 0.0163
- Coverage: 0.9702
- Diversity: 0.6446
- Novelty: 22.1563

## Epsilon-Greedy Evaluation

### Best Epsilon Value

- For diversity: 0.5
- For precision: 0.2

### Epsilon Impact Summary

| Epsilon | Precision | Recall | Diversity | Novelty |
|---------|-----------|--------|-----------|--------|
| 0.0 | 0.0090 | 0.0990 | 0.6446 | 22.1563 |
| 0.1 | 0.0088 | 0.0957 | 0.6821 | 23.2053 |
| 0.2 | 0.0093 | 0.1009 | 0.7144 | 24.2730 |
| 0.3 | 0.0090 | 0.0998 | 0.7411 | 25.3369 |
| 0.4 | 0.0089 | 0.0977 | 0.7621 | 26.4142 |
| 0.5 | 0.0083 | 0.0917 | 0.7874 | 26.3781 |

## Conclusion

The evaluation shows that the Item-Based Collaborative Filtering approach performs best with k=10, min_sim=0.001, and min_overlap=2. Adding randomness with epsilon=0.5 improves recommendation diversity from 0.6446 to 0.7874, with a small precision trade-off of 0.0007.