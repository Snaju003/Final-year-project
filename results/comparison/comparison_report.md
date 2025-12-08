# Ensemble Model Comparison Report

**Generated:** 2025-12-09 03:39:15

## Summary

- **Overall Winner:** Ensemble V2
- **Better Metrics:** 4/5

## Performance Metrics

| Metric | Ensemble V1 | Ensemble V2 | Improvement |
|--------|-------------|-------------|-------------|
| Accuracy | 68.07% | 89.70% | +21.63% |
| Precision | 61.73% | 88.16% | +26.43% |
| Recall | 95.61% | 91.79% | -3.81% |
| F1 Score | 75.02% | 89.94% | +14.92% |
| AUC | 73.13% | 97.18% | +24.05% |

## Class-wise Performance

| Class | V1 Accuracy | V2 Accuracy | Improvement |
|-------|-------------|-------------|-------------|
| Real | 40.35% | 87.59% | +47.24% |
| Fake | 95.61% | 91.79% | -3.81% |

## Confusion Matrices

### Ensemble V1
```
             Predicted
             Real    Fake
Actual Real   5009   7404
Actual Fake    549  11943
```

### Ensemble V2
```
             Predicted
             Real    Fake
Actual Real  10873   1540
Actual Fake   1025  11467
```

## Evaluation Time

- **Ensemble V1:** 106.37s
- **Ensemble V2:** 110.15s

## Visualizations

Generated visualizations:
- `metrics_comparison.png` - Overall metrics comparison
- `confusion_matrices.png` - Confusion matrices side-by-side
- `source_comparison.png` - Per-source performance
- `class_accuracy.png` - Class-wise accuracy comparison
