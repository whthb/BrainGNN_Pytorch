# Retained Same-Input Baseline

This experiment gives the retained RBF-SVM baseline the same sample-level
information as BrainGNN:

- the full `68 x 68` Pearson matrix used as BrainGNN node features;
- the weighted positive top-10% partial-correlation adjacency matrix used as
  BrainGNN edges;
- fixed ROI order, which implicitly represents BrainGNN's constant identity
  position matrix.

The two matrices are flattened and concatenated into 9,248 features. The model
is trained only on the exact BrainGNN training graphs from each fixed
subject-wise fold. Validation graphs are not added to training.

## Result

| Model | Complexity | Balanced accuracy | Difference vs. BrainGNN | Paired p-value |
|---|---:|---:|---:|---:|
| BrainGNN paper-like | 55,719 trainable parameters | 0.7525 +/- 0.0281 | - | - |
| RBF-SVM | 558.2 support vectors | 0.7377 +/- 0.0312 | -0.0148 | 0.14560 |

The RBF-SVM mean balanced accuracy is slightly below BrainGNN. The difference
is not statistically conclusive under the paired five-fold test.

## Interpretation

The RBF-SVM consumes the same information after fixed-order vectorization; it
does not process it with graph message passing. Its support-vector count is
not directly equivalent to a neural trainable-parameter count. The paired
test contains only five folds and should be treated as exploratory.
