# HCP900 Subject-Wise New Baseline

This is the retrained current BrainGNN baseline on the complete run-level HCP
dataset.

## Protocol

- 343 subjects and 1235 graphs
- deterministic subject-wise five-fold split
- Pearson-correlation rows as node features
- positive top-10% partial-correlation edges
- current Ra-GConv and per-graph standardized ROI pooling
- 100 epochs, batch size 64, Adam learning rate 0.001
- checkpoint selected by validation balanced accuracy

## Results

| Fold | Accuracy | Balanced accuracy | Macro F1 |
|---:|---:|---:|---:|
| 0 | 0.7942 | 0.7919 | 0.7782 |
| 1 | 0.8565 | 0.8070 | 0.8186 |
| 2 | 0.8421 | 0.8425 | 0.8409 |
| 3 | 0.8734 | 0.8578 | 0.8597 |
| 4 | 0.8764 | 0.8408 | 0.8367 |
| Mean | 0.8485 | 0.8280 | 0.8268 |

The historical June 6 model had mean balanced accuracy `0.7888`. The current
model improves it by `0.0392`.

CUDA deterministic algorithms are not enabled, so repeated runs can differ
slightly despite using seed 123.

Model checkpoints and TensorBoard event files remain local because they are
ignored by Git.
