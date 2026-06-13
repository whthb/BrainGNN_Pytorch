# HCP Direction-Robust BrainGNN Results

## Protocol

- Strong baseline: `CE + unit + 0.1 GLC`, with TPK disabled.
- Five folds and seeds: `[123, 456, 789]`.
- Each seed controls both the task-direction-stratified subject splits and model initialization.
- Direction-adversarial weight selected on seed `123` validation folds: `0.1`.
- Cross-direction baselines use source-direction train/validation graphs and target-direction test graphs with subject-disjoint splits.
- These direction-stratified results are not directly comparable to the earlier random-split reproduction result.

## Main Three-Seed Results

| Model | Balanced accuracy | Macro F1 | LR/RL pair agreement | Direction probe balanced accuracy |
|---|---:|---:|---:|---:|
| Strong baseline | 0.7972 +/- 0.0486 | 0.7997 +/- 0.0477 | 0.7258 +/- 0.0616 | 0.5884 +/- 0.0413 |
| Direction-adversarial | 0.7921 +/- 0.0560 | 0.7954 +/- 0.0572 | 0.7335 +/- 0.0930 | 0.5764 +/- 0.0326 |

## Cross-Direction Baselines

| Protocol | Balanced accuracy | Macro F1 |
|---|---:|---:|
| LR to RL | 0.5978 +/- 0.0522 | 0.5892 +/- 0.0496 |
| RL to LR | 0.5681 +/- 0.0986 | 0.5530 +/- 0.1107 |

## Paired Baseline Versus Direction-Adversarial Comparison

| Metric | Baseline mean | Adversarial mean | Mean delta | Paired t-test p |
|---|---:|---:|---:|---:|
| balanced_accuracy | 0.7972 | 0.7921 | -0.0051 | 0.51221 |
| macro_f1 | 0.7997 | 0.7954 | -0.0043 | 0.60871 |
| pair_prediction_agreement | 0.7258 | 0.7335 | +0.0077 | 0.67112 |
| direction_probe_balanced_accuracy | 0.5884 | 0.5764 | -0.0120 | 0.32225 |
| absolute_lr_rl_balanced_accuracy_gap | 0.0391 | 0.0320 | -0.0071 | 0.47598 |

The selected direction-adversarial model does not meet the predefined success criteria across all 15 matched seed-fold pairs. Its task balanced accuracy is slightly lower, while the small gains in pair agreement and the reduction in direction decodability are not statistically significant.

The paired t-tests are exploratory because cross-validation folds share training subjects and the 15 seed-fold outcomes are not fully independent.

The cross-direction baselines are substantially below the mixed-direction baseline, establishing a difficult direction-transfer setting. Because source-only training also uses fewer labeled graphs, this comparison alone does not isolate the effect of direction shift.

## Tuning

| Adversarial weight | Validation balanced accuracy | Test balanced accuracy |
|---:|---:|---:|
| 0.01 | 0.8544 +/- 0.0204 | 0.8353 +/- 0.0149 |
| 0.05 | 0.8555 +/- 0.0321 | 0.8090 +/- 0.0265 |
| 0.1 | 0.8655 +/- 0.0331 | 0.8290 +/- 0.0286 |
| 0.2 | 0.8619 +/- 0.0265 | 0.8325 +/- 0.0143 |
