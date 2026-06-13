# HCP LR/RL Pair-Consistency BrainGNN Results

## Protocol

- All initialization seeds use the same subject splits generated with split seed `123`.
- Strong baseline: `CE + unit + 0.1 GLC`, with TPK disabled.
- Pair model adds JS consistency between matched LR/RL task predictions.
- Pair task balance: `sqrt_inverse`; warm-up: start epoch `10`, ramp for `20` epochs.
- Baseline and pair models use the same pair-aware non-inferiority checkpoint rule.
- Selected pair-consistency weight: `0.2`.

## Main Results

| Model | Balanced accuracy | Macro F1 | Pair agreement | Pair macro-task agreement | Pair ensemble BAcc |
|---|---:|---:|---:|---:|---:|
| Strong baseline | 0.7938 +/- 0.0556 | 0.7978 +/- 0.0547 | 0.7418 +/- 0.0572 | 0.7030 +/- 0.0733 | 0.8692 +/- 0.0754 |
| Pair consistency | 0.8058 +/- 0.0505 | 0.8091 +/- 0.0509 | 0.7506 +/- 0.0582 | 0.7159 +/- 0.0813 | 0.8727 +/- 0.0771 |

## Matched Fold Comparisons

| Metric | Baseline | Pair consistency | Mean delta | Paired t-test p |
|---|---:|---:|---:|---:|
| balanced_accuracy | 0.7938 | 0.8058 | +0.0119 | 0.03136 |
| macro_f1 | 0.7978 | 0.8091 | +0.0114 | 0.03119 |
| pair_prediction_agreement | 0.7418 | 0.7506 | +0.0088 | 0.34127 |
| pair_macro_task_agreement | 0.7030 | 0.7159 | +0.0129 | 0.35737 |
| pair_both_correct | 0.7186 | 0.7313 | +0.0128 | 0.21628 |
| pair_prediction_js | 0.1241 | 0.1207 | -0.0034 | 0.13203 |
| pair_ensemble_balanced_accuracy | 0.8692 | 0.8727 | +0.0036 | 0.76179 |
| pair_pool1_jaccard | 0.6718 | 0.6759 | +0.0041 | 0.08373 |
| direction_probe_balanced_accuracy | 0.6011 | 0.5968 | -0.0043 | 0.63458 |

The paired t-tests are exploratory because cross-validation folds share training subjects.

The selected pair-consistency model improves task balanced accuracy and macro F1 across the matched folds. Its aggregate pair-consistency diagnostics move in the intended direction, but those gains are small and not statistically significant. The supported claim is therefore improved task decoding with mild direction-consistency benefits, not strong LR/RL invariance.

## Per-Task Pair Diagnostics

| Task | Agreement baseline | Agreement pair model | Delta | Both-correct delta | Ensemble accuracy delta | JS delta |
|---|---:|---:|---:|---:|---:|---:|
| EMOTION | 0.7821 | 0.8078 | +0.0257 | +0.0228 | +0.0006 | -0.0088 |
| GAMBLING | 0.6352 | 0.6826 | +0.0474 | +0.0501 | +0.0154 | -0.0102 |
| LANGUAGE | 0.9241 | 0.9167 | -0.0074 | -0.0074 | -0.0231 | +0.0022 |
| MOTOR | 0.4056 | 0.4944 | +0.0889 | +0.0889 | -0.0222 | -0.0320 |
| RELATIONAL | 0.7244 | 0.6928 | -0.0317 | -0.0011 | +0.0428 | -0.0003 |
| SOCIAL | 0.8681 | 0.8620 | -0.0061 | -0.0083 | -0.0074 | +0.0020 |
| WM | 0.5819 | 0.5552 | -0.0267 | -0.0194 | +0.0190 | +0.0047 |

## Tuning

| Pair weight | Validation BAcc | Validation pair agreement | Test BAcc |
|---:|---:|---:|---:|
| 0.05 | 0.8556 +/- 0.0209 | 0.7958 +/- 0.0521 | 0.8397 +/- 0.0191 |
| 0.1 | 0.8634 +/- 0.0216 | 0.7999 +/- 0.0355 | 0.8348 +/- 0.0237 |
| 0.2 | 0.8664 +/- 0.0231 | 0.8034 +/- 0.0545 | 0.8371 +/- 0.0250 |
| 0.5 | 0.8596 +/- 0.0197 | 0.8018 +/- 0.0441 | 0.8334 +/- 0.0281 |
