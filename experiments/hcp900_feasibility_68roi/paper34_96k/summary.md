# Paper Section 3.4 Parameter-Count Experiment

Section 3.2 specifies the HCP graph architecture with two convolution and
pooling layers, `K=8`, and `d1=d2=32`. Section 3.4 Table 2 reports about
`96k` trainable parameters for HCP BrainGNN.

The local 68-ROI model has only 55,719 parameters with `fc_dim=512`. This
experiment preserves the paper-specified graph architecture and changes only
the classification-head width to `fc_dim=1472`, producing 96,039 trainable
parameters. All other settings, data, and subject-wise folds match the
paper-like local main experiment.

| Setting | Parameters | Balanced accuracy |
|---|---:|---:|
| Local paper-like BrainGNN | 55,719 | 0.7525 +/- 0.0281 |
| Section 3.4 approximately 96k | 96,039 | 0.7694 +/- 0.0173 |

## Fold-Wise Balanced Accuracy

| Fold | 55,719 parameters | 96,039 parameters | Difference |
|---|---:|---:|---:|
| 0 | 0.7154 | 0.7871 | +0.0717 |
| 1 | 0.7506 | 0.7925 | +0.0419 |
| 2 | 0.8002 | 0.7552 | -0.0450 |
| 3 | 0.7592 | 0.7625 | +0.0033 |
| 4 | 0.7373 | 0.7496 | +0.0123 |

The mean balanced-accuracy change is `+0.0169`, with improvement in four of
five folds. The paired five-fold comparison gives `p=0.438`, so the increase
is not statistically conclusive.

Increasing the classification-head capacity improves the mean and reduces
fold-to-fold test variation, but it also raises final-epoch training balanced
accuracy in most folds. The inconsistent fold-wise gain indicates that
parameter count alone is not the main limitation of the local BrainGNN
experiment.
