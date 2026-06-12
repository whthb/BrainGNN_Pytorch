# Local HCP900 BrainGNN Feasibility Results

These results validate BrainGNN behavior on the local incomplete 68-ROI HCP900
task-fMRI subset. They do not reproduce the paper's 268-ROI HCP result.

## Dataset and Evaluation

- 343 subjects
- 944 merged subject-task graphs
- Seven classes
- Fixed nested subject-wise five-fold splits
- Primary metric: balanced accuracy
- Retained baselines: majority-class prediction and same-input unweighted
  RBF-SVM

## Classification Results

| Method | Balanced accuracy, mean +/- std |
|---|---:|
| Chance / majority | 0.1429 +/- 0.0000 |
| BrainGNN paper-like | 0.7525 +/- 0.0281 |
| BrainGNN class-weighted | 0.7593 +/- 0.0158 |

BrainGNN clearly exceeds the retained chance / majority baseline.

## Retained Same-Input Baseline

The same-input experiment concatenates the full Pearson node-feature matrix
and the weighted positive top-10% partial-correlation adjacency matrix in
fixed ROI order. The RBF-SVM uses the same subject-wise folds and only the
BrainGNN training graphs.

| Method | Balanced accuracy | Difference vs. paper-like BrainGNN |
|---|---:|---:|
| BrainGNN paper-like | 0.7525 +/- 0.0281 | - |
| RBF-SVM | 0.7377 +/- 0.0312 | -0.0148 |

The retained unweighted RBF-SVM is slightly worse than BrainGNN. The paired
five-fold difference is not statistically conclusive (`p=0.14560`). See
`same_input_baselines_summary.md` for details.

## ROI-Aware Convolution Ablation

| Convolution | Balanced accuracy, mean +/- std |
|---|---:|
| Ra-GConv | 0.7634 +/- 0.0331 |
| Shared-kernel vanilla-GConv | 0.7204 +/- 0.0370 |

The paired five-fold balanced-accuracy difference is +0.0430 in favor of
Ra-GConv (`p=0.00996`, paired t-test).

## Loss Ablation

| Loss | Balanced accuracy, mean +/- std |
|---|---:|
| CE only | 0.7546 +/- 0.0312 |
| CE + unit | 0.7561 +/- 0.0162 |
| CE + unit + TPK | 0.7578 +/- 0.0180 |
| CE + unit + GLC | 0.7438 +/- 0.0432 |
| Full loss | 0.7508 +/- 0.0267 |

TPK provides a small classification improvement. GLC trades classification
performance for more consistent ROI selection.

## Lambda Sweep

The best tested setting is the paper default:

- `lambda1_TPK=0.1`
- `lambda2_GLC=0.1`
- balanced accuracy `0.7566 +/- 0.0235`

Large regularization values reduce performance:

- `lambda1_TPK=0.5, lambda2_GLC=0.1`: balanced accuracy `0.7261`
- `lambda1_TPK=0.1, lambda2_GLC=0.5`: balanced accuracy `0.6970`
- `lambda1_TPK=0.1, lambda2_GLC=1.0`: balanced accuracy `0.6957`

## Interpretability Behavior

| GLC weight | Balanced accuracy | Mean within-task top-17 ROI Jaccard | Community stability |
|---:|---:|---:|---:|
| 0.0 | 0.7500 | 0.4954 | 0.9459 |
| 0.1 | 0.7566 | 0.7745 | 0.9745 |
| 0.5 | 0.6970 | 0.8376 | 0.9928 |

Increasing GLC strongly increases ROI-selection and community consistency, but
an overly large value substantially harms classification. This matches the
qualitative behavior described in the paper.

The final top-versus-bottom pooling-score gap changes only slightly with TPK
(`0.3535` without regularizers versus `0.3539` with TPK). The expected TPK
score-separation effect is therefore not strongly demonstrated by the final
checkpoint statistic on this dataset.

## Section 3.5-Style Figures

Local equivalents of the paper's interpretation figures are available under
`interpretability_figures/`.

- In the Fig. 5-style comparison of three correctly classified SOCIAL samples,
  increasing GLC from `0` to `0.1` and `0.5` increases mean pairwise top-17 ROI
  Jaccard from `0.575` to `0.676` and `0.760`. The number of ROIs common to all
  three samples increases from `10` to `12` and `14`.
- The Fig. 7-style maps show substantial overlap among task-specific salient
  ROI sets, consistent with the limited task specificity in the numerical
  interpretation summary.
- The best-fold first-layer Ra-GConv `alpha+` matrix used in the Fig. 9- and
  Fig. 10-style plots has a zero fraction of `0.570`, demonstrating sparse
  community-membership weights.

The Fig. 5- and Fig. 7-style maps use first-pooling top-17 scores as a proxy
because second-pooling node mappings were not saved. The Fig. 8 proxy reports
cross-task ROI-set similarity and is not the paper's Neurosynth decoding
analysis. The cortical drawings are labeled anatomical schematics rather than
surface-coordinate renderings.

## Conclusion

The local study supports the feasibility of BrainGNN's ROI-aware convolution
and GLC-controlled group-level interpretation. BrainGNN's mean balanced
accuracy exceeds both retained baselines.
