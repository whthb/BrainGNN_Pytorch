# Current-Model HCP Subject-Wise BrainGNN Reproduction Report

## Scope

This report reruns the paper-motivated HCP experiments with the current `07-main_hcp_subjectwise.py` model and the unified five-fold `15-run_hcp_subjectwise_new_baseline.py` entry point.

This is a local-data method reproduction, not an exact numerical reproduction of the paper's HCP result. The local dataset contains 343 partially observed subjects, 1,235 LR/RL run-level graphs, seven tasks, and 68 cortical ROIs; the paper used a different complete-subject cohort and 268 ROIs.

## Protocol

- Completed unique configurations: 15
- Folds: `[0, 1, 2, 3, 4]`; epochs: `100`; batch size: `64`
- Subject-wise deterministic 60/20/20-style train/validation/test split
- Pearson-correlation rows as node features
- Positive top-10% partial-correlation edges
- Adam, learning rate 0.001, weight decay 0.005, step decay every 20 epochs
- Checkpoint selected by validation balanced accuracy
- CUDA deterministic algorithms are not forced; repeated training can vary despite seed 123

## Main Five-Fold Result

| Metric | Mean +/- std |
|---|---:|
| Accuracy | 0.8622 +/- 0.0215 |
| Balanced accuracy | 0.8449 +/- 0.0205 |
| Macro F1 | 0.8430 +/- 0.0234 |
| Macro precision | 0.8443 +/- 0.0255 |

| Fold | Accuracy | Balanced accuracy | Macro F1 |
|---:|---:|---:|---:|
| 0 | 0.8272 | 0.8262 | 0.8138 |
| 1 | 0.8826 | 0.8561 | 0.8550 |
| 2 | 0.8609 | 0.8529 | 0.8567 |
| 3 | 0.8865 | 0.8726 | 0.8726 |
| 4 | 0.8539 | 0.8169 | 0.8171 |

### Classification Baselines

| Method | Balanced accuracy | Macro F1 |
|---|---:|---:|
| Majority class | 0.1429 +/- 0.0000 | 0.0518 +/- 0.0037 |
| Same-input RBF-SVM | 0.7942 +/- 0.0403 | 0.8238 +/- 0.0344 |
| BrainGNN current paper-like | 0.8449 +/- 0.0205 | 0.8430 +/- 0.0234 |

BrainGNN changes mean balanced accuracy by `+0.0507` versus the same-input RBF-SVM (paired five-fold t-test `p=0.06260`).

### Per-Task Recall

| Task | Recall, mean +/- std |
|---|---:|
| EMOTION | 0.9139 +/- 0.0310 |
| GAMBLING | 0.7738 +/- 0.0910 |
| LANGUAGE | 0.9000 +/- 0.0287 |
| MOTOR | 0.6981 +/- 0.0956 |
| RELATIONAL | 0.7919 +/- 0.0487 |
| SOCIAL | 0.9634 +/- 0.0341 |
| WM | 0.8737 +/- 0.0650 |

## ROI-Aware Convolution Ablation

| Convolution | Balanced accuracy |
|---|---:|
| Ra-GConv | 0.8449 +/- 0.0205 |
| Shared-kernel vanilla-GConv | 0.8226 +/- 0.0555 |

Ra-GConv changes mean balanced accuracy by `+0.0224` versus vanilla-GConv (paired five-fold t-test `p=0.31437`).

## Loss Ablation

| Loss | Balanced accuracy | Macro F1 |
|---|---:|---:|
| CE only | 0.8129 +/- 0.0213 | 0.8087 +/- 0.0321 |
| CE + unit | 0.8183 +/- 0.0158 | 0.8176 +/- 0.0191 |
| CE + unit + TPK | 0.8041 +/- 0.0231 | 0.8031 +/- 0.0265 |
| CE + unit + GLC | 0.8601 +/- 0.0204 | 0.8550 +/- 0.0251 |
| Full loss | 0.8449 +/- 0.0205 | 0.8430 +/- 0.0234 |

## Lambda Sweep

| lambda1 TPK | lambda2 GLC | Balanced accuracy |
|---:|---:|---:|
| 0 | 0 | 0.8183 +/- 0.0158 |
| 0 | 0.1 | 0.8601 +/- 0.0204 |
| 0 | 0.2 | 0.8219 +/- 0.0156 |
| 0 | 0.5 | 0.8298 +/- 0.0288 |
| 0 | 1 | 0.8066 +/- 0.0363 |
| 0.1 | 0 | 0.8041 +/- 0.0231 |
| 0.1 | 0.1 | 0.8449 +/- 0.0205 |
| 0.1 | 0.2 | 0.8327 +/- 0.0242 |
| 0.1 | 0.5 | 0.8279 +/- 0.0186 |
| 0.1 | 1 | 0.8266 +/- 0.0262 |
| 0.2 | 0.1 | 0.8528 +/- 0.0169 |
| 0.5 | 0.1 | 0.8377 +/- 0.0364 |

The best tested setting is `tpk_0_glc_0.1` with balanced accuracy `0.8601 +/- 0.0204`.

## Parameter Capacity

| Setting | Trainable parameters | Balanced accuracy |
|---|---:|---:|
| Current default head | 55,719 | 0.8449 +/- 0.0205 |
| Approximately 96k paper-capacity head | 96,039 | 0.8269 +/- 0.0246 |

The approximately 96k setting changes mean balanced accuracy by `-0.0180` (paired five-fold t-test `p=0.24052`).

## Interpretability

| GLC weight | Mean within-task top-17 ROI Jaccard | Community stability | Top-bottom score gap |
|---:|---:|---:|---:|
| 0 | 0.4826 | 0.8614 | 0.3548 |
| 0.1 | 0.8172 | 0.9081 | 0.3429 |
| 0.5 | 0.8832 | 0.9626 | 0.3077 |

Section 3.5-style outputs are under `interpretability_figures/`:

- `figure5_glc_individual_group`: individual/group ROI consistency across GLC weights
- `figure7_task_salient_rois`: task-level mean first-pooling saliency
- `figure8_proxy_task_roi_similarity`: top-ROI Jaccard proxy, not Neurosynth decoding
- `figure9_community_assignments`: strongest qualifying Ra-GConv community per ROI
- `figure10_alpha_positive_heatmap`: first-layer non-negative community weights

The current artifacts save first-pooling scores but not second-pooling ROI mappings. The Fig. 5- and Fig. 7-style maps therefore use first-pooling top-17 scores as a documented proxy. Cortical maps are anatomical schematics, not surface renderings.

## Conclusion

The current-model main experiment reaches balanced accuracy `0.8449 +/- 0.0205` on the local run-level HCP subset. The ablations, lambda sweep, capacity comparison, and interpretation figures above must be interpreted within the documented local-data deviations and are not directly comparable to the paper's reported HCP accuracy.
