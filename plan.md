# BrainGNN Local HCP Feasibility Study Plan

## Objective

Use the locally available incomplete HCP900 task-fMRI subset to test whether
BrainGNN's classification, ROI-selection, and ROI-community mechanisms behave
as described in the paper.

This is a feasibility and method-validation study. It is not a numerical
reproduction of the paper because the local dataset uses a 68-ROI cortical
atlas and does not contain all seven tasks for every subject.

## Local Dataset

Source dataset:

`data/HCP900_subjectwise_qc_allsub_7task_lrrl`

Observed source characteristics:

- 343 subjects
- 1,235 LR/RL samples
- 944 unique subject-task pairs
- 68 ROIs
- Seven task labels
- Incomplete task coverage and class imbalance

## Execution Checklist

### Phase 1: Reproducible Dataset

- [x] Merge LR and RL runs for each subject-task pair.
- [x] Z-score each run independently before concatenating time series.
- [x] Recompute Pearson correlation and partial correlation after merging.
- [x] Preserve single available runs instead of dropping incomplete subjects.
- [x] Validate graph count, ROI count, finite values, and task distribution.
- [x] Build and save deterministic subject-wise five-fold train/validation/test
      manifests.
- [x] Verify that no subject occurs in more than one split and that every split
      contains all seven classes.

Expected merged dataset size: 944 graphs.

### Phase 2: Paper-Like BrainGNN Configuration

- [x] Use Pearson correlation rows as node features.
- [x] Use positive partial-correlation top-10% edges.
- [x] Use two Ra-GConv and two R-pool layers.
- [x] Use eight communities and hidden dimensions 32/32.
- [x] Use a 0.5 pooling ratio: 68 -> 34 -> 17 ROIs.
- [x] Apply per-graph z-score normalization to R-pool projection scores.
- [x] Use the paper loss:
      `L = L_ce + sum(L_unit) + lambda1 * sum(L_TPK) + lambda2 * L_GLC`.
- [x] Set unit-loss weights to 1 and default `lambda1=lambda2=0.1`.
- [x] Save machine-readable run configuration, split, metrics, predictions,
      pooling scores, and checkpoints.

Paper-like optimizer settings:

- Adam
- 100 epochs
- learning rate 0.001
- halve learning rate every 20 epochs
- weight decay 0.005

### Phase 3: Main Feasibility Experiment

- [x] Run a smoke test on one fold.
- [x] Run five subject-wise folds.
- [x] Select checkpoints using validation balanced accuracy.
- [x] Report mean, standard deviation, and per-fold:
      accuracy, balanced accuracy, macro F1, macro recall, macro precision,
      per-class recall, and confusion matrix.
- [x] Compare against random chance (1/7) and majority-class accuracy.

Primary success criterion:

BrainGNN must clearly exceed chance and majority baselines on balanced accuracy
and macro F1. The paper's reported 94.4% HCP accuracy is not a target.

### Phase 4: Baselines and Ablations

- [x] Compare against the majority-class baseline, which performs worse than
      BrainGNN on the local dataset.
- [x] Retain the RBF-SVM baseline using the same Pearson node features and
      sparse weighted partial-correlation graph; its mean performance is below
      BrainGNN.
- [x] Compare complete Ra-GConv with shared-kernel vanilla-GConv.
- [x] Compare loss configurations:
      CE only, CE + unit, CE + unit + TPK, CE + unit + GLC, and full loss.
- [x] Run the paper-style lambda sweep:
      - `lambda1=0`, vary `lambda2={0,0.1,0.2,0.5,1.0}`
      - `lambda1=0.1`, vary the same lambda2 values
      - `lambda2=0.1`, vary `lambda1={0,0.1,0.2,0.5}`

### Phase 5: Interpretability

- [x] Compare pooling-score separation with and without TPK loss.
- [x] Compare within-task selected-ROI Jaccard similarity for
      `lambda2={0,0.1,0.5}`.
- [x] Export per-task mean ROI scores, selection frequencies, and top 17 ROIs.
- [x] Extract first-layer non-negative Ra-GConv community memberships.
- [x] Measure salient-ROI and community stability across folds.
- [x] Generate local equivalents of the paper Section 3.5 interpretation
      figures, with explicit first-pool and non-Neurosynth limitations.

### Phase 6: Reporting

- [x] Keep exact dataset and split manifests with each result.
- [x] Clearly label results as local-data feasibility results.
- [x] Record deviations from the paper.
- [x] Summarize completed experiments and remaining work in this file.

## Required Reporting Language

Results from this plan must be described as:

> BrainGNN feasibility validation on a local incomplete HCP900 task-fMRI
> subset.

They must not be described as reproducing the paper's HCP accuracy.

## Deviations From the Paper

- Local atlas: 68 cortical ROIs instead of Shen 268 ROIs.
- Local sample coverage: 343 partially observed subjects instead of 506
  subjects with all seven tasks.
- LR/RL runs are merged into one graph per available subject-task pair.
- Class imbalance requires balanced accuracy and macro metrics as primary
  evaluation measures.

## Progress Log

- 2026-06-11: Plan created after auditing the paper, repository, and local
  dataset.
- 2026-06-11: Built and audited the 944-graph merged dataset, saved fixed
  subject-wise splits, implemented paper-style pooling and loss inputs, passed
  five automated tests, completed the majority baseline, and started the main
  five-fold BrainGNN experiment.
- 2026-06-11: Completed the main and class-weighted five-fold experiments,
  Ra-GConv ablation, loss ablation, full paper-style lambda sweep, and
  interpretability stability summaries. See
  `experiments/hcp900_feasibility_68roi/results_summary.md`.
- 2026-06-12: Generated local Section 3.5-style GLC, task saliency, task-ROI
  similarity, community-assignment, and Ra-GConv `alpha+` figures.
- 2026-06-12: Retained only baseline implementations and results whose mean
  balanced accuracy is below BrainGNN.

## Completed Results

- Paper-like BrainGNN balanced accuracy: `0.7525 +/- 0.0281`.
- Class-weighted BrainGNN balanced accuracy: `0.7593 +/- 0.0158`.
- Ra-GConv improves over vanilla-GConv by `0.0430` balanced accuracy
  (`p=0.00996`, paired t-test).
- The tested lambda optimum is the paper default `lambda1=lambda2=0.1`.
- Increasing GLC from `0` to `0.5` increases mean within-task top-17 ROI
  Jaccard similarity from `0.4954` to `0.8376`, while reducing balanced
  accuracy from `0.7500` to `0.6970`.
- Majority-class baseline balanced accuracy: `0.1429 +/- 0.0000`.
- BrainGNN clearly exceeds both retained baselines by mean balanced accuracy.
- With the same node-feature and edge inputs, the retained unweighted RBF-SVM
  reaches `0.7377 +/- 0.0312` balanced accuracy.
