# Section 3.5 Local Interpretation Figures

These figures are local-data equivalents of the interpretation figures in
BrainGNN Section 3.5. They are not exact reproductions of the paper figures.

- `figure5_glc_individual_group`: compares three correctly classified samples
  from the same task across `lambda2_GLC = 0, 0.1, 0.5`. Blue rings denote ROIs
  common to all three samples in a row.
- `figure7_task_salient_rois`: displays per-task mean first-pooling scores.
- `figure8_proxy_task_roi_similarity`: displays top-17 ROI-set Jaccard
  similarity. It is not the paper's Neurosynth decoding analysis.
- `figure9_community_assignments`: displays the strongest qualifying community
  for each ROI from `fold_2`. The paper's threshold
  `alpha_iu > mean(alpha_i) + std(alpha_i)` is applied. Overlapping memberships
  are retained in `interpretability_figure_data.json`, but only the strongest
  qualifying community can be displayed as one node color.
- `figure10_alpha_positive_heatmap`: directly visualizes the saved first-layer
  non-negative Ra-GConv community-membership matrix.

Important limitations:

- Training outputs save first R-pool scores only. The Fig. 5- and Fig. 7-style
  maps therefore use first-layer top-17 ROIs as a proxy for the nodes
  remaining after the second R-pool layer.
- The double-hemisphere drawing is a schematic layout using the real
  Desikan-Killiany ROI names; it is not a surface-coordinate rendering.
- Neurosynth masks and decoding scores are unavailable, so the paper's Fig. 8
  cannot be reproduced from the current artifacts.
