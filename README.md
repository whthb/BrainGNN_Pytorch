# BrainGNN_Pytorch

Compact course-project repository for reproducing BrainGNN on HCP task-fMRI
subject-wise classification and extending it with LR/RL direction-robustness
experiments.

GitHub repository: <https://github.com/whthb/BrainGNN_Pytorch>

## Repository Structure

```text
BrainGNN_Pytorch/
├── README.md
├── requirements.txt
├── requirements-rtx5070ti.txt
├── net/
├── 07-main_hcp_subjectwise.py
├── 13-plot_hcp_interpretability.py
├── 14-run_hcp_same_input_baselines.py
├── 15-run_hcp_subjectwise_new_baseline.py
├── 16-report_hcp_subjectwise_reproduction.py
├── 17-run_hcp_direction_robustness.py
├── 18-run_hcp_pair_consistency.py
├── experiments/
│   ├── hcp900_subjectwise_paper_reproduction_current_20260612/
│   ├── hcp900_direction_robustness_20260612/
│   └── hcp900_pair_consistency_20260613/
└── reports/
    └── braingnn_course_report/
        ├── main.tex
        ├── build.sh
        ├── figures/
        └── sections/
```

`experiments/` keeps compact summaries, aggregate metrics, protocols, split
manifests, and generated reports. Large per-fold training outputs under
`experiments/**/runs/`, raw datasets, logs, model checkpoints, and local LaTeX
build products are intentionally ignored.

## Environment

For an RTX 50-series GPU with CUDA 12.8:

```bash
conda create -n braingnn_rtx5070ti python=3.11 pip -y
conda activate braingnn_rtx5070ti
python -m pip install -r requirements-rtx5070ti.txt
```

The generic dependency list is kept in `requirements.txt`; the RTX-specific
file pins the CUDA 12.8 PyTorch/PyG wheels used in the local experiments.

## Dataset

The HCP data are not committed to this repository. Experiment scripts expect
the local run-level graph dataset at:

```text
data/HCP900_subjectwise_qc_allsub_7task_lrrl/
  task_label_map.json
  sample_metadata.csv
  subjects/*.h5
```

Each graph uses Pearson-correlation rows as node features and positive top-10%
partial-correlation edges. Splits are deterministic and subject-wise, so one
subject never appears in more than one train, validation, or test partition
within the same fold.

## Paper Reproduction

Run the five-fold BrainGNN reproduction matrix:

```bash
conda run --no-capture-output -n braingnn_rtx5070ti \
  python -u 15-run_hcp_subjectwise_new_baseline.py \
  --experiment all \
  --jobs 3 \
  --output-dir experiments/hcp900_subjectwise_paper_reproduction_current_20260612
```

The matrix covers the main BrainGNN setting, loss ablations, convolution
ablation, `lambda1_TPK` / `lambda2_GLC` scanning, and capacity comparison.

Run the same-input RBF-SVM baseline on the exported subject splits:

```bash
ROOT=experiments/hcp900_subjectwise_paper_reproduction_current_20260612

conda run --no-capture-output -n braingnn_rtx5070ti \
  python 14-run_hcp_same_input_baselines.py \
  --dataroot data/HCP900_subjectwise_qc_allsub_7task_lrrl \
  --split-manifest "$ROOT/split_manifest.json" \
  --output "$ROOT/baselines/same_input_rbf_svm.json"
```

Generate compact reproduction tables:

```bash
conda run --no-capture-output -n braingnn_rtx5070ti \
  python 16-report_hcp_subjectwise_reproduction.py \
  --experiment-root "$ROOT"
```

Final reproduction results are stored in:

```text
experiments/hcp900_subjectwise_paper_reproduction_current_20260612/
  REPORT.md
  protocol.json
  split_manifest.json
  summary.json
  summary.csv
  report_data.json
  aggregates/*.json
  baselines/*.json
```

## Interpretability Figures

The report figures are generated from saved experiment summaries and copied
into `reports/braingnn_course_report/figures/`. If the full per-fold `runs/`
outputs are available locally, figures can be regenerated with:

```bash
MPLCONFIGDIR=.tmp/matplotlib \
conda run --no-capture-output -n braingnn_rtx5070ti \
  python 13-plot_hcp_interpretability.py \
  --experiment-root experiments/hcp900_subjectwise_paper_reproduction_current_20260612 \
  --atlas data/hcp_atlas_workbench/100307.aparc.32k_fs_LR.dlabel.nii \
  --output-dir reports/braingnn_course_report/figures
```

## Direction-Robustness Extension

Run the LR/RL direction-adversarial experiment:

```bash
conda run --no-capture-output -n braingnn_rtx5070ti \
  python -u 17-run_hcp_direction_robustness.py \
  --phase all \
  --jobs 3 \
  --output-dir experiments/hcp900_direction_robustness_20260612
```

The compact results are in
`experiments/hcp900_direction_robustness_20260612/REPORT.md`,
`summary.json`, `protocol.json`, `tuning_selection.json`, and `splits/`.

Run the LR/RL pair-consistency experiment:

```bash
conda run --no-capture-output -n braingnn_rtx5070ti \
  python -u 18-run_hcp_pair_consistency.py \
  --phase all \
  --jobs 1 \
  --output-dir experiments/hcp900_pair_consistency_20260613
```

The compact results are in
`experiments/hcp900_pair_consistency_20260613/REPORT.md`, `summary.json`,
`protocol.json`, `tuning_selection.json`, and `splits/`.

## Single-Fold Debugging

For one BrainGNN fold:

```bash
conda run --no-capture-output -n braingnn_rtx5070ti \
  python -u 07-main_hcp_subjectwise.py \
  --dataroot data/HCP900_subjectwise_qc_allsub_7task_lrrl \
  --fold 0 \
  --n_epochs 100 \
  --batchSize 64 \
  --edge_source pcorr \
  --edge_top_percent 0.10 \
  --positive_edges_only \
  --best_metric balanced_acc \
  --output_dir experiments/debug_hcp_fold0
```

## Course Report

The final LaTeX report is under `reports/braingnn_course_report/`.

```bash
cd reports/braingnn_course_report
./build.sh
```

The compiled PDF is written to `reports/braingnn_course_report/build/main.pdf`.

## Citation

```bibtex
@article{li2020braingnn,
  title={Braingnn: Interpretable brain graph neural network for fmri analysis},
  author={Li, Xiaoxiao and Zhou, Yuan and Dvornek, Nicha and Zhang, Muhan and Gao, Siyuan and Zhuang, Juntang and Scheinost, Dustin and Staib, Lawrence and Ventola, Pamela and Duncan, James},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}
```
