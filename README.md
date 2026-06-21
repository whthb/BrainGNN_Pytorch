# BrainGNN_Pytorch

本仓库用于课程项目中的 BrainGNN 复现与改进实验，主要内容包括 HCP
task-fMRI subject-wise 脑图分类复现，以及针对 HCP LR/RL 采集方向差异的方向鲁棒性实验。

GitHub 仓库地址：<https://github.com/whthb/BrainGNN_Pytorch>

## 仓库结构

```text
BrainGNN_Pytorch/
├── README.md
├── requirements.txt
├── requirements-rtx5070ti.txt
├── net/
│   ├── braingnn.py
│   ├── braingraphconv.py
│   ├── brainmsgpassing.py
│   ├── inits.py
│   └── roi_pool.py
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
├── reports/
│   └── braingnn_course_report/
│       ├── main.tex
│       ├── build.sh
│       ├── figures/
│       └── sections/
└── .gitignore
```

`experiments/` 中保留的是 compact summaries、aggregate metrics、protocols、
split manifests 和生成的实验报告。原始数据、逐折训练输出、TensorBoard 日志、
模型 checkpoint、本地临时文件以及课程报告编译产物不纳入 GitHub 仓库。

## 环境配置

对于 RTX 50 系列 GPU 和 CUDA 12.8，可使用：

```bash
conda create -n braingnn_rtx5070ti python=3.11 pip -y
conda activate braingnn_rtx5070ti
python -m pip install -r requirements-rtx5070ti.txt
```

通用依赖列表见 `requirements.txt`；`requirements-rtx5070ti.txt` 固定了本地实验使用的
CUDA 12.8 PyTorch / PyG wheel 版本。

## 数据集

HCP 原始数据和处理后的图数据不上传至本仓库。实验脚本默认期望本地数据目录为：

```text
data/HCP900_subjectwise_qc_allsub_7task_lrrl/
  task_label_map.json
  sample_metadata.csv
  subjects/*.h5
```

每个图样本以 Pearson correlation 行向量作为节点特征，并使用 positive top-10\%
partial correlation edges 构建稀疏边。划分采用 deterministic subject-wise split，
同一被试不会同时出现在同一 fold 的训练集、验证集和测试集中。

## 论文复现实验

运行五折 BrainGNN 复现实验矩阵：

```bash
conda run --no-capture-output -n braingnn_rtx5070ti \
  python -u 15-run_hcp_subjectwise_new_baseline.py \
  --experiment all \
  --jobs 3 \
  --output-dir experiments/hcp900_subjectwise_paper_reproduction_current_20260612
```

该矩阵包括主模型结果、损失函数消融、卷积层消融、`lambda1_TPK` / `lambda2_GLC`
扫描以及模型容量对比。

在相同 subject-wise split 上运行 RBF-SVM baseline：

```bash
ROOT=experiments/hcp900_subjectwise_paper_reproduction_current_20260612

conda run --no-capture-output -n braingnn_rtx5070ti \
  python 14-run_hcp_same_input_baselines.py \
  --dataroot data/HCP900_subjectwise_qc_allsub_7task_lrrl \
  --split-manifest "$ROOT/split_manifest.json" \
  --output "$ROOT/baselines/same_input_rbf_svm.json"
```

生成 compact reproduction report：

```bash
conda run --no-capture-output -n braingnn_rtx5070ti \
  python 16-report_hcp_subjectwise_reproduction.py \
  --experiment-root "$ROOT"
```

最终复现实验结果保存在：

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

## 可解释性图示

若本地保留完整逐折训练输出 `runs/` 和 atlas 文件，可重新生成可解释性图示。报告中使用的最终图示保存在
`reports/braingnn_course_report/figures/`：

```bash
MPLCONFIGDIR=.tmp/matplotlib \
conda run --no-capture-output -n braingnn_rtx5070ti \
  python 13-plot_hcp_interpretability.py \
  --experiment-root experiments/hcp900_subjectwise_paper_reproduction_current_20260612 \
  --atlas data/hcp_atlas_workbench/100307.aparc.32k_fs_LR.dlabel.nii \
  --output-dir reports/braingnn_course_report/figures
```

## LR/RL 方向鲁棒性实验

运行方向对抗训练实验：

```bash
conda run --no-capture-output -n braingnn_rtx5070ti \
  python -u 17-run_hcp_direction_robustness.py \
  --phase all \
  --jobs 3 \
  --output-dir experiments/hcp900_direction_robustness_20260612
```

结果保存在 `experiments/hcp900_direction_robustness_20260612/`，其中包括
`REPORT.md`、`summary.json`、`protocol.json`、`tuning_selection.json` 和 `splits/`。

运行 LR/RL 配对一致性实验：

```bash
conda run --no-capture-output -n braingnn_rtx5070ti \
  python -u 18-run_hcp_pair_consistency.py \
  --phase all \
  --jobs 1 \
  --output-dir experiments/hcp900_pair_consistency_20260613
```

结果保存在 `experiments/hcp900_pair_consistency_20260613/`，其中包括
`REPORT.md`、`summary.json`、`protocol.json`、`tuning_selection.json` 和 `splits/`。

## 单折调试

直接运行一个 BrainGNN fold：

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

## 课程报告

课程报告源码位于 `reports/braingnn_course_report/`。该目录包含 LaTeX 主文件、章节文件、
报告图示和编译脚本。编译方式为：

```bash
cd reports/braingnn_course_report
./build.sh
```

生成的 PDF 位于 `reports/braingnn_course_report/build/main.pdf`。`build/` 为本地编译产物，
不纳入 GitHub 仓库。

## 引用

```bibtex
@article{li2020braingnn,
  title={Braingnn: Interpretable brain graph neural network for fmri analysis},
  author={Li, Xiaoxiao and Zhou, Yuan and Dvornek, Nicha and Zhang, Muhan and Gao, Siyuan and Zhuang, Juntang and Scheinost, Dustin and Staib, Lawrence and Ventola, Pamela and Duncan, James},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}
```
