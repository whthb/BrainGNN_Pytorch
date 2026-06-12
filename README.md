# Graph Neural Network for Brain Network Analysis
 A preliminary implementation of BrainGNN. The example presented here is on the public resting-state fMRI ABIDE for the convenience of development. This dataset was different from the ones used in our publication, which are cleaner task-fMRI. Still seeking solutions improve representation learning on the noisy data.


## Usage
### Setup
**pip**

See the `requirements.txt` for environment configuration. 
```bash
pip install -r requirements.txt
```
**PYG**

To install PyG, [please refer to the document](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

**RTX 5070 Ti**

For RTX 50-series GPUs, create a CUDA 12.8 environment and install the modern
dependency set:
```bash
conda create -n braingnn_rtx5070ti python=3.11 pip -y
conda activate braingnn_rtx5070ti
python -m pip install -r requirements-rtx5070ti.txt
```

The modern environment uses `processed/data_pyg2.pt` so that its PyG cache does
not overwrite the legacy `processed/data.pt`.

### Dataset 
**ABIDE**

We treat each fMRI as a brain graph. How to download and construct the graphs?
```
python 01-fetch_data.py
python 02-process_data.py
```

`01-fetch_data.py` automatically retries transient connection, timeout, and TLS
errors while downloading from S3. To limit retries or change the delay:
```
python 01-fetch_data.py --max-retries 20 --retry-delay 10
```

### How to run classification?
Training and testing are integrated in file `03-main.py`. To run
```
python 03-main.py 
```

### Local HCP feasibility study

The local HCP workflow validates BrainGNN behavior on an incomplete 68-ROI
HCP900 task-fMRI subset. It is not a reproduction of the paper's 268-ROI HCP
result. See `plan.md` for the experiment definition and reporting constraints.

The feasibility workflow is organized as follows:

- `08-build_hcp_feasibility_dataset.py` merges available LR/RL runs into one
  graph per subject-task pair.
- `09-build_hcp_feasibility_splits.py` creates fixed subject-wise folds.
- `10-run_hcp_feasibility_baselines.py` and
  `14-run_hcp_same_input_baselines.py` run the retained baselines whose mean
  balanced accuracy is below BrainGNN.
- `11-run_hcp_feasibility_experiments.py` runs the main experiment, ablations,
  and lambda sweep; `12-summarize_hcp_feasibility.py` aggregates their outputs.
- `13-plot_hcp_interpretability.py` generates the local Section 3.5-style
  figures.
- `imports/hcp_feasibility.py`, `imports/hcp_splits.py`, and `net/roi_pool.py`
  provide reusable data, split, and paper-style ROI pooling support.
- `configs/hcp900_feasibility_68roi_subjectwise_5fold.json` stores the fixed
  split; `experiments/hcp900_feasibility_68roi/` stores compact aggregate
  results and figures. Per-fold training artifacts are intentionally ignored.
- `tests/` covers dataset construction, split integrity, and ROI pooling.

Build one graph per available subject-task pair by merging LR/RL runs:

```bash
python 08-build_hcp_feasibility_dataset.py \
  --source-root data/HCP900_subjectwise_qc_allsub_7task_lrrl \
  --output-root data/HCP900_feasibility_68roi_merged
```

Build deterministic subject-wise five-fold splits:

```bash
python 09-build_hcp_feasibility_splits.py \
  --dataroot data/HCP900_feasibility_68roi_merged \
  --output configs/hcp900_feasibility_68roi_subjectwise_5fold.json
```

Run the majority-class baseline and the paper-like BrainGNN configuration:

```bash
python 10-run_hcp_feasibility_baselines.py \
  --dataroot data/HCP900_feasibility_68roi_merged \
  --split-manifest configs/hcp900_feasibility_68roi_subjectwise_5fold.json \
  --output experiments/hcp900_feasibility_68roi/baselines.json

python 11-run_hcp_feasibility_experiments.py \
  --dataroot data/HCP900_feasibility_68roi_merged \
  --split-manifest configs/hcp900_feasibility_68roi_subjectwise_5fold.json \
  --output-root experiments/hcp900_feasibility_68roi/main \
  --experiment main
```

The experiment runner also supports `loss_ablation`, `conv_ablation`, and
`lambda_sweep`.

Generate local equivalents of the interpretation figures in paper Section 3.5:

```bash
python 13-plot_hcp_interpretability.py \
  --experiment-root experiments/hcp900_feasibility_68roi \
  --atlas data/hcp_atlas_workbench/100307.aparc.32k_fs_LR.dlabel.nii \
  --output-dir experiments/hcp900_feasibility_68roi/interpretability_figures
```

The generated Fig. 5- and Fig. 7-style maps use saved first-pooling top-17
scores as a proxy because second-pooling node mappings were not saved. The
Fig. 8 proxy is a task-ROI Jaccard heatmap, not Neurosynth decoding.

Run the retained RBF-SVM baseline using the same Pearson node features and
positive top-10% partial-correlation weighted graph as BrainGNN:

```bash
python 14-run_hcp_same_input_baselines.py \
  --dataroot data/HCP900_feasibility_68roi_merged \
  --split-manifest configs/hcp900_feasibility_68roi_subjectwise_5fold.json \
  --output experiments/hcp900_feasibility_68roi/same_input_baselines.json
```


## Citation
If you find the code and dataset useful, please cite our paper.
```latex
@article{li2020braingnn,
  title={Braingnn: Interpretable brain graph neural network for fmri analysis},
  author={Li, Xiaoxiao and Zhou,Yuan and Dvornek, Nicha and Zhang, Muhan and Gao, Siyuan and Zhuang, Juntang and Scheinost, Dustin and Staib, Lawrence and Ventola, Pamela and Duncan, James},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}
```
