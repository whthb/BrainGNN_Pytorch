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
