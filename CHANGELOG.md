# 变更日志

## 2026-06-03 - 论文 3.3 节超参数选择实验

### 实验脚本

- 新增 `06-paper33_hparam_sweep.py`。
- 功能：
  - 按论文 3.3 节网格运行 TPK/GLC 正则权重实验。
  - 论文 `lambda1` 对应当前代码的 `--lamb3` 和 `--lamb4`。
  - 论文 `lambda2` 对应当前代码的 `--lamb5`。
  - 自动保存每个 run 的 stdout、TensorBoard log、checkpoint。
  - 自动解析 best validation accuracy、best validation balanced accuracy、final test 指标，输出 `summary.json` 和 `summary.csv`。

### 运行配置

- 数据目录：`data/ABIDE_pcp/cpac/filt_noglobal_qc_fd020_perc20`。
- PyG cache：`data_pcorr_pos_top10pct.pt`。
- 图构造：`edge_source=pcorr`、`edge_top_percent=0.10`、`positive_edges_only=True`。
- 模型：`dim1=32`、`dim2=32`、`fc_dim=512`、`dropout=0.5`。
- 优化：`lr=0.001`、`stepsize=20`、`gamma=0.5`、`weightdecay=0.005`、`batchSize=200`。
- 训练：fold 0，100 epochs，按 validation accuracy 保存 checkpoint。
- 输出目录：`experiments/paper33_hparam_qc_pcorr_pos_top10_fold0_20260603_2310`。

### Fold 0 结果

固定 `lambda1=0`，扫描 `lambda2`：

| lambda1 | lambda2 | best val acc | best val bacc | final test acc | final test bacc |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.6282 | 0.6037 | 0.5192 | 0.4910 |
| 0 | 0.05 | 0.6346 | 0.6004 | 0.5192 | 0.4850 |
| 0 | 0.1 | 0.6474 | 0.6179 | 0.5449 | 0.5095 |
| 0 | 0.2 | 0.6282 | 0.5977 | 0.5128 | 0.4643 |
| 0 | 0.5 | 0.6154 | 0.5922 | 0.5385 | 0.5127 |
| 0 | 1.0 | 0.6667 | 0.6427 | 0.5513 | 0.5212 |

固定 `lambda1=0.1`，扫描 `lambda2`：

| lambda1 | lambda2 | best val acc | best val bacc | final test acc | final test bacc |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0 | 0.6474 | 0.6224 | 0.5192 | 0.4940 |
| 0.1 | 0.05 | 0.6538 | 0.6267 | 0.5449 | 0.5110 |
| 0.1 | 0.1 | 0.6282 | 0.6022 | 0.4936 | 0.4605 |
| 0.1 | 0.2 | 0.6346 | 0.6004 | 0.5705 | 0.5310 |
| 0.1 | 0.5 | 0.6474 | 0.6179 | 0.5577 | 0.5210 |
| 0.1 | 1.0 | 0.6218 | 0.5965 | 0.5705 | 0.5385 |

固定 `lambda2=0.1`，扫描 `lambda1`：

| lambda1 | lambda2 | best val acc | best val bacc | final test acc | final test bacc |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.1 | 0.6474 | 0.6179 | 0.5449 | 0.5095 |
| 0.05 | 0.1 | 0.6538 | 0.6299 | 0.5385 | 0.5052 |
| 0.1 | 0.1 | 0.6282 | 0.6022 | 0.4936 | 0.4605 |
| 0.2 | 0.1 | 0.6538 | 0.6267 | 0.5192 | 0.4925 |
| 0.5 | 0.1 | 0.6218 | 0.6009 | 0.5128 | 0.4748 |

### 观察

- fold 0 上，按论文 validation accuracy 口径，最优点是 `lambda1=0, lambda2=1.0`，best validation accuracy 为 `0.6667`。
- 论文默认式的 `lambda1=0.1, lambda2=0.1` 在本 ABIDE QC fold 0 上不占优，best validation accuracy 为 `0.6282`，final test balanced accuracy 只有 `0.4605`。
- validation accuracy 与 final test balanced accuracy 仍明显脱节；即使 best validation accuracy 达到 `0.6667`，对应 final test balanced accuracy 也只有 `0.5212`。
- 当前结果只代表 fold 0，不足以确定 ABIDE 上的最优正则权重；若要严格复现论文 3.3 节口径，应扩展到 5-fold 后报告 mean/std。

## 2026-06-03 - 论文风格 positive pcorr top10% 实验记录

### 图构造与训练入口

- 扩展 `read_abide_stats_parall.py` / `ABIDEDataset`：
  - 支持 `edge_top_percent`。
  - 支持 `positive_edges_only`。
  - `edge_top_percent=0.10` 表示每个 ROI 保留 `ceil((N-1)*0.10)` 条最强正边，再对称化。
- 扩展 `03-main.py`：
  - 增加 `--edge_top_percent`。
  - 增加 `--positive_edges_only`。
  - 增加 `--best_metric {loss,acc,balanced_acc}`。
  - 输出 train/validation balanced accuracy。
- 扩展 `net/braingnn.py`：
  - 增加 `dim1`、`dim2`、`fc_dim`、`dropout` 参数，默认值保持原模型不变。

### PyG Cache

- 数据目录：`data/ABIDE_pcp/cpac/filt_noglobal_qc_fd020_perc20`。
- cache：`processed/data_pcorr_pos_top10pct.pt`。
- 配置：
  - `edge_source=pcorr`
  - `edge_top_percent=0.10`
  - `positive_edges_only=True`
- 结果：777 个图，标签分布为 0 类 433 个、1 类 344 个。
- 边权：全为正且 finite，min/max 为 `1.93e-07 / 0.9613`。
- 每图边数 min/mean/max：`2148 / 4798.20 / 5288`。

### 训练结果

- 原模型，`lr=0.001`，按 validation balanced accuracy 保存：
  - model: `model_qc_pcorr_pos_top10_lr001_bacc_20260603`
  - log: `log_qc_pcorr_pos_top10_lr001_bacc_20260603`
  - 训练中最高 validation balanced accuracy 约 `0.6687`。
  - final test: accuracy `0.5449`, balanced accuracy `0.5110`, loss `1.1003`。

- 原模型，`lr=0.001`，按 validation loss 保存：
  - model: `model_qc_pcorr_pos_top10_lr001_loss_20260603`
  - log: `log_qc_pcorr_pos_top10_lr001_loss_20260603`
  - final test: accuracy `0.5513`, balanced accuracy `0.5272`, loss `0.9979`。

- 小模型，`dim1=16`、`dim2=16`、`fc_dim=128`、`dropout=0.6`，`lr=0.001`，按 validation loss 保存：
  - model: `model_qc_pcorr_pos_top10_small_lr001_loss_20260603`
  - log: `log_qc_pcorr_pos_top10_small_lr001_loss_20260603`
  - final test: accuracy `0.5321`, balanced accuracy `0.4920`, loss `1.0061`。

- 小模型长训练，`dim1=16`、`dim2=16`、`fc_dim=128`、`dropout=0.6`，`lr=0.001`，`n_epochs=300`，`stepsize=50`，按 validation loss 保存：
  - model: `model_qc_pcorr_pos_top10_small_lr001_loss_300e_s50_20260603`
  - log: `log_qc_pcorr_pos_top10_small_lr001_loss_300e_s50_20260603`
  - final test: accuracy `0.5833`, balanced accuracy `0.5635`, loss `0.9902`。
  - 观察：比 100 epoch 小模型明显改善，但 150 epoch 后训练 balanced accuracy 继续升到 `0.9+`，validation loss 开始上升，说明仍会过拟合。

### 观察

- `positive pcorr top10% + lr=0.001` 的 validation 表现比上一轮 `corr top20 + lr=0.01` 更稳定，训练过拟合速度也更慢。
- 按 validation balanced accuracy 保存会明显偏向验证集，测试集表现反而较差。
- 按 validation loss 保存更稳，但 test balanced accuracy 仍只有 `0.5272`，没有接近 QC 线性基线。
- 过小模型加高 dropout 在 100 epoch 内欠拟合；延长到 300 epoch 并放慢学习率衰减后能学到更多信号，但后期仍过拟合。
- 当前 fold 0 上，BrainGNN 仍明显弱于 QC 线性基线的 `corr` 特征结果：balanced accuracy `0.6485`, AUC `0.7240`。

## 2026-06-03 - QC + corr top-k 实验记录

### 数据处理

- 新增独立数据目录：`data/ABIDE_pcp/cpac/filt_noglobal_qc_fd020_perc20`。
- 过滤条件：
  - `qc_rater_1 == "OK"`
  - `qc_func_rater_2 == "OK"` 或 `qc_func_rater_3 == "OK"`
  - `func_mean_fd < 0.2`
  - `func_perc_fd < 20`
- 生成结果：777 个样本，标签分布为 0 类 433 个、1 类 344 个，保留 20 个站点。
- 新增脚本：`04-build_qc_dataset.py`。

### 线性基线

- 新增脚本：`05-linear_baseline.py`。
- 结果保存到：`data/ABIDE_pcp/cpac/filt_noglobal_qc_fd020_perc20/linear_baseline_results.json`。
- 5 折逻辑回归结果：
  - `corr`: accuracy `0.6499 +/- 0.0384`, balanced accuracy `0.6485 +/- 0.0350`, AUC `0.7240 +/- 0.0304`
  - `pcorr`: accuracy `0.6332 +/- 0.0267`, balanced accuracy `0.6329 +/- 0.0285`, AUC `0.6685 +/- 0.0283`
  - `abs_pcorr`: accuracy `0.5727 +/- 0.0373`, balanced accuracy `0.5709 +/- 0.0390`, AUC `0.6270 +/- 0.0416`

### BrainGNN

- 扩展 `ABIDEDataset` 和 `read_abide_stats_parall.py`，支持 `edge_source` 与 `edge_topk`。
- 训练配置：
  - `--dataroot data/ABIDE_pcp/cpac/filt_noglobal_qc_fd020_perc20`
  - `--processed_file data_corr_top20.pt`
  - `--edge_source corr`
  - `--edge_topk 20`
- PyG cache：`data/ABIDE_pcp/cpac/filt_noglobal_qc_fd020_perc20/processed/data_corr_top20.pt`。
- 稀疏图规模：每图边数 min/mean/max 为 `2250 / 5303.25 / 5954`。
- 训练输出目录：
  - model: `model_qc_corr_top20_gpu_20260603`
  - log: `log_qc_corr_top20_gpu_20260603`
- 最终测试结果：`Test Acc = 0.5705128`, `Test Loss = 0.9681596`。
- 观察：训练准确率后期达到 `1.0`，但验证准确率约在 `0.55-0.62` 波动，仍有明显过拟合。

## 2026-06-03 - 本次源码修复记录

### 背景

针对 ABIDE 数据集上 BrainGNN 训练误差低但测试误差高的问题，检查了数据下载、预处理、PyG 数据缓存、数据切分、模型 forward、pooling 正则项和轻量传统基线。

### 诊断结论

- ABIDE 示例数据本身噪声较高，且原仓库 README 明确说明该 ABIDE 示例不同于论文正式实验使用的数据。
- 本地标签映射未发现错误：1035 个样本标签分布为 0 类 530 个、1 类 505 个。
- PyG 缓存中的 `data.x` 存在非有限值，主要来自相关矩阵对角线 `arctanh(1)=inf`。
- 原切分函数写死 `n_sub = 1035`，且使用普通 `KFold`，没有按类别分层。
- 高版本 PyG 的 `TopKPooling` 已经对 score 应用传入的 `torch.sigmoid`，原模型返回时又额外 `sigmoid`，训练中的 consistency loss 又再次 `sigmoid`，会压缩 pooling score 的动态范围。
- 同一批 ABIDE 数据的轻量线性基线显示：`corr` 上三角逻辑回归约 `0.67 acc / 0.75 AUC`，`abs(pcorr)` 约 `0.59 acc`，说明数据并非完全无泛化信号，但当前图构造和模型容量容易放大过拟合。

### 修改内容

- `03-main.py`
  - 将训练入口的数据清理从只替换 `+inf` 改为 `torch.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)`。
  - 调用 `train_val_test_split(..., labels=...)`，使用标签分层切分。
  - 移除 `consist_loss` 内部对 pooling score 的重复 `torch.sigmoid`。

- `imports/utils.py`
  - `train_val_test_split` 增加 `labels` 和 `n_sub` 参数。
  - 传入 `labels` 时使用 `StratifiedKFold`；未传入时保留普通 `KFold` 的兼容行为。
  - 不再在有标签输入时写死样本数。

- `net/braingnn.py`
  - 模型返回 `TopKPooling` 的 score 原值，不再额外应用 `torch.sigmoid`。

- `imports/read_abide_stats_parall.py`
  - 在图构造阶段对 `pcorr` 和 `corr` 做 `np.nan_to_num`，避免重新生成 PyG cache 时继续写入非有限值。

### 验证

- 运行 `python -m py_compile 03-main.py imports/utils.py imports/read_abide_stats_parall.py net/braingnn.py`，通过。
- 小 batch 前向传播通过。
- 修复后的 fold 0 标签分布：
  - train: 303 / 318
  - val: 101 / 106
  - test: 101 / 106
- 训练入口清理后 `data.x` 检查为 finite。

### 未完成项

- 未执行完整 100 epoch 或 5 折重训。
- 后续建议用修复后的代码重跑 5 折，并与线性基线的 balanced accuracy 和 AUC 对照。
