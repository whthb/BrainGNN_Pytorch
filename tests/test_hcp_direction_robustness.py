import argparse
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from net.braingnn import GradientReversal


def load_module(filename, name):
    path = Path(__file__).parents[1] / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graph(direction, subject="subject", task="TASK"):
    data = Data(y=torch.tensor(0), direction=torch.tensor(direction))
    data.subject = subject
    data.sample_name = f"{task}_{'LR' if direction == 0 else 'RL'}"
    return data


def test_direction_protocol_filters_source_and_target_runs():
    trainer = load_module("07-main_hcp_subjectwise.py", "hcp_direction_trainer")
    train = [graph(0), graph(1)]
    val = [graph(0), graph(1)]
    test = [graph(0), graph(1)]

    lr_train, lr_val, rl_test = trainer.apply_direction_protocol(
        train, val, test, "lr_to_rl"
    )
    assert [int(item.direction) for item in lr_train] == [0]
    assert [int(item.direction) for item in lr_val] == [0]
    assert [int(item.direction) for item in rl_test] == [1]


def test_gradient_reversal_reverses_and_scales_gradients():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    GradientReversal.apply(x, 0.25).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([-0.25, -0.25]))


def test_complete_direction_pairs_only_returns_matched_subject_tasks():
    trainer = load_module("07-main_hcp_subjectwise.py", "hcp_pair_trainer")
    graphs = [
        graph(0, "001", "WM"),
        graph(1, "001", "WM"),
        graph(0, "001", "MOTOR"),
        graph(1, "002", "WM"),
    ]

    pairs = trainer.complete_direction_pairs(graphs)

    assert len(pairs) == 1
    assert pairs[0][0].sample_name == "WM_LR"
    assert pairs[0][1].sample_name == "WM_RL"


def test_pair_js_divergence_is_zero_for_equal_predictions_and_symmetric():
    trainer = load_module("07-main_hcp_subjectwise.py", "hcp_pair_js")
    lr = torch.log_softmax(torch.tensor([[2.0, 0.0], [0.0, 1.0]]), dim=1)
    rl = torch.log_softmax(torch.tensor([[0.0, 2.0], [1.0, 0.0]]), dim=1)

    torch.testing.assert_close(
        trainer.pair_js_divergence(lr, lr), torch.tensor(0.0), atol=1e-7, rtol=0
    )
    torch.testing.assert_close(
        trainer.pair_js_divergence(lr, rl),
        trainer.pair_js_divergence(rl, lr),
    )
    assert float(trainer.pair_js_divergence(lr, rl)) > 0


def test_pair_task_weights_balance_tasks_and_have_unit_pair_mean():
    trainer = load_module("07-main_hcp_subjectwise.py", "hcp_pair_weights")
    pairs = [
        (graph(0, "001", "COMMON"), graph(1, "001", "COMMON")),
        (graph(0, "002", "COMMON"), graph(1, "002", "COMMON")),
        (graph(0, "003", "COMMON"), graph(1, "003", "COMMON")),
        (graph(0, "004", "RARE"), graph(1, "004", "RARE")),
    ]
    pairs[-1][0].y = torch.tensor(1)
    pairs[-1][1].y = torch.tensor(1)

    weights = trainer.pair_task_weights(pairs, "sqrt_inverse")

    assert weights[1] > weights[0]
    pair_mean = sum(weights[int(lr.y)] for lr, _ in pairs) / len(pairs)
    assert abs(pair_mean - 1) < 1e-7


def test_pair_weight_warmup_schedule():
    trainer = load_module("07-main_hcp_subjectwise.py", "hcp_pair_warmup")

    assert trainer.pair_weight_at_epoch(0.2, 9, 10, 20) == 0
    assert trainer.pair_weight_at_epoch(0.2, 10, 10, 20) == pytest.approx(0.01)
    assert trainer.pair_weight_at_epoch(0.2, 29, 10, 20) == 0.2
    assert trainer.pair_weight_at_epoch(0.2, 50, 10, 20) == 0.2


def test_partition_direction_pairs_uses_each_pair_once():
    trainer = load_module("07-main_hcp_subjectwise.py", "hcp_pair_partition")
    pairs = [(graph(0, str(index)), graph(1, str(index))) for index in range(7)]

    batches = trainer.partition_direction_pairs(pairs, n_batches=3, shuffle=False)

    assert [len(batch) for batch in batches] == [3, 2, 2]
    assert [pair for batch in batches for pair in batch] == [
        pairs[0], pairs[3], pairs[6], pairs[1], pairs[4], pairs[2], pairs[5]
    ]


def test_pair_consistency_loss_preserves_train_mode_and_gradients():
    trainer = load_module("07-main_hcp_subjectwise.py", "hcp_pair_loss")

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x, edge_index, batch, edge_attr, pos):
            assert not self.training
            pooled = torch.zeros(int(batch.max()) + 1, device=x.device)
            pooled.index_add_(0, batch, x.view(-1))
            output = F.log_softmax(
                torch.stack([self.scale * pooled, -self.scale * pooled], dim=1),
                dim=1,
            )
            return output, None, None, None, None

    def pair_graph(value, direction):
        data = graph(direction)
        data.x = torch.tensor([[value]], dtype=torch.float32)
        data.edge_index = torch.empty((2, 0), dtype=torch.long)
        data.edge_attr = torch.empty((0, 1), dtype=torch.float32)
        data.pos = torch.eye(1)
        return data

    model = DummyModel()
    model.train()
    loss = trainer.pair_consistency_loss(
        model, [(pair_graph(1.0, 0), pair_graph(-1.0, 1))], torch.device("cpu")
    )
    loss.backward()

    assert model.training
    assert model.scale.grad is not None
    assert float(model.scale.grad.abs()) > 0


def test_paired_metrics_include_task_macro_js_ensemble_and_bootstrap():
    trainer = load_module("07-main_hcp_subjectwise.py", "hcp_pair_metrics")
    log_probs = torch.log_softmax(
        torch.tensor([[4.0, 0.0], [0.0, 4.0], [3.0, 0.0], [2.0, 0.0]]),
        dim=1,
    ).numpy()
    outputs = {
        "subjects": ["001", "001", "002", "002"],
        "sample_names": ["A_LR", "A_RL", "B_LR", "B_RL"],
        "pred": log_probs.argmax(axis=1),
        "true": torch.tensor([0, 0, 0, 0]).numpy(),
        "log_probs": log_probs,
        "scores": None,
    }

    metrics = trainer.paired_direction_metrics(outputs, topk=1, seed=123, bootstrap_samples=20)

    assert metrics["n_pairs"] == 2
    assert set(metrics["per_task"]) == {"A", "B"}
    assert metrics["macro_task_prediction_agreement"] == 0.5
    assert metrics["mean_prediction_js"] > 0
    assert metrics["ensemble_metrics"]["accuracy"] == 1
    assert metrics["subject_bootstrap_95ci"]["prediction_agreement"]["n_bootstrap"] == 20


def test_direction_stratified_splits_are_subject_disjoint_and_complete():
    runner = load_module("17-run_hcp_direction_robustness.py", "hcp_direction_runner")
    records = []
    for subject_index in range(40):
        for task in range(7):
            for run in ("LR", "RL"):
                records.append(
                    {
                        "subject": f"{subject_index:03d}",
                        "task_label": str(task),
                        "run": run,
                    }
                )
    folds = runner.build_direction_stratified_splits(records, seed=123)
    assert len(folds) == 5
    for fold in folds:
        train = set(fold["train_subjects"])
        val = set(fold["val_subjects"])
        test = set(fold["test_subjects"])
        assert not train & val
        assert not train & test
        assert not val & test
        assert len(train | val | test) == 40


def test_direction_runner_uses_strong_baseline_loss(tmp_path):
    runner = load_module("17-run_hcp_direction_robustness.py", "hcp_direction_runner_command")
    dataroot = tmp_path / "data"
    dataroot.mkdir()
    (dataroot / "sample_metadata.csv").write_text(
        "subject,task_label,run\n001,0,LR\n"
    )
    args = argparse.Namespace(
        dataroot=dataroot,
        output_dir=tmp_path / "output",
        n_epochs=3,
        batch_size=4,
        save_model=False,
        pair_balance="sqrt_inverse",
        pair_warmup_start=10,
        pair_warmup_epochs=20,
        max_validation_bacc_drop=0.005,
        early_stopping_patience=20,
        early_stopping_min_epochs=40,
        bootstrap_samples=100,
    )
    split_dir = args.output_dir / "splits"
    split_dir.mkdir(parents=True)
    (split_dir / "seed_123.json").write_text("{}")
    command = runner.fold_command(args, "mixed", 0.1, 123, 0, tmp_path / "fold")
    assert command[command.index("--lamb3") + 1] == "0"
    assert command[command.index("--lamb4") + 1] == "0"
    assert command[command.index("--lamb5") + 1] == "0.1"
    assert command[command.index("--direction_adv_weight") + 1] == "0.1"


def test_pair_runner_uses_fixed_split_and_pair_consistency_loss(tmp_path):
    runner = load_module("18-run_hcp_pair_consistency.py", "hcp_pair_runner_command")
    dataroot = tmp_path / "data"
    dataroot.mkdir()
    (dataroot / "sample_metadata.csv").write_text(
        "subject,task_label,run\n001,0,LR\n"
    )
    args = argparse.Namespace(
        dataroot=dataroot,
        output_dir=tmp_path / "output",
        split_seed=123,
        n_epochs=3,
        batch_size=4,
        save_model=False,
        pair_balance="sqrt_inverse",
        pair_warmup_start=10,
        pair_warmup_epochs=20,
        max_validation_bacc_drop=0.005,
        early_stopping_patience=20,
        early_stopping_min_epochs=40,
        bootstrap_samples=100,
    )
    split_dir = args.output_dir / "splits"
    split_dir.mkdir(parents=True)
    manifest = split_dir / "split_seed_123.json"
    manifest.write_text("{}")

    command = runner.fold_command(args, 0.05, 456, 0, tmp_path / "fold")

    assert command[command.index("--split_manifest") + 1] == str(manifest)
    assert command[command.index("--seed") + 1] == "456"
    assert command[command.index("--pair_consistency_weight") + 1] == "0.05"
    assert command[command.index("--direction_adv_weight") + 1] == "0"
    assert command[command.index("--pair_balance") + 1] == "sqrt_inverse"
    assert command[command.index("--checkpoint_selection") + 1] == "pair_noninferiority"
    assert command[command.index("--pair_warmup_start") + 1] == "10"
