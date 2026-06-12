#!/usr/bin/env python
"""Generate a Markdown report from the current HCP subject-wise experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel


TASK_ORDER = ("EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def aggregate(root: Path, name: str) -> dict[str, object]:
    return load_json(root / "aggregates" / f"{name}.json")


def config_result(result: dict[str, object], name: str) -> dict[str, object]:
    return result["configs"][name]


def metric(result: dict[str, object], name: str = "balanced_accuracy") -> tuple[float, float]:
    values = result["metrics"][name]
    return float(values["mean"]), float(values["std"])


def fmt(result: dict[str, object], name: str = "balanced_accuracy") -> str:
    mean, std = metric(result, name)
    return f"{mean:.4f} +/- {std:.4f}"


def paired_comparison(first: dict[str, object], second: dict[str, object],
                      metric_name: str = "balanced_accuracy") -> dict[str, float]:
    first_by_fold = {int(row["fold"]): float(row[metric_name]) for row in first["folds"]}
    second_by_fold = {int(row["fold"]): float(row[metric_name]) for row in second["folds"]}
    folds = sorted(set(first_by_fold) & set(second_by_fold))
    first_values = np.asarray([first_by_fold[fold] for fold in folds])
    second_values = np.asarray([second_by_fold[fold] for fold in folds])
    return {
        "mean_difference": float(np.mean(first_values - second_values)),
        "p_value": float(ttest_rel(first_values, second_values).pvalue),
    }


def parameter_count(root: Path, source_run: str) -> int | None:
    summaries = sorted((root / "runs" / source_run).glob("fold_*/summary.json"))
    if not summaries:
        return None
    return load_json(summaries[0]).get("model_parameter_count")


def per_task_recall(root: Path) -> dict[str, dict[str, float]]:
    rows = []
    for path in sorted((root / "runs" / "paper_like").glob("fold_*/summary.json")):
        report = load_json(path)["classification_report"]
        rows.append([float(report[str(label)]["recall"]) for label in range(len(TASK_ORDER))])
    values = np.asarray(rows)
    return {
        task: {"mean": float(values[:, label].mean()), "std": float(values[:, label].std())}
        for label, task in enumerate(TASK_ORDER)
    }


def interpretation_rows(root: Path) -> list[tuple[str, str, str, str]]:
    mappings = [
        ("0", "glc_0"),
        ("0.1", "glc_0.1"),
        ("0.5", "glc_0.5"),
    ]
    rows = []
    for glc, name in mappings:
        path = root / "interpretability_summaries" / f"{name}.json"
        if not path.exists():
            continue
        values = load_json(path)
        task_values = [
            score
            for fold in values["within_task_selected_roi_jaccard_by_fold"]
            for score in fold["task_jaccard"].values()
        ]
        community = values["community_aligned_cosine_similarity"]["mean"]
        rows.append((
            glc,
            f"{float(np.mean(task_values)):.4f}",
            f"{float(community):.4f}" if community is not None else "n/a",
            f"{float(values['pool_score_top_bottom_mean_gap']['mean']):.4f}",
        ))
    return rows


def baseline_rows(root: Path) -> list[tuple[str, str, str]]:
    rows = []
    for filename, model_name, label in [
        ("majority.json", "majority", "Majority class"),
        ("same_input_rbf_svm.json", "rbf_svm", "Same-input RBF-SVM"),
    ]:
        path = root / "baselines" / filename
        if not path.exists():
            continue
        values = load_json(path)["aggregate"][model_name]
        rows.append((
            label,
            f"{float(values['balanced_accuracy_mean']):.4f} +/- "
            f"{float(values['balanced_accuracy_std']):.4f}",
            f"{float(values['macro_f1_mean']):.4f} +/- {float(values['macro_f1_std']):.4f}",
        ))
    return rows


def main() -> None:
    args = parse_args()
    root = args.experiment_root
    output = args.output or root / "REPORT.md"
    protocol = load_json(root / "protocol.json")
    main_result = config_result(aggregate(root, "main"), "paper_like")
    loss = aggregate(root, "loss_ablation")
    conv = aggregate(root, "conv_ablation")
    lambdas = aggregate(root, "lambda_sweep")
    capacity = aggregate(root, "capacity")

    ra = config_result(conv, "ra")
    vanilla = config_result(conv, "vanilla")
    conv_comparison = paired_comparison(ra, vanilla)
    local_capacity = config_result(capacity, "paper_like_55k")
    paper_capacity = config_result(capacity, "paper_96k")
    capacity_comparison = paired_comparison(paper_capacity, local_capacity)
    lambda_rows = sorted(
        lambdas["configs"].items(),
        key=lambda item: (
            float(item[1]["parameters"]["tpk"]),
            float(item[1]["parameters"]["glc"]),
        ),
    )
    best_lambda_name, best_lambda = max(
        lambda_rows, key=lambda item: metric(item[1], "balanced_accuracy")[0]
    )
    rbf_path = root / "baselines" / "same_input_rbf_svm.json"
    rbf_comparison = (
        paired_comparison(main_result, load_json(rbf_path))
        if rbf_path.exists() else None
    )
    task_recall = per_task_recall(root)
    completed_runs = {
        path.parent.parent.name
        for path in (root / "runs").glob("*/fold_*/summary.json")
    }

    lines = [
        "# Current-Model HCP Subject-Wise BrainGNN Reproduction Report",
        "",
        "## Scope",
        "",
        "This report reruns the paper-motivated HCP experiments with the current "
        "`07-main_hcp_subjectwise.py` model and the unified five-fold "
        "`15-run_hcp_subjectwise_new_baseline.py` entry point.",
        "",
        "This is a local-data method reproduction, not an exact numerical reproduction "
        "of the paper's HCP result. The local dataset contains 343 partially observed "
        "subjects, 1,235 LR/RL run-level graphs, seven tasks, and 68 cortical ROIs; the "
        "paper used a different complete-subject cohort and 268 ROIs.",
        "",
        "## Protocol",
        "",
        f"- Completed unique configurations: {len(completed_runs)}",
        f"- Folds: `{protocol['folds']}`; epochs: `{protocol['n_epochs']}`; "
        f"batch size: `{protocol['batch_size']}`",
        "- Subject-wise deterministic 60/20/20-style train/validation/test split",
        "- Pearson-correlation rows as node features",
        "- Positive top-10% partial-correlation edges",
        "- Adam, learning rate 0.001, weight decay 0.005, step decay every 20 epochs",
        "- Checkpoint selected by validation balanced accuracy",
        "- CUDA deterministic algorithms are not forced; repeated training can vary despite seed 123",
        "",
        "## Main Five-Fold Result",
        "",
        "| Metric | Mean +/- std |",
        "|---|---:|",
        f"| Accuracy | {fmt(main_result, 'accuracy')} |",
        f"| Balanced accuracy | {fmt(main_result)} |",
        f"| Macro F1 | {fmt(main_result, 'macro_f1')} |",
        f"| Macro precision | {fmt(main_result, 'macro_precision')} |",
        "",
        "| Fold | Accuracy | Balanced accuracy | Macro F1 |",
        "|---:|---:|---:|---:|",
    ]
    for row in main_result["folds"]:
        lines.append(
            f"| {row['fold']} | {row['accuracy']:.4f} | "
            f"{row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} |"
        )

    lines.extend([
        "",
        "### Classification Baselines",
        "",
        "| Method | Balanced accuracy | Macro F1 |",
        "|---|---:|---:|",
    ])
    for row in baseline_rows(root):
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} |")
    lines.extend([
        f"| BrainGNN current paper-like | {fmt(main_result)} | {fmt(main_result, 'macro_f1')} |",
    ])
    if rbf_comparison is not None:
        lines.extend([
            "",
            f"BrainGNN changes mean balanced accuracy by "
            f"`{rbf_comparison['mean_difference']:+.4f}` versus the same-input RBF-SVM "
            f"(paired five-fold t-test `p={rbf_comparison['p_value']:.5f}`).",
        ])
    lines.extend([
        "",
        "### Per-Task Recall",
        "",
        "| Task | Recall, mean +/- std |",
        "|---|---:|",
    ])
    for task in TASK_ORDER:
        values = task_recall[task]
        lines.append(f"| {task} | {values['mean']:.4f} +/- {values['std']:.4f} |")

    lines.extend([
        "",
        "## ROI-Aware Convolution Ablation",
        "",
        "| Convolution | Balanced accuracy |",
        "|---|---:|",
        f"| Ra-GConv | {fmt(ra)} |",
        f"| Shared-kernel vanilla-GConv | {fmt(vanilla)} |",
        "",
        f"Ra-GConv changes mean balanced accuracy by "
        f"`{conv_comparison['mean_difference']:+.4f}` versus vanilla-GConv "
        f"(paired five-fold t-test `p={conv_comparison['p_value']:.5f}`).",
        "",
        "## Loss Ablation",
        "",
        "| Loss | Balanced accuracy | Macro F1 |",
        "|---|---:|---:|",
    ])
    loss_labels = {
        "ce_only": "CE only",
        "ce_unit": "CE + unit",
        "ce_unit_tpk": "CE + unit + TPK",
        "ce_unit_glc": "CE + unit + GLC",
        "full": "Full loss",
    }
    for name, label in loss_labels.items():
        result = config_result(loss, name)
        lines.append(f"| {label} | {fmt(result)} | {fmt(result, 'macro_f1')} |")

    lines.extend([
        "",
        "## Lambda Sweep",
        "",
        "| lambda1 TPK | lambda2 GLC | Balanced accuracy |",
        "|---:|---:|---:|",
    ])
    for _, result in lambda_rows:
        params = result["parameters"]
        lines.append(f"| {params['tpk']:g} | {params['glc']:g} | {fmt(result)} |")
    lines.extend([
        "",
        f"The best tested setting is `{best_lambda_name}` with balanced accuracy "
        f"`{fmt(best_lambda)}`.",
        "",
        "## Parameter Capacity",
        "",
        "| Setting | Trainable parameters | Balanced accuracy |",
        "|---|---:|---:|",
        f"| Current default head | {parameter_count(root, local_capacity['source_run']):,} | "
        f"{fmt(local_capacity)} |",
        f"| Approximately 96k paper-capacity head | "
        f"{parameter_count(root, paper_capacity['source_run']):,} | {fmt(paper_capacity)} |",
        "",
        f"The approximately 96k setting changes mean balanced accuracy by "
        f"`{capacity_comparison['mean_difference']:+.4f}` "
        f"(paired five-fold t-test `p={capacity_comparison['p_value']:.5f}`).",
        "",
        "## Interpretability",
        "",
        "| GLC weight | Mean within-task top-17 ROI Jaccard | Community stability | "
        "Top-bottom score gap |",
        "|---:|---:|---:|---:|",
    ])
    for row in interpretation_rows(root):
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    lines.extend([
        "",
        "Section 3.5-style outputs are under `interpretability_figures/`:",
        "",
        "- `figure5_glc_individual_group`: individual/group ROI consistency across GLC weights",
        "- `figure7_task_salient_rois`: task-level mean first-pooling saliency",
        "- `figure8_proxy_task_roi_similarity`: top-ROI Jaccard proxy, not Neurosynth decoding",
        "- `figure9_community_assignments`: strongest qualifying Ra-GConv community per ROI",
        "- `figure10_alpha_positive_heatmap`: first-layer non-negative community weights",
        "",
        "The current artifacts save first-pooling scores but not second-pooling ROI mappings. "
        "The Fig. 5- and Fig. 7-style maps therefore use first-pooling top-17 scores as a "
        "documented proxy. Cortical maps are anatomical schematics, not surface renderings.",
        "",
        "## Conclusion",
        "",
        f"The current-model main experiment reaches balanced accuracy `{fmt(main_result)}` "
        "on the local run-level HCP subset. The ablations, lambda sweep, capacity comparison, "
        "and interpretation figures above must be interpreted within the documented local-data "
        "deviations and are not directly comparable to the paper's reported HCP accuracy.",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    report_data = {
        "main": main_result,
        "conv_comparison": conv_comparison,
        "capacity_comparison": capacity_comparison,
        "rbf_comparison": rbf_comparison,
        "best_lambda": {"name": best_lambda_name, **best_lambda},
        "per_task_recall": task_recall,
        "baselines": baseline_rows(root),
    }
    (root / "report_data.json").write_text(json.dumps(report_data, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
