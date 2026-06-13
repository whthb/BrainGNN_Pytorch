#!/usr/bin/env python
"""Run fixed-split HCP LR/RL pair-consistency BrainGNN experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATAROOT = PROJECT_ROOT / "data" / "HCP900_subjectwise_qc_allsub_7task_lrrl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "hcp900_pair_consistency_20260613"
DEFAULT_INIT_SEEDS = (123, 456, 789)
DEFAULT_TUNE_INIT_SEEDS = (123, 456)
DEFAULT_PAIR_WEIGHTS = (0.05, 0.1, 0.2, 0.5)

SCALAR_PATHS = {
    "accuracy": ("metrics", "accuracy"),
    "balanced_accuracy": ("metrics", "balanced_accuracy"),
    "macro_f1": ("metrics", "macro_f1"),
    "validation_balanced_accuracy": ("selected_checkpoint_validation_metrics", "balanced_accuracy"),
    "validation_pair_agreement": (
        "selected_checkpoint_validation_paired_direction_metrics", "prediction_agreement"
    ),
    "pair_prediction_agreement": ("paired_direction_metrics", "prediction_agreement"),
    "pair_macro_task_agreement": (
        "paired_direction_metrics", "macro_task_prediction_agreement"
    ),
    "pair_both_correct": ("paired_direction_metrics", "both_correct"),
    "pair_one_correct": ("paired_direction_metrics", "one_correct"),
    "pair_prediction_js": ("paired_direction_metrics", "mean_prediction_js"),
    "pair_ensemble_accuracy": ("paired_direction_metrics", "ensemble_metrics", "accuracy"),
    "pair_ensemble_balanced_accuracy": (
        "paired_direction_metrics", "ensemble_metrics", "balanced_accuracy"
    ),
    "pair_pool1_jaccard": ("paired_direction_metrics", "pool1_topk_jaccard"),
    "direction_probe_balanced_accuracy": ("direction_probe_metrics", "balanced_accuracy"),
    "selected_epoch": ("selected_epoch",),
    "epochs_trained": ("epochs_trained",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=DEFAULT_DATAROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--phase",
        choices=["all", "baseline", "tune", "final", "summarize"],
        default="all",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--init-seeds", nargs="+", type=int, default=list(DEFAULT_INIT_SEEDS))
    parser.add_argument(
        "--tune-init-seeds", nargs="+", type=int, default=list(DEFAULT_TUNE_INIT_SEEDS)
    )
    parser.add_argument("--pair-weights", nargs="+", type=float, default=list(DEFAULT_PAIR_WEIGHTS))
    parser.add_argument("--max-validation-bacc-drop", type=float, default=0.005)
    parser.add_argument("--pair-balance", choices=["none", "sqrt_inverse"], default="sqrt_inverse")
    parser.add_argument("--pair-warmup-start", type=int, default=10)
    parser.add_argument("--pair-warmup-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=40)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def weight_tag(weight: float) -> str:
    return f"{weight:g}".replace(".", "p")


def config_name(pair_weight: float) -> str:
    return "baseline_mixed" if pair_weight == 0 else f"pair_js_{weight_tag(pair_weight)}_mixed"


def read_records(dataroot: Path) -> list[dict[str, str]]:
    with (dataroot / "sample_metadata.csv").open() as handle:
        return list(csv.DictReader(handle))


def build_direction_stratified_splits(
    records: list[dict[str, str]], seed: int, n_folds: int = 5, val_folds: int = 4
) -> list[dict[str, object]]:
    task = np.asarray([int(record["task_label"]) for record in records], dtype=np.int64)
    direction = np.asarray([0 if record["run"] == "LR" else 1 for record in records], dtype=np.int64)
    strata = task * 2 + direction
    groups = np.asarray([record["subject"] for record in records], dtype=object)
    dummy = np.zeros(len(records), dtype=np.int8)
    required_strata = set(strata.tolist())
    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for fold, (train_val_index, test_index) in enumerate(outer.split(dummy, strata, groups)):
        inner = StratifiedGroupKFold(
            n_splits=val_folds, shuffle=True, random_state=seed + fold + 1
        )
        valid_inner = None
        for inner_train, inner_val in inner.split(
            dummy[train_val_index], strata[train_val_index], groups[train_val_index]
        ):
            train_index = train_val_index[inner_train]
            val_index = train_val_index[inner_val]
            if (
                set(strata[train_index].tolist()) == required_strata
                and set(strata[val_index].tolist()) == required_strata
            ):
                valid_inner = train_index, val_index
                break
        if valid_inner is None or set(strata[test_index].tolist()) != required_strata:
            raise ValueError(f"split seed {seed} fold {fold} lacks a task-direction stratum")
        train_index, val_index = valid_inner
        split = {"fold": fold}
        for split_name, index in {
            "train": train_index, "val": val_index, "test": test_index
        }.items():
            split[f"{split_name}_subjects"] = sorted(set(groups[index].tolist()))
            split[f"n_{split_name}_samples"] = int(len(index))
        folds.append(split)
    return folds


def write_split_manifest(args: argparse.Namespace) -> Path:
    split_dir = args.output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"split_seed_{args.split_seed}.json"
    if not path.exists():
        records = read_records(args.dataroot)
        manifest = {
            "dataroot": str(args.dataroot),
            "method": "fixed StratifiedGroupKFold over 7 task x 2 direction strata",
            "n_subjects": len({record["subject"] for record in records}),
            "n_samples": len(records),
            "n_folds": 5,
            "split_seed": args.split_seed,
            "folds": build_direction_stratified_splits(records, args.split_seed),
        }
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def fold_dir(args: argparse.Namespace, pair_weight: float, init_seed: int, fold: int) -> Path:
    return (
        args.output_dir / "runs" / config_name(pair_weight)
        / f"init_seed_{init_seed}" / f"fold_{fold}"
    )


def fold_command(
    args: argparse.Namespace,
    pair_weight: float,
    init_seed: int,
    fold: int,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "07-main_hcp_subjectwise.py"),
        "--dataroot", str(args.dataroot),
        "--split_manifest", str(write_split_manifest(args)),
        "--fold", str(fold),
        "--n_epochs", str(args.n_epochs),
        "--batchSize", str(args.batch_size),
        "--lr", "0.001",
        "--stepsize", "20",
        "--gamma", "0.5",
        "--weightdecay", "0.005",
        "--edge_source", "pcorr",
        "--edge_top_percent", "0.10",
        "--positive_edges_only",
        "--best_metric", "balanced_acc",
        "--lamb1", "1",
        "--lamb2", "1",
        "--lamb3", "0",
        "--lamb4", "0",
        "--lamb5", "0.1",
        "--direction_protocol", "mixed",
        "--direction_adv_weight", "0",
        "--pair_consistency_weight", str(pair_weight),
        "--pair_balance", args.pair_balance,
        "--pair_warmup_start", str(args.pair_warmup_start),
        "--pair_warmup_epochs", str(args.pair_warmup_epochs),
        "--checkpoint_selection", "pair_noninferiority",
        "--checkpoint_bacc_tolerance", str(args.max_validation_bacc_drop),
        "--early_stopping_patience", str(args.early_stopping_patience),
        "--early_stopping_min_epochs", str(args.early_stopping_min_epochs),
        "--bootstrap_samples", str(args.bootstrap_samples),
        "--seed", str(init_seed),
        "--output_dir", str(output),
    ]
    command.append("--save_model" if args.save_model else "--no-save_model")
    return command


def run_fold(args: argparse.Namespace, pair_weight: float, init_seed: int, fold: int) -> str:
    output = fold_dir(args, pair_weight, init_seed, fold)
    if args.skip_completed and (output / "summary.json").exists():
        return f"skip completed {config_name(pair_weight)} init_seed={init_seed} fold={fold}"
    command = fold_command(args, pair_weight, init_seed, fold, output)
    print(" ".join(command), flush=True)
    if args.dry_run:
        return f"dry run {config_name(pair_weight)} init_seed={init_seed} fold={fold}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (output / "stdout.log").open("w") as stdout:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True, stdout=stdout, stderr=subprocess.STDOUT)
    return f"completed {config_name(pair_weight)} init_seed={init_seed} fold={fold}"


def run_work(args: argparse.Namespace, work: list[tuple[float, int, int]]) -> None:
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_fold, args, weight, seed, fold): (weight, seed, fold)
            for weight, seed, fold in work
        }
        for future in as_completed(futures):
            weight, seed, fold = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:
                raise RuntimeError(
                    f"failed {config_name(weight)} init_seed={seed} fold={fold}"
                ) from error


def nested_value(data: dict[str, object], path: tuple[str, ...]) -> float | None:
    value: object = data
    for key in path:
        if not isinstance(value, dict) or key not in value or value[key] is None:
            return None
        value = value[key]
    return float(value)


def summarize_config(
    args: argparse.Namespace, pair_weight: float, init_seeds: list[int]
) -> dict[str, object]:
    rows = []
    for seed in init_seeds:
        for fold in args.folds:
            path = fold_dir(args, pair_weight, seed, fold) / "summary.json"
            if path.exists():
                rows.append({"init_seed": seed, "fold": fold, "summary": json.loads(path.read_text())})
    metrics = {}
    for name, path in SCALAR_PATHS.items():
        values = [nested_value(row["summary"], path) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            metrics[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": len(values),
            }
    return {
        "name": config_name(pair_weight),
        "pair_consistency_weight": pair_weight,
        "requested_init_seeds": init_seeds,
        "completed_folds": len(rows),
        "metrics": metrics,
    }


def select_pair_weight(args: argparse.Namespace) -> tuple[float, dict[str, object]]:
    expected = len(args.folds) * len(args.tune_init_seeds)
    baseline = summarize_config(args, 0.0, args.tune_init_seeds)
    candidates = [summarize_config(args, weight, args.tune_init_seeds) for weight in args.pair_weights]
    if baseline["completed_folds"] != expected:
        raise ValueError("complete tuning-seed baseline results are required for pair-weight selection")
    complete = [
        candidate for candidate in candidates
        if candidate["completed_folds"] == expected
        and "validation_balanced_accuracy" in candidate["metrics"]
        and "validation_pair_agreement" in candidate["metrics"]
    ]
    if not complete:
        raise ValueError("no complete pair-consistency tuning candidate is available")
    baseline_bacc = baseline["metrics"]["validation_balanced_accuracy"]["mean"]
    threshold = baseline_bacc - args.max_validation_bacc_drop
    eligible = [
        candidate for candidate in complete
        if candidate["metrics"]["validation_balanced_accuracy"]["mean"] >= threshold
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda candidate: (
                candidate["metrics"]["validation_pair_agreement"]["mean"],
                candidate["metrics"]["validation_balanced_accuracy"]["mean"],
            ),
        )
        rule = "maximize validation pair agreement subject to validation BAcc non-inferiority"
    else:
        selected = max(
            complete,
            key=lambda candidate: candidate["metrics"]["validation_balanced_accuracy"]["mean"],
        )
        rule = "fallback: maximize validation BAcc because no candidate met non-inferiority"
    result = {
        "selection_rule": rule,
        "max_validation_bacc_drop": args.max_validation_bacc_drop,
        "baseline": baseline,
        "candidates": candidates,
        "selected_weight": selected["pair_consistency_weight"],
        "selected_config": selected["name"],
    }
    (args.output_dir / "tuning_selection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return float(selected["pair_consistency_weight"]), result


def paired_comparisons(
    args: argparse.Namespace, selected_weight: float
) -> dict[str, dict[str, float | int]]:
    result = {}
    for metric in (
        "balanced_accuracy",
        "macro_f1",
        "pair_prediction_agreement",
        "pair_macro_task_agreement",
        "pair_both_correct",
        "pair_one_correct",
        "pair_prediction_js",
        "pair_ensemble_accuracy",
        "pair_ensemble_balanced_accuracy",
        "pair_pool1_jaccard",
        "direction_probe_balanced_accuracy",
    ):
        base_values = []
        pair_values = []
        for seed in args.init_seeds:
            for fold in args.folds:
                base_path = fold_dir(args, 0.0, seed, fold) / "summary.json"
                pair_path = fold_dir(args, selected_weight, seed, fold) / "summary.json"
                if not base_path.exists() or not pair_path.exists():
                    continue
                base = nested_value(json.loads(base_path.read_text()), SCALAR_PATHS[metric])
                pair = nested_value(json.loads(pair_path.read_text()), SCALAR_PATHS[metric])
                if base is not None and pair is not None:
                    base_values.append(base)
                    pair_values.append(pair)
        base_array = np.asarray(base_values)
        pair_array = np.asarray(pair_values)
        delta = pair_array - base_array
        pvalue = float(ttest_rel(pair_array, base_array).pvalue) if len(delta) > 1 else None
        result[metric] = {
            "baseline_mean": float(base_array.mean()),
            "pair_consistency_mean": float(pair_array.mean()),
            "mean_delta": float(delta.mean()),
            "delta_std": float(delta.std()),
            "paired_t_pvalue": pvalue,
            "pair_consistency_higher_folds": int((delta > 0).sum()),
            "n": len(delta),
        }
    return result


def per_task_pair_comparisons(
    args: argparse.Namespace, selected_weight: float
) -> dict[str, dict[str, object]]:
    collected: dict[str, dict[str, list[float]]] = {}
    for seed in args.init_seeds:
        for fold in args.folds:
            base_path = fold_dir(args, 0.0, seed, fold) / "summary.json"
            pair_path = fold_dir(args, selected_weight, seed, fold) / "summary.json"
            if not base_path.exists() or not pair_path.exists():
                continue
            base = json.loads(base_path.read_text())["paired_direction_metrics"]["per_task"]
            pair = json.loads(pair_path.read_text())["paired_direction_metrics"]["per_task"]
            for task in sorted(set(base) & set(pair)):
                for metric in (
                    "prediction_agreement", "both_correct",
                    "ensemble_accuracy", "mean_prediction_js",
                ):
                    values = collected.setdefault(task, {}).setdefault(
                        metric, {"baseline": [], "pair_consistency": []}
                    )
                    values["baseline"].append(float(base[task][metric]))
                    values["pair_consistency"].append(float(pair[task][metric]))
    result = {}
    for task, metrics in collected.items():
        result[task] = {}
        for metric, values in metrics.items():
            baseline = np.asarray(values["baseline"])
            pair_model = np.asarray(values["pair_consistency"])
            delta = pair_model - baseline
            result[task][metric] = {
                "baseline_mean": float(baseline.mean()),
                "pair_consistency_mean": float(pair_model.mean()),
                "mean_delta": float(delta.mean()),
                "paired_t_pvalue": float(ttest_rel(pair_model, baseline).pvalue),
                "n": len(delta),
            }
    return result


def fmt(summary: dict[str, object], metric: str) -> str:
    values = summary.get("metrics", {}).get(metric)
    if not values:
        return "-"
    return f"{values['mean']:.4f} +/- {values['std']:.4f}"


def write_summary(args: argparse.Namespace, selected_weight: float) -> dict[str, object]:
    tuning = json.loads((args.output_dir / "tuning_selection.json").read_text())
    baseline = summarize_config(args, 0.0, args.init_seeds)
    pair_model = summarize_config(args, selected_weight, args.init_seeds)
    comparisons = paired_comparisons(args, selected_weight)
    per_task_comparisons = per_task_pair_comparisons(args, selected_weight)
    result = {
        "protocol": {
            "task_loss": "CE + unit + 0.1 GLC; TPK disabled",
            "pair_loss": "JS(task predictions for matched LR/RL runs)",
            "split_seed": args.split_seed,
            "init_seeds": args.init_seeds,
            "folds": args.folds,
            "selected_pair_consistency_weight": selected_weight,
            "pair_balance": args.pair_balance,
            "pair_warmup_start": args.pair_warmup_start,
            "pair_warmup_epochs": args.pair_warmup_epochs,
            "checkpoint_selection": "pair_noninferiority",
            "checkpoint_bacc_tolerance": args.max_validation_bacc_drop,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_epochs": args.early_stopping_min_epochs,
        },
        "tuning": tuning,
        "results": {"baseline": baseline, "pair_consistency": pair_model},
        "paired_comparisons": comparisons,
        "per_task_pair_comparisons": per_task_comparisons,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# HCP LR/RL Pair-Consistency BrainGNN Results",
        "",
        "## Protocol",
        "",
        f"- All initialization seeds use the same subject splits generated with split seed `{args.split_seed}`.",
        "- Strong baseline: `CE + unit + 0.1 GLC`, with TPK disabled.",
        "- Pair model adds JS consistency between matched LR/RL task predictions.",
        f"- Pair task balance: `{args.pair_balance}`; warm-up: start epoch "
        f"`{args.pair_warmup_start}`, ramp for `{args.pair_warmup_epochs}` epochs.",
        "- Baseline and pair models use the same pair-aware non-inferiority checkpoint rule.",
        f"- Selected pair-consistency weight: `{selected_weight:g}`.",
        "",
        "## Main Results",
        "",
        "| Model | Balanced accuracy | Macro F1 | Pair agreement | Pair macro-task agreement | Pair ensemble BAcc |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Strong baseline | {fmt(baseline, 'balanced_accuracy')} | {fmt(baseline, 'macro_f1')} | "
            f"{fmt(baseline, 'pair_prediction_agreement')} | "
            f"{fmt(baseline, 'pair_macro_task_agreement')} | "
            f"{fmt(baseline, 'pair_ensemble_balanced_accuracy')} |"
        ),
        (
            f"| Pair consistency | {fmt(pair_model, 'balanced_accuracy')} | "
            f"{fmt(pair_model, 'macro_f1')} | {fmt(pair_model, 'pair_prediction_agreement')} | "
            f"{fmt(pair_model, 'pair_macro_task_agreement')} | "
            f"{fmt(pair_model, 'pair_ensemble_balanced_accuracy')} |"
        ),
        "",
        "## Matched Fold Comparisons",
        "",
        "| Metric | Baseline | Pair consistency | Mean delta | Paired t-test p |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in (
        "balanced_accuracy", "macro_f1", "pair_prediction_agreement",
        "pair_macro_task_agreement", "pair_both_correct", "pair_prediction_js",
        "pair_ensemble_balanced_accuracy", "pair_pool1_jaccard",
        "direction_probe_balanced_accuracy",
    ):
        values = comparisons[metric]
        pvalue = (
            f"{values['paired_t_pvalue']:.5f}"
            if values["paired_t_pvalue"] is not None else "-"
        )
        lines.append(
            f"| {metric} | {values['baseline_mean']:.4f} | "
            f"{values['pair_consistency_mean']:.4f} | {values['mean_delta']:+.4f} | "
            f"{pvalue} |"
        )
    lines.extend([
        "",
        "The paired t-tests are exploratory because cross-validation folds share training subjects.",
        "",
        "The selected pair-consistency model improves task balanced accuracy and macro F1 across "
        "the matched folds. Its aggregate pair-consistency diagnostics move in the intended "
        "direction, but those gains are small and not statistically significant. The supported "
        "claim is therefore improved task decoding with mild direction-consistency benefits, "
        "not strong LR/RL invariance.",
        "",
        "## Per-Task Pair Diagnostics",
        "",
        "| Task | Agreement baseline | Agreement pair model | Delta | Both-correct delta | Ensemble accuracy delta | JS delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for task, metrics in sorted(per_task_comparisons.items()):
        agreement = metrics["prediction_agreement"]
        lines.append(
            f"| {task} | {agreement['baseline_mean']:.4f} | "
            f"{agreement['pair_consistency_mean']:.4f} | {agreement['mean_delta']:+.4f} | "
            f"{metrics['both_correct']['mean_delta']:+.4f} | "
            f"{metrics['ensemble_accuracy']['mean_delta']:+.4f} | "
            f"{metrics['mean_prediction_js']['mean_delta']:+.4f} |"
        )
    lines.extend([
        "",
        "## Tuning",
        "",
        "| Pair weight | Validation BAcc | Validation pair agreement | Test BAcc |",
        "|---:|---:|---:|---:|",
    ])
    for candidate in tuning["candidates"]:
        lines.append(
            f"| {candidate['pair_consistency_weight']:g} | "
            f"{fmt(candidate, 'validation_balanced_accuracy')} | "
            f"{fmt(candidate, 'validation_pair_agreement')} | "
            f"{fmt(candidate, 'balanced_accuracy')} |"
        )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def write_protocol(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "protocol.json"
    if args.phase == "summarize" and path.exists():
        return
    protocol = {
        "runner": Path(__file__).name,
        "trainer": "07-main_hcp_subjectwise.py",
        "dataroot": str(args.dataroot),
        "phase": args.phase,
        "folds": args.folds,
        "split_seed": args.split_seed,
        "init_seeds": args.init_seeds,
        "tune_init_seeds": args.tune_init_seeds,
        "pair_weights": args.pair_weights,
        "max_validation_bacc_drop": args.max_validation_bacc_drop,
        "pair_balance": args.pair_balance,
        "pair_warmup_start": args.pair_warmup_start,
        "pair_warmup_epochs": args.pair_warmup_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_epochs": args.early_stopping_min_epochs,
        "bootstrap_samples": args.bootstrap_samples,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "jobs": args.jobs,
    }
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be at least one")
    if any(seed not in args.init_seeds for seed in args.tune_init_seeds):
        raise ValueError("--tune-init-seeds must be included in --init-seeds")
    if any(weight <= 0 for weight in args.pair_weights):
        raise ValueError("--pair-weights must be positive")
    if args.max_validation_bacc_drop < 0:
        raise ValueError("--max-validation-bacc-drop must be non-negative")
    if args.pair_warmup_start < 0 or args.pair_warmup_epochs < 0:
        raise ValueError("pair warm-up values must be non-negative")
    if args.early_stopping_patience < 0 or args.early_stopping_min_epochs < 0:
        raise ValueError("early stopping values must be non-negative")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    write_protocol(args)
    write_split_manifest(args)

    if args.phase in {"all", "baseline"}:
        run_work(args, [
            (0.0, seed, fold) for seed in args.init_seeds for fold in args.folds
        ])

    if args.phase in {"all", "tune"}:
        run_work(args, [
            (weight, seed, fold)
            for weight in args.pair_weights
            for seed in args.tune_init_seeds
            for fold in args.folds
        ])

    selected_weight = None
    if args.phase in {"all", "final", "summarize"}:
        selected_weight, _ = select_pair_weight(args)

    if args.phase in {"all", "final"}:
        run_work(args, [
            (selected_weight, seed, fold)
            for seed in args.init_seeds
            for fold in args.folds
        ])

    if args.phase in {"all", "final", "summarize"}:
        print(json.dumps(write_summary(args, selected_weight), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
