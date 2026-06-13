#!/usr/bin/env python
"""Run HCP cross-direction baselines and direction-adversarial BrainGNN experiments."""

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "hcp900_direction_robustness_20260612"
DEFAULT_SEEDS = (123, 456, 789)
DEFAULT_ADV_WEIGHTS = (0.01, 0.05, 0.1, 0.2)

SCALAR_PATHS = {
    "accuracy": ("metrics", "accuracy"),
    "balanced_accuracy": ("metrics", "balanced_accuracy"),
    "macro_f1": ("metrics", "macro_f1"),
    "validation_balanced_accuracy": ("selected_checkpoint_validation_metrics", "balanced_accuracy"),
    "lr_balanced_accuracy": ("direction_subgroup_metrics", "LR", "balanced_accuracy"),
    "rl_balanced_accuracy": ("direction_subgroup_metrics", "RL", "balanced_accuracy"),
    "pair_prediction_agreement": ("paired_direction_metrics", "prediction_agreement"),
    "pair_one_correct": ("paired_direction_metrics", "one_correct"),
    "pair_pool1_jaccard": ("paired_direction_metrics", "pool1_topk_jaccard"),
    "direction_probe_balanced_accuracy": ("direction_probe_metrics", "balanced_accuracy"),
    "direction_head_balanced_accuracy": ("direction_head_metrics", "balanced_accuracy"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=DEFAULT_DATAROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--phase",
        choices=["all", "baselines", "tune", "final", "summarize"],
        default="all",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--tune-seed", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--adv-weights", nargs="+", type=float, default=list(DEFAULT_ADV_WEIGHTS))
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def weight_tag(weight: float) -> str:
    return f"{weight:g}".replace(".", "p")


def config_name(protocol: str, adv_weight: float) -> str:
    return (
        f"baseline_{protocol}"
        if adv_weight == 0
        else f"direction_adv_{weight_tag(adv_weight)}_{protocol}"
    )


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
            raise ValueError(f"seed {seed} fold {fold} lacks one or more task-direction strata")
        train_index, val_index = valid_inner
        split_indices = {"train": train_index, "val": val_index, "test": test_index}
        split = {"fold": fold}
        for split_name, index in split_indices.items():
            split[f"{split_name}_subjects"] = sorted(set(groups[index].tolist()))
            split[f"n_{split_name}_samples"] = int(len(index))
        folds.append(split)
    return folds


def write_split_manifest(args: argparse.Namespace, seed: int) -> Path:
    split_dir = args.output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"seed_{seed}.json"
    if not path.exists():
        records = read_records(args.dataroot)
        manifest = {
            "dataroot": str(args.dataroot),
            "method": "StratifiedGroupKFold over 7 task x 2 direction strata",
            "n_subjects": len({record["subject"] for record in records}),
            "n_samples": len(records),
            "n_folds": 5,
            "seed": seed,
            "folds": build_direction_stratified_splits(records, seed),
        }
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def fold_command(
    args: argparse.Namespace,
    protocol: str,
    adv_weight: float,
    seed: int,
    fold: int,
    fold_dir: Path,
) -> list[str]:
    split_manifest = write_split_manifest(args, seed)
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "07-main_hcp_subjectwise.py"),
        "--dataroot", str(args.dataroot),
        "--split_manifest", str(split_manifest),
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
        "--direction_protocol", protocol,
        "--direction_adv_weight", str(adv_weight),
        "--seed", str(seed),
        "--output_dir", str(fold_dir),
    ]
    command.append("--save_model" if args.save_model else "--no-save_model")
    return command


def fold_dir(args: argparse.Namespace, protocol: str, adv_weight: float, seed: int, fold: int) -> Path:
    return args.output_dir / "runs" / config_name(protocol, adv_weight) / f"seed_{seed}" / f"fold_{fold}"


def run_fold(
    args: argparse.Namespace, protocol: str, adv_weight: float, seed: int, fold: int
) -> str:
    output = fold_dir(args, protocol, adv_weight, seed, fold)
    summary = output / "summary.json"
    name = config_name(protocol, adv_weight)
    if args.skip_completed and summary.exists():
        return f"skip completed {name} seed={seed} fold={fold}"
    command = fold_command(args, protocol, adv_weight, seed, fold, output)
    print(" ".join(command), flush=True)
    if args.dry_run:
        return f"dry run {name} seed={seed} fold={fold}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (output / "stdout.log").open("w") as stdout:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True, stdout=stdout, stderr=subprocess.STDOUT)
    return f"completed {name} seed={seed} fold={fold}"


def run_work(args: argparse.Namespace, work: list[tuple[str, float, int, int]]) -> None:
    if not work:
        return
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_fold, args, protocol, weight, seed, fold):
                (protocol, weight, seed, fold)
            for protocol, weight, seed, fold in work
        }
        for future in as_completed(futures):
            protocol, weight, seed, fold = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:
                raise RuntimeError(
                    f"failed {config_name(protocol, weight)} seed={seed} fold={fold}"
                ) from error


def nested_value(data: dict[str, object], path: tuple[str, ...]) -> float | None:
    value: object = data
    for key in path:
        if not isinstance(value, dict) or key not in value or value[key] is None:
            return None
        value = value[key]
    return float(value)


def fold_summaries(
    args: argparse.Namespace, protocol: str, adv_weight: float, seeds: list[int]
) -> dict[tuple[int, int], dict[str, object]]:
    summaries = {}
    for seed in seeds:
        for fold in args.folds:
            path = fold_dir(args, protocol, adv_weight, seed, fold) / "summary.json"
            if path.exists():
                summaries[(seed, fold)] = json.loads(path.read_text())
    return summaries


def direction_gap(summary: dict[str, object]) -> float | None:
    subgroups = summary.get("direction_subgroup_metrics", {})
    if "LR" not in subgroups or "RL" not in subgroups:
        return None
    return abs(
        float(subgroups["LR"]["balanced_accuracy"])
        - float(subgroups["RL"]["balanced_accuracy"])
    )


def paired_comparisons(args: argparse.Namespace, selected_weight: float) -> dict[str, object]:
    baseline = fold_summaries(args, "mixed", 0.0, args.seeds)
    adversarial = fold_summaries(args, "mixed", selected_weight, args.seeds)
    matched = sorted(set(baseline) & set(adversarial))
    comparisons = {}
    paths = {
        "balanced_accuracy": SCALAR_PATHS["balanced_accuracy"],
        "macro_f1": SCALAR_PATHS["macro_f1"],
        "pair_prediction_agreement": SCALAR_PATHS["pair_prediction_agreement"],
        "pair_one_correct": SCALAR_PATHS["pair_one_correct"],
        "pair_pool1_jaccard": SCALAR_PATHS["pair_pool1_jaccard"],
        "direction_probe_balanced_accuracy": SCALAR_PATHS["direction_probe_balanced_accuracy"],
    }
    for name, path in paths.items():
        paired = [
            (nested_value(baseline[key], path), nested_value(adversarial[key], path))
            for key in matched
        ]
        paired = [(base, adv) for base, adv in paired if base is not None and adv is not None]
        base = np.asarray([item[0] for item in paired])
        adv = np.asarray([item[1] for item in paired])
        delta = adv - base
        comparisons[name] = {
            "baseline_mean": float(base.mean()),
            "adversarial_mean": float(adv.mean()),
            "mean_delta": float(delta.mean()),
            "delta_std": float(delta.std()),
            "paired_t_pvalue": float(ttest_rel(adv, base).pvalue),
            "adversarial_higher_folds": int((delta > 0).sum()),
            "n": len(delta),
        }
    base_gap = np.asarray([direction_gap(baseline[key]) for key in matched])
    adv_gap = np.asarray([direction_gap(adversarial[key]) for key in matched])
    gap_delta = adv_gap - base_gap
    comparisons["absolute_lr_rl_balanced_accuracy_gap"] = {
        "baseline_mean": float(base_gap.mean()),
        "adversarial_mean": float(adv_gap.mean()),
        "mean_delta": float(gap_delta.mean()),
        "delta_std": float(gap_delta.std()),
        "paired_t_pvalue": float(ttest_rel(adv_gap, base_gap).pvalue),
        "adversarial_higher_folds": int((gap_delta > 0).sum()),
        "n": len(gap_delta),
    }
    return comparisons


def summarize_config(
    args: argparse.Namespace, protocol: str, adv_weight: float, seeds: list[int]
) -> dict[str, object]:
    rows = []
    for seed in seeds:
        for fold in args.folds:
            path = fold_dir(args, protocol, adv_weight, seed, fold) / "summary.json"
            if path.exists():
                summary = json.loads(path.read_text())
                rows.append({"seed": seed, "fold": fold, "summary": summary})
    scalars = {}
    for name, path in SCALAR_PATHS.items():
        values = [nested_value(row["summary"], path) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            scalars[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": len(values),
            }
    per_seed = {}
    for seed in seeds:
        seed_rows = [row for row in rows if row["seed"] == seed]
        if seed_rows:
            per_seed[str(seed)] = {
                name: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "n": len(values),
                }
                for name, path in SCALAR_PATHS.items()
                if (values := [
                    value
                    for row in seed_rows
                    if (value := nested_value(row["summary"], path)) is not None
                ])
            }
    return {
        "name": config_name(protocol, adv_weight),
        "protocol": protocol,
        "direction_adv_weight": adv_weight,
        "requested_seeds": seeds,
        "completed_folds": len(rows),
        "metrics": scalars,
        "per_seed": per_seed,
    }


def select_adv_weight(args: argparse.Namespace) -> tuple[float, dict[str, object]]:
    candidates = [
        summarize_config(args, "mixed", weight, [args.tune_seed])
        for weight in args.adv_weights
    ]
    complete = [
        candidate
        for candidate in candidates
        if candidate["completed_folds"] == len(args.folds)
        and "validation_balanced_accuracy" in candidate["metrics"]
    ]
    if not complete:
        raise ValueError("no complete adversarial tuning candidate is available")
    selected = max(
        complete, key=lambda candidate: candidate["metrics"]["validation_balanced_accuracy"]["mean"]
    )
    result = {
        "selection_metric": "selected-checkpoint validation balanced accuracy",
        "tune_seed": args.tune_seed,
        "candidates": candidates,
        "selected_weight": selected["direction_adv_weight"],
        "selected_config": selected["name"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tuning_selection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return float(selected["direction_adv_weight"]), result


def fmt(summary: dict[str, object], metric: str) -> str:
    values = summary.get("metrics", {}).get(metric)
    if not values:
        return "-"
    return f"{values['mean']:.4f} +/- {values['std']:.4f}"


def write_summary(args: argparse.Namespace, selected_weight: float | None = None) -> dict[str, object]:
    if selected_weight is None:
        selected_weight, tuning = select_adv_weight(args)
    else:
        tuning = json.loads((args.output_dir / "tuning_selection.json").read_text())
    summaries = {
        "baseline_mixed": summarize_config(args, "mixed", 0.0, args.seeds),
        "baseline_lr_to_rl": summarize_config(args, "lr_to_rl", 0.0, [args.tune_seed]),
        "baseline_rl_to_lr": summarize_config(args, "rl_to_lr", 0.0, [args.tune_seed]),
        "direction_adversarial": summarize_config(args, "mixed", selected_weight, args.seeds),
    }
    comparisons = paired_comparisons(args, selected_weight)
    result = {
        "protocol": {
            "task_loss": "CE + unit + 0.1 GLC; TPK disabled",
            "seeds": args.seeds,
            "folds": args.folds,
            "tune_seed": args.tune_seed,
            "selected_direction_adv_weight": selected_weight,
        },
        "tuning": tuning,
        "results": summaries,
        "paired_comparisons": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# HCP Direction-Robust BrainGNN Results",
        "",
        "## Protocol",
        "",
        "- Strong baseline: `CE + unit + 0.1 GLC`, with TPK disabled.",
        f"- Five folds and seeds: `{args.seeds}`.",
        "- Each seed controls both the task-direction-stratified subject splits and model initialization.",
        f"- Direction-adversarial weight selected on seed `{args.tune_seed}` validation folds: `{selected_weight:g}`.",
        "- Cross-direction baselines use source-direction train/validation graphs and target-direction test graphs with subject-disjoint splits.",
        "- These direction-stratified results are not directly comparable to the earlier random-split reproduction result.",
        "",
        "## Main Three-Seed Results",
        "",
        "| Model | Balanced accuracy | Macro F1 | LR/RL pair agreement | Direction probe balanced accuracy |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Strong baseline | {fmt(summaries['baseline_mixed'], 'balanced_accuracy')} | "
            f"{fmt(summaries['baseline_mixed'], 'macro_f1')} | "
            f"{fmt(summaries['baseline_mixed'], 'pair_prediction_agreement')} | "
            f"{fmt(summaries['baseline_mixed'], 'direction_probe_balanced_accuracy')} |"
        ),
        (
            f"| Direction-adversarial | {fmt(summaries['direction_adversarial'], 'balanced_accuracy')} | "
            f"{fmt(summaries['direction_adversarial'], 'macro_f1')} | "
            f"{fmt(summaries['direction_adversarial'], 'pair_prediction_agreement')} | "
            f"{fmt(summaries['direction_adversarial'], 'direction_probe_balanced_accuracy')} |"
        ),
        "",
        "## Cross-Direction Baselines",
        "",
        "| Protocol | Balanced accuracy | Macro F1 |",
        "|---|---:|---:|",
        f"| LR to RL | {fmt(summaries['baseline_lr_to_rl'], 'balanced_accuracy')} | {fmt(summaries['baseline_lr_to_rl'], 'macro_f1')} |",
        f"| RL to LR | {fmt(summaries['baseline_rl_to_lr'], 'balanced_accuracy')} | {fmt(summaries['baseline_rl_to_lr'], 'macro_f1')} |",
        "",
        "## Paired Baseline Versus Direction-Adversarial Comparison",
        "",
        "| Metric | Baseline mean | Adversarial mean | Mean delta | Paired t-test p |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in (
        "balanced_accuracy",
        "macro_f1",
        "pair_prediction_agreement",
        "direction_probe_balanced_accuracy",
        "absolute_lr_rl_balanced_accuracy_gap",
    ):
        values = comparisons[metric]
        lines.append(
            f"| {metric} | {values['baseline_mean']:.4f} | "
            f"{values['adversarial_mean']:.4f} | {values['mean_delta']:+.4f} | "
            f"{values['paired_t_pvalue']:.5f} |"
        )
    lines.extend([
        "",
        "The selected direction-adversarial model does not meet the predefined success criteria "
        "across all 15 matched seed-fold pairs. Its task balanced accuracy is slightly lower, "
        "while the small gains in pair agreement and the reduction in direction decodability are "
        "not statistically significant.",
        "",
        "The paired t-tests are exploratory because cross-validation folds share training subjects "
        "and the 15 seed-fold outcomes are not fully independent.",
        "",
        "The cross-direction baselines are substantially below the mixed-direction baseline, "
        "establishing a difficult direction-transfer setting. Because source-only training also "
        "uses fewer labeled graphs, this comparison alone does not isolate the effect of direction shift.",
        "",
        "## Tuning",
        "",
        "| Adversarial weight | Validation balanced accuracy | Test balanced accuracy |",
        "|---:|---:|---:|",
    ])
    for candidate in tuning["candidates"]:
        lines.append(
            f"| {candidate['direction_adv_weight']:g} | "
            f"{fmt(candidate, 'validation_balanced_accuracy')} | "
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
        "seeds": args.seeds,
        "tune_seed": args.tune_seed,
        "adv_weights": args.adv_weights,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "jobs": args.jobs,
        "strong_baseline": {
            "unit": 1.0,
            "tpk": 0.0,
            "glc": 0.1,
            "edge_source": "pcorr",
            "edge_top_percent": 0.10,
            "positive_edges_only": True,
        },
    }
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be at least one")
    if args.tune_seed not in args.seeds:
        raise ValueError("--tune-seed must be included in --seeds")
    if any(weight <= 0 for weight in args.adv_weights):
        raise ValueError("--adv-weights must be positive")
    write_protocol(args)
    for seed in set(args.seeds):
        write_split_manifest(args, seed)

    if args.phase in {"all", "baselines"}:
        work = [
            ("mixed", 0.0, seed, fold)
            for seed in args.seeds
            for fold in args.folds
        ]
        work.extend(
            (protocol, 0.0, args.tune_seed, fold)
            for protocol in ("lr_to_rl", "rl_to_lr")
            for fold in args.folds
        )
        run_work(args, work)

    if args.phase in {"all", "tune"}:
        run_work(
            args,
            [
                ("mixed", weight, args.tune_seed, fold)
                for weight in args.adv_weights
                for fold in args.folds
            ],
        )

    selected_weight = None
    if args.phase in {"all", "final", "summarize"}:
        selected_weight, _ = select_adv_weight(args)

    if args.phase in {"all", "final"}:
        run_work(
            args,
            [
                ("mixed", selected_weight, seed, fold)
                for seed in args.seeds
                for fold in args.folds
            ],
        )

    if args.phase in {"all", "final", "summarize"}:
        result = write_summary(args, selected_weight)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
