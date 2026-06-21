#!/usr/bin/env python
"""Plot local equivalents of the BrainGNN Section 3.5 interpretation figures."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
import nibabel as nib
import numpy as np


TASK_ORDER = ("EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")
GLC_VALUES = ("0", "0.1", "0.5")
CURRENT_GLC_RUNS = {
    "0": "ce_unit_tpk",
    "0.1": "paper_like",
    "0.5": "tpk_0.1_glc_0.5",
}
LOBE_REGIONS = {
    "occipital": ("cuneus", "lateraloccipital", "lingual", "pericalcarine"),
    "parietal": (
        "inferiorparietal", "postcentral", "precuneus", "superiorparietal",
        "supramarginal", "paracentral", "posteriorcingulate", "isthmuscingulate",
    ),
    "frontal": (
        "caudalanteriorcingulate", "caudalmiddlefrontal", "lateralorbitofrontal",
        "medialorbitofrontal", "parsopercularis", "parsorbitalis", "parstriangularis",
        "precentral", "rostralanteriorcingulate", "rostralmiddlefrontal",
        "superiorfrontal", "frontalpole",
    ),
    "temporal": (
        "bankssts", "entorhinal", "fusiform", "inferiortemporal", "middletemporal",
        "parahippocampal", "superiortemporal", "temporalpole", "transversetemporal",
    ),
    "insula": ("insula",),
}
LOBE_BOXES = {
    "occipital": (-0.68, -0.34, -0.18, 0.46),
    "parietal": (-0.34, 0.02, 0.30, 0.58),
    "frontal": (0.08, 0.68, -0.02, 0.58),
    "temporal": (-0.36, 0.50, -0.58, -0.20),
    "insula": (-0.04, 0.04, -0.06, 0.06),
}


def cjk_title_options(fontsize: int) -> dict[str, object]:
    candidates = []
    for root in (Path.cwd(), Path(__file__).resolve().parent):
        candidates.extend((root / ".tmp" / "tectonic-cache").rglob("FandolSong-Regular.otf"))
    if candidates:
        return {"fontsize": fontsize, "fontproperties": font_manager.FontProperties(fname=str(candidates[0]))}
    return {"fontsize": fontsize}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold", type=int, default=0, help="fold used for the individual/group ROI comparison")
    parser.add_argument("--task", default="SOCIAL", choices=TASK_ORDER)
    parser.add_argument("--top-rois", type=int, default=17)
    return parser.parse_args()


def atlas_rois(atlas_path: Path) -> list[dict[str, object]]:
    image = nib.load(str(atlas_path))
    values = np.asarray(image.get_fdata()).squeeze().astype(np.int32)
    roi_ids = sorted(int(value) for value in np.unique(values) if int(value) != 0)
    label_table = image.header.get_axis(0).label[0]
    return [
        {
            "zero_based_index": index,
            "atlas_id": roi_id,
            "name": str(label_table[roi_id][0]),
        }
        for index, roi_id in enumerate(roi_ids)
    ]


def local_grid(regions: tuple[str, ...], box: tuple[float, float, float, float]) -> dict[str, tuple[float, float]]:
    xmin, xmax, ymin, ymax = box
    columns = 3 if len(regions) > 4 else 2
    rows = int(np.ceil(len(regions) / columns))
    xs = np.linspace(xmin, xmax, columns)
    ys = np.linspace(ymax, ymin, rows)
    return {region: (float(xs[index % columns]), float(ys[index // columns]))
            for index, region in enumerate(regions)}


def schematic_positions(rois: list[dict[str, object]]) -> np.ndarray:
    region_positions = {}
    for lobe, regions in LOBE_REGIONS.items():
        region_positions.update(local_grid(regions, LOBE_BOXES[lobe]))
    positions = []
    for roi in rois:
        hemisphere, region = str(roi["name"]).split("_", 1)
        local_x, local_y = region_positions[region]
        center = -1.05 if hemisphere == "L" else 1.05
        direction = -1.0 if hemisphere == "L" else 1.0
        positions.append((center + direction * local_x, local_y))
    return np.asarray(positions, dtype=np.float32)


def top_indices(values: np.ndarray, count: int) -> np.ndarray:
    return np.argsort(values)[-count:][::-1]


def mean_pairwise_jaccard(selections: list[set[int]]) -> float:
    values = [
        len(first & second) / len(first | second)
        for first, second in itertools.combinations(selections, 2)
    ]
    return float(np.mean(values))


def add_brain_outline(ax: plt.Axes) -> None:
    for center in (-1.05, 1.05):
        ax.add_patch(Ellipse((center, 0), 1.75, 1.42, facecolor="#f4f4f4",
                             edgecolor="#8a8a8a", linewidth=0.8, zorder=0))
    ax.text(-1.05, 0.78, "L", ha="center", va="bottom", fontsize=7, color="#555555")
    ax.text(1.05, 0.78, "R", ha="center", va="bottom", fontsize=7, color="#555555")


def plot_saliency_panel(
    ax: plt.Axes,
    positions: np.ndarray,
    values: np.ndarray,
    selected: set[int],
    title: str,
    vmin: float,
    vmax: float,
    common: set[int] | None = None,
):
    add_brain_outline(ax)
    unselected = sorted(set(range(len(values))) - selected)
    if unselected:
        ax.scatter(positions[unselected, 0], positions[unselected, 1], s=9, c="#d6d6d6",
                   edgecolors="none", zorder=1)
    chosen = sorted(selected)
    scatter = ax.scatter(
        positions[chosen, 0], positions[chosen, 1], s=30, c=values[chosen],
        cmap="YlOrRd", vmin=vmin, vmax=vmax, edgecolors="#444444", linewidths=0.25, zorder=2,
    )
    if common:
        common_indices = sorted(common)
        ax.scatter(positions[common_indices, 0], positions[common_indices, 1], s=65,
                   facecolors="none", edgecolors="#1769aa", linewidths=1.0, zorder=3)
    ax.set_title(title, fontsize=8)
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-0.85, 0.92)
    ax.set_aspect("equal")
    ax.axis("off")
    return scatter


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_pool_scores(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    score_rows, label_rows, subject_rows, sample_rows = [], [], [], []
    for path in sorted(root.glob("fold_*/test_pool1_scores.npz")):
        values = np.load(path)
        score_rows.append(values["scores"])
        label_rows.append(values["true"])
        subject_rows.append(values["subjects"].astype(str))
        sample_rows.append(values["sample_names"].astype(str))
    if not score_rows:
        raise ValueError(f"no fold score files under {root}")
    return (
        np.concatenate(score_rows),
        np.concatenate(label_rows),
        np.concatenate(subject_rows),
        np.concatenate(sample_rows),
    )


def sample_lookup(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    values = np.load(path)
    return {
        (str(subject), str(sample)): {
            "score": score,
            "true": int(true),
            "pred": int(pred),
        }
        for subject, sample, score, true, pred in zip(
            values["subjects"], values["sample_names"], values["scores"], values["true"], values["pred"]
        )
    }


def sample_matches_task(sample_name: str, task: str) -> bool:
    return sample_name == task or sample_name.startswith(f"{task}_")


def current_or_legacy_run_root(experiment_root: Path, run_name: str) -> Path:
    current = experiment_root / "runs" / run_name
    if current.exists():
        return current
    if run_name == "paper_like":
        return experiment_root / "main_paper_consistent" / "paper_like"
    return experiment_root / "lambda_sweep" / run_name


def figure5_glc_comparison(
    experiment_root: Path,
    output_dir: Path,
    positions: np.ndarray,
    rois: list[dict[str, object]],
    fold: int,
    task: str,
    top_rois: int,
) -> dict[str, object]:
    lookups = {
        glc: sample_lookup(
            current_or_legacy_run_root(
                experiment_root,
                CURRENT_GLC_RUNS[glc] if (experiment_root / "runs").exists() else f"tpk_0.1_glc_{glc}",
            ) / f"fold_{fold}" / "test_pool1_scores.npz"
        )
        for glc in GLC_VALUES
    }
    common_keys = set.intersection(*(set(lookup) for lookup in lookups.values()))
    candidates = [
        key for key in sorted(common_keys)
        if sample_matches_task(key[1], task)
        and all(lookups[glc][key]["true"] == lookups[glc][key]["pred"] for glc in GLC_VALUES)
    ]
    if len(candidates) < 3:
        candidates = [key for key in sorted(common_keys) if sample_matches_task(key[1], task)]
    selected_keys = candidates[:3]
    if len(selected_keys) < 3:
        raise ValueError(f"fewer than three shared {task} samples in fold {fold}")

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.8), constrained_layout=True)
    details: dict[str, object] = {"fold": fold, "task": task, "top_rois": top_rois, "rows": {}}
    last_scatter = None
    for row, glc in enumerate(GLC_VALUES):
        scores = [np.asarray(lookups[glc][key]["score"]) for key in selected_keys]
        selections = [set(top_indices(score, top_rois).tolist()) for score in scores]
        common = set.intersection(*selections)
        jaccard = mean_pairwise_jaccard(selections)
        details["rows"][glc] = {
            "mean_pairwise_jaccard": jaccard,
            "common_roi_count": len(common),
            "common_rois": [rois[index] for index in sorted(common)],
            "samples": [
                {
                    "subject": key[0],
                    "sample": key[1],
                    "top_rois": [rois[index] for index in top_indices(score, top_rois)],
                }
                for key, score in zip(selected_keys, scores)
            ],
        }
        for column, (key, score, selection) in enumerate(zip(selected_keys, scores, selections)):
            last_scatter = plot_saliency_panel(
                axes[row, column], positions, score, selection,
                f"Subject {key[0]}" if row == 0 else "",
                vmin=0.1, vmax=1.0, common=common,
            )
        axes[row, 0].text(
            -2.25, 0, f"GLC lambda2={glc}\nJaccard={jaccard:.3f}\nCommon={len(common)}",
            ha="right", va="center", fontsize=8,
        )
    fig.suptitle(
        f"图 3.4  GLC 正则下的个体/群体 ROI 选择（{task}, first-pool top-{top_rois}）",
        **cjk_title_options(11),
    )
    fig.colorbar(last_scatter, ax=axes, shrink=0.55, label="First R-pool score")
    save_figure(fig, output_dir, "figure5_glc_individual_group")
    return details


def task_mean_scores(main_root: Path) -> dict[str, np.ndarray]:
    scores, labels, _, _ = load_pool_scores(main_root)
    return {
        task: scores[labels == label].mean(axis=0)
        for label, task in enumerate(TASK_ORDER)
    }


def figure7_task_saliency(
    output_dir: Path,
    positions: np.ndarray,
    rois: list[dict[str, object]],
    means: dict[str, np.ndarray],
    top_rois: int,
) -> dict[str, object]:
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.7), constrained_layout=True)
    axes_flat = axes.ravel()
    details = {}
    last_scatter = None
    for axis, task in zip(axes_flat, TASK_ORDER):
        values = means[task]
        selected_array = top_indices(values, top_rois)
        selected = set(selected_array.tolist())
        last_scatter = plot_saliency_panel(
            axis, positions, values, selected, task, vmin=0.1, vmax=1.0,
        )
        details[task] = [
            {**rois[index], "mean_first_pool_score": float(values[index])}
            for index in selected_array
        ]
    axes_flat[-1].axis("off")
    fig.suptitle(
        f"图 3.5  七类 HCP 任务的 ROI saliency（mean first-pool score; top-{top_rois}）",
        **cjk_title_options(11),
    )
    fig.colorbar(last_scatter, ax=axes, shrink=0.62, label="Mean first R-pool score")
    save_figure(fig, output_dir, "figure7_task_salient_rois")
    return details


def figure8_proxy(output_dir: Path, means: dict[str, np.ndarray], top_rois: int) -> dict[str, object]:
    selections = {task: set(top_indices(values, top_rois).tolist()) for task, values in means.items()}
    matrix = np.zeros((len(TASK_ORDER), len(TASK_ORDER)), dtype=np.float32)
    for row, first in enumerate(TASK_ORDER):
        for column, second in enumerate(TASK_ORDER):
            matrix[row, column] = len(selections[first] & selections[second]) / len(
                selections[first] | selections[second]
            )
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Reds", vmin=0, vmax=1)
    for row in range(len(TASK_ORDER)):
        for column in range(len(TASK_ORDER)):
            ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if matrix[row, column] > 0.55 else "black")
    ax.set_xticks(range(len(TASK_ORDER)), TASK_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(len(TASK_ORDER)), TASK_ORDER)
    ax.set_title(
        "图 3.6  任务 top ROI 集合的 Jaccard 相似性",
        **cjk_title_options(11),
    )
    fig.colorbar(image, ax=ax, label=f"Top-{top_rois} ROI Jaccard")
    save_figure(fig, output_dir, "figure8_proxy_task_roi_similarity")
    return {"tasks": list(TASK_ORDER), "top_roi_jaccard": matrix.tolist()}


def best_fold(main_root: Path) -> Path:
    candidates = []
    for summary_path in main_root.glob("fold_*/summary.json"):
        summary = json.loads(summary_path.read_text())
        candidates.append((float(summary["metrics"]["balanced_accuracy"]), summary_path.parent))
    if not candidates:
        raise ValueError(f"no completed folds under {main_root}")
    return max(candidates, key=lambda item: item[0])[1]


def community_details(alpha: np.ndarray, rois: list[dict[str, object]]) -> list[dict[str, object]]:
    threshold = alpha.mean(axis=1) + alpha.std(axis=1)
    rows = []
    for index, values in enumerate(alpha):
        memberships = np.where(values > threshold[index])[0].tolist()
        strongest = int(np.argmax(values)) if memberships else None
        rows.append(
            {
                **rois[index],
                "threshold": float(threshold[index]),
                "community_memberships_zero_based": memberships,
                "strongest_qualifying_community_zero_based": strongest,
                "alpha_positive": values.tolist(),
            }
        )
    return rows


def figure9_communities(
    output_dir: Path,
    positions: np.ndarray,
    alpha: np.ndarray,
    rois: list[dict[str, object]],
    fold_name: str,
) -> list[dict[str, object]]:
    details = community_details(alpha, rois)
    assignments = np.asarray([
        -1 if row["strongest_qualifying_community_zero_based"] is None
        else int(row["strongest_qualifying_community_zero_based"])
        for row in details
    ])
    palette = plt.get_cmap("tab10")(np.arange(8))
    fig, ax = plt.subplots(figsize=(8.5, 3.5), constrained_layout=True)
    add_brain_outline(ax)
    unassigned = np.where(assignments < 0)[0]
    assigned = np.where(assignments >= 0)[0]
    ax.scatter(positions[unassigned, 0], positions[unassigned, 1], s=28, c="#d6d6d6",
               edgecolors="#666666", linewidths=0.25, label="No qualifying community")
    scatter = ax.scatter(
        positions[assigned, 0], positions[assigned, 1], s=55, c=assignments[assigned],
        cmap=ListedColormap(palette), vmin=-0.5, vmax=7.5, edgecolors="#333333", linewidths=0.35,
    )
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-0.85, 0.92)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"图 3.7  Ra-GConv community 归属（{fold_name}; mean+std 阈值）",
        **cjk_title_options(11),
    )
    colorbar = fig.colorbar(scatter, ax=ax, ticks=range(8), shrink=0.75)
    colorbar.set_label("Ra-GConv community (zero-based)")
    save_figure(fig, output_dir, "figure9_community_assignments")
    return details


def figure10_alpha_heatmap(output_dir: Path, alpha: np.ndarray, fold_name: str) -> dict[str, object]:
    fig, ax = plt.subplots(figsize=(13, 3.6), constrained_layout=True)
    image = ax.imshow(alpha.T, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xlabel("ROI zero-based index")
    ax.set_ylabel("Community zero-based index")
    ax.set_yticks(range(alpha.shape[1]))
    ax.set_xticks(np.arange(0, alpha.shape[0], 2))
    ax.set_title(
        f"图 3.8  第一层 Ra-GConv alpha+ 矩阵（{fold_name}）",
        **cjk_title_options(11),
    )
    fig.colorbar(image, ax=ax, label="Positive community-membership weight")
    save_figure(fig, output_dir, "figure10_alpha_positive_heatmap")
    return {
        "fold": fold_name,
        "shape": list(alpha.shape),
        "zero_fraction": float(np.mean(alpha == 0)),
        "mean": float(alpha.mean()),
        "max": float(alpha.max()),
    }


def write_notes(output_dir: Path, best_fold_name: str, top_rois: int) -> None:
    text = f"""# Section 3.5 Local Interpretation Figures

These figures are local-data equivalents of the interpretation figures in
BrainGNN Section 3.5. They are not exact reproductions of the paper figures.

- `figure5_glc_individual_group`: compares three correctly classified samples
  from the same task across `lambda2_GLC = 0, 0.1, 0.5`. Blue rings denote ROIs
  common to all three samples in a row.
- `figure7_task_salient_rois`: displays per-task mean first-pooling scores.
- `figure8_proxy_task_roi_similarity`: displays top-{top_rois} ROI-set Jaccard
  similarity. It is not the paper's Neurosynth decoding analysis.
- `figure9_community_assignments`: displays the strongest qualifying community
  for each ROI from `{best_fold_name}`. The paper's threshold
  `alpha_iu > mean(alpha_i) + std(alpha_i)` is applied. Overlapping memberships
  are retained in `interpretability_figure_data.json`, but only the strongest
  qualifying community can be displayed as one node color.
- `figure10_alpha_positive_heatmap`: directly visualizes the saved first-layer
  non-negative Ra-GConv community-membership matrix.

Important limitations:

- Training outputs save first R-pool scores only. The Figure 3.4 and Figure 3.5
  maps therefore use first-layer top-{top_rois} ROIs as a proxy for the nodes
  remaining after the second R-pool layer.
- The double-hemisphere drawing is a schematic layout using the real
  Desikan-Killiany ROI names; it is not a surface-coordinate rendering.
- Neurosynth masks and decoding scores are unavailable, so the paper's Figure 8
  cannot be reproduced from the current artifacts.
"""
    (output_dir / "README.md").write_text(text)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rois = atlas_rois(args.atlas)
    positions = schematic_positions(rois)
    main_root = current_or_legacy_run_root(args.experiment_root, "paper_like")

    data = {
        "roi_labels": rois,
        "figure5": figure5_glc_comparison(
            args.experiment_root, args.output_dir, positions, rois, args.fold, args.task, args.top_rois
        ),
    }
    means = task_mean_scores(main_root)
    data["figure7"] = figure7_task_saliency(
        args.output_dir, positions, rois, means, args.top_rois
    )
    data["figure8_proxy"] = figure8_proxy(args.output_dir, means, args.top_rois)

    fold_dir = best_fold(main_root)
    alpha = np.load(fold_dir / "community_membership_alpha_positive.npy")
    data["figure9"] = figure9_communities(
        args.output_dir, positions, alpha, rois, fold_dir.name
    )
    data["figure10"] = figure10_alpha_heatmap(args.output_dir, alpha, fold_dir.name)
    (args.output_dir / "interpretability_figure_data.json").write_text(
        json.dumps(data, indent=2, sort_keys=True)
    )
    write_notes(args.output_dir, fold_dir.name, args.top_rois)
    print(f"Wrote Section 3.5-style figures to {args.output_dir}")


if __name__ == "__main__":
    main()
