import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime


VAL_RE = re.compile(
    r"Epoch:\s+(?P<epoch>\d+),.*?"
    r"Val Loss:\s+(?P<val_loss>[0-9.eE+-]+),\s+"
    r"Val Acc:\s+(?P<val_acc>[0-9.eE+-]+),\s+"
    r"Val Balanced Acc:\s+(?P<val_bacc>[0-9.eE+-]+)"
)
TEST_RE = re.compile(
    r"Test Acc:\s+(?P<test_acc>[0-9.eE+-]+),\s+"
    r"Test Balanced Acc:\s+(?P<test_bacc>[0-9.eE+-]+),\s+"
    r"Test Loss:\s+(?P<test_loss>[0-9.eE+-]+)"
)


def paper33_configs():
    lambda2_values = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    lambda1_values = [0.0, 0.05, 0.1, 0.2, 0.5]

    configs = []
    for value in lambda2_values:
        configs.append(("lambda2_l1_0", 0.0, value))
    for value in lambda2_values:
        configs.append(("lambda2_l1_0p1", 0.1, value))
    for value in lambda1_values:
        configs.append(("lambda1_l2_0p1", value, 0.1))
    return configs


def run_command(command, stdout_path, env, progress_every):
    with open(stdout_path, "w", encoding="utf-8") as stdout_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in process.stdout:
            stdout_file.write(line)
            if progress_every > 0:
                val_match = VAL_RE.search(line)
                test_match = TEST_RE.search(line)
                if val_match and int(val_match.group("epoch")) % progress_every == 0:
                    print(line, end="")
                elif test_match:
                    print(line, end="")
        return process.wait()


def parse_stdout(stdout_path):
    best_val_acc = None
    best_val_acc_epoch = None
    best_val_bacc = None
    best_val_bacc_epoch = None
    best_val_loss = None
    best_val_loss_epoch = None
    final_test = {}

    with open(stdout_path, "r", encoding="utf-8") as handle:
        for line in handle:
            val_match = VAL_RE.search(line)
            if val_match:
                epoch = int(val_match.group("epoch"))
                val_loss = float(val_match.group("val_loss"))
                val_acc = float(val_match.group("val_acc"))
                val_bacc = float(val_match.group("val_bacc"))

                if best_val_acc is None or val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_acc_epoch = epoch
                if best_val_bacc is None or val_bacc > best_val_bacc:
                    best_val_bacc = val_bacc
                    best_val_bacc_epoch = epoch
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_loss_epoch = epoch

            test_match = TEST_RE.search(line)
            if test_match:
                final_test = {
                    "test_acc": float(test_match.group("test_acc")),
                    "test_bacc": float(test_match.group("test_bacc")),
                    "test_loss": float(test_match.group("test_loss")),
                }

    return {
        "best_val_acc": best_val_acc,
        "best_val_acc_epoch": best_val_acc_epoch,
        "best_val_bacc": best_val_bacc,
        "best_val_bacc_epoch": best_val_bacc_epoch,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        **final_test,
    }


def write_results(results, output_dir):
    json_path = os.path.join(output_dir, "summary.json")
    csv_path = os.path.join(output_dir, "summary.csv")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    fieldnames = [
        "stage",
        "fold",
        "paper_lambda1_tpk",
        "paper_lambda2_glc",
        "run_id",
        "status",
        "returncode",
        "best_val_acc",
        "best_val_acc_epoch",
        "best_val_bacc",
        "best_val_bacc_epoch",
        "best_val_loss",
        "best_val_loss_epoch",
        "test_acc",
        "test_bacc",
        "test_loss",
        "stdout",
        "log_path",
        "save_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batchSize", type=int, default=200)
    parser.add_argument("--dataroot", default="data/ABIDE_pcp/cpac/filt_noglobal_qc_fd020_perc20")
    parser.add_argument("--processed_file", default="data_pcorr_pos_top10pct.pt")
    parser.add_argument("--edge_source", choices=["pcorr", "corr"], default="pcorr")
    parser.add_argument("--edge_top_percent", type=float, default=0.10)
    parser.add_argument("--positive_edges_only", action="store_true", default=True)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--stepsize", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--weightdecay", type=float, default=0.005)
    parser.add_argument("--best_metric", choices=["loss", "acc", "balanced_acc"], default="acc")
    parser.add_argument("--dim1", type=int, default=32)
    parser.add_argument("--dim2", type=int, default=32)
    parser.add_argument("--fc_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        "experiments", f"paper33_hparam_qc_pcorr_pos_top10_{timestamp}"
    )
    runs_dir = os.path.join(output_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    results = []
    cache = {}
    configs = paper33_configs()

    for stage, paper_lambda1, paper_lambda2 in configs:
        for fold in args.folds:
            key = (fold, paper_lambda1, paper_lambda2)
            if key in cache:
                cached = dict(cache[key])
                cached["stage"] = stage
                results.append(cached)
                write_results(results, output_dir)
                continue

            run_id = (
                f"fold{fold}_pl1_{paper_lambda1:g}_pl2_{paper_lambda2:g}"
                .replace(".", "p")
            )
            run_dir = os.path.join(runs_dir, run_id)
            stdout_path = os.path.join(run_dir, "stdout.log")
            log_path = os.path.join(run_dir, "tb")
            save_path = os.path.join(run_dir, "model")
            os.makedirs(run_dir, exist_ok=True)

            command = [
                sys.executable,
                "-u",
                "03-main.py",
                "--dataroot",
                args.dataroot,
                "--processed_file",
                args.processed_file,
                "--edge_source",
                args.edge_source,
                "--edge_top_percent",
                str(args.edge_top_percent),
                "--lr",
                str(args.lr),
                "--stepsize",
                str(args.stepsize),
                "--gamma",
                str(args.gamma),
                "--weightdecay",
                str(args.weightdecay),
                "--best_metric",
                args.best_metric,
                "--fold",
                str(fold),
                "--n_epochs",
                str(args.epochs),
                "--batchSize",
                str(args.batchSize),
                "--lamb3",
                str(paper_lambda1),
                "--lamb4",
                str(paper_lambda1),
                "--lamb5",
                str(paper_lambda2),
                "--dim1",
                str(args.dim1),
                "--dim2",
                str(args.dim2),
                "--fc_dim",
                str(args.fc_dim),
                "--dropout",
                str(args.dropout),
                "--log_path",
                log_path,
                "--save_path",
                save_path,
            ]
            if args.positive_edges_only:
                command.append("--positive_edges_only")

            result = {
                "stage": stage,
                "fold": fold,
                "paper_lambda1_tpk": paper_lambda1,
                "paper_lambda2_glc": paper_lambda2,
                "run_id": run_id,
                "stdout": stdout_path,
                "log_path": log_path,
                "save_path": save_path,
            }

            print("\n" + "=" * 80)
            print(f"Running {stage} fold={fold} lambda1={paper_lambda1:g} lambda2={paper_lambda2:g}")
            print(" ".join(command))
            if args.dry_run:
                result.update({"status": "dry_run", "returncode": None})
            else:
                start = time.time()
                returncode = run_command(command, stdout_path, env, args.progress_every)
                result.update(parse_stdout(stdout_path))
                result.update({
                    "status": "ok" if returncode == 0 else "failed",
                    "returncode": returncode,
                    "elapsed_sec": round(time.time() - start, 3),
                })
            results.append(result)
            cache[key] = dict(result)
            write_results(results, output_dir)

            if result.get("status") == "failed":
                raise SystemExit(f"Run failed: {run_id}, see {stdout_path}")

    print("\nSummary written to:")
    print(os.path.join(output_dir, "summary.json"))
    print(os.path.join(output_dir, "summary.csv"))


if __name__ == "__main__":
    main()
