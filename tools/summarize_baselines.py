import argparse
import os
from glob import glob

import numpy as np


def parse_metric_file(path: str) -> np.ndarray:
    values = []
    if not os.path.exists(path):
        return np.array(values, dtype=np.float64)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            for p in parts:
                if p.startswith("value="):
                    try:
                        values.append(float(p.split("=", 1)[1]))
                    except ValueError:
                        pass
    return np.array(values, dtype=np.float64)


def summarize_algo_dataset(algo: str, dataset: str) -> dict:
    seed_dirs = sorted(glob(os.path.join("results", algo, dataset, "seed_*")))
    if not seed_dirs:
        return {}

    accs = []
    losses = []
    leaks = []
    for seed_dir in seed_dirs:
        acc = parse_metric_file(os.path.join(seed_dir, "centralized_accuracy.txt"))
        loss = parse_metric_file(os.path.join(seed_dir, "centralized_loss.txt"))
        leak = parse_metric_file(os.path.join(seed_dir, "privacy_leakage.txt"))
        if acc.size > 0:
            accs.append(acc[-1])
        if loss.size > 0:
            losses.append(np.mean(loss[-15:]))
        if leak.size > 0:
            leaks.append(np.mean(leak))

    def _mean_std(x):
        if len(x) == 0:
            return (np.nan, np.nan)
        return (float(np.mean(x)), float(np.std(x)))

    acc_m, acc_s = _mean_std(accs)
    loss_m, loss_s = _mean_std(losses)
    leak_m, leak_s = _mean_std(leaks)
    return {
        "algo": algo,
        "dataset": dataset,
        "num_seeds": len(seed_dirs),
        "accuracy_mean": acc_m,
        "accuracy_std": acc_s,
        "loss_last15_mean": loss_m,
        "loss_last15_std": loss_s,
        "leakage_mean": leak_m,
        "leakage_std": leak_s,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize CVB/DCS2 baseline results.")
    parser.add_argument("--algos", nargs="+", default=["cvb_fl", "dcs2_fl"])
    parser.add_argument("--datasets", nargs="+", default=["fashion", "cifar"])
    args = parser.parse_args()

    print("algo,dataset,num_seeds,acc_mean,acc_std,loss15_mean,loss15_std,leak_mean,leak_std")
    for algo in args.algos:
        for dataset in args.datasets:
            row = summarize_algo_dataset(algo, dataset)
            if not row:
                continue
            print(
                f"{row['algo']},{row['dataset']},{row['num_seeds']},"
                f"{row['accuracy_mean']:.6f},{row['accuracy_std']:.6f},"
                f"{row['loss_last15_mean']:.6f},{row['loss_last15_std']:.6f},"
                f"{row['leakage_mean']:.6f},{row['leakage_std']:.6f}"
            )


if __name__ == "__main__":
    main()
