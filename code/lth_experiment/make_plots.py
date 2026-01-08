import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ACTIVATIONS = ["relu", "tanh", "sigmoid"]
SPARSITIES = [0.5, 0.8, 0.9]
METHODS = ["prune_mag", "prune_grasp"]
RESULTS_ROOT = Path("results")


def load_last_val_acc(log_path):
    if not log_path.exists():
        return None
    df = pd.read_csv(log_path)
    return df["val_acc"].iloc[-1]


def load_weight_distance(stability_path):
    if not stability_path.exists():
        return None
    with open(stability_path) as f:
        stab = json.load(f)
    distances = stab["weight_distance"]
    return np.mean(list(distances.values()))


def load_mask_sparsity(stability_path):
    if not stability_path.exists():
        return None
    with open(stability_path) as f:
        stab = json.load(f)
    return stab["mask_info"]["sparsity"]


def plot_accuracy_vs_sparsity():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for i, act in enumerate(ACTIVATIONS):
        ax = axes[i]
        for method in METHODS:
            vals = []
            for s in SPARSITIES:
                log_path = RESULTS_ROOT / act / f"{method}_{s}" / "metrics.csv"
                vals.append(load_last_val_acc(log_path))
            ax.plot(SPARSITIES, vals, marker="o", label=method)

        ax.set_title(f"{act} — Accuracy vs Sparsity")
        ax.set_xlabel("Sparsity")
        if i == 0:
            ax.set_ylabel("Validation Accuracy")
        ax.legend()

    plt.tight_layout()
    plt.savefig("plot_accuracy_vs_sparsity.png")
    plt.close()


def plot_stability_vs_sparsity():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for i, act in enumerate(ACTIVATIONS):
        ax = axes[i]
        for method in METHODS:
            vals = []
            for s in SPARSITIES:
                stab_path = RESULTS_ROOT / act / f"{method}_{s}" / "stability.json"
                vals.append(load_weight_distance(stab_path))
            ax.plot(SPARSITIES, vals, marker="o", label=method)

        ax.set_title(f"{act} — Stability vs Sparsity")
        ax.set_xlabel("Sparsity")
        if i == 0:
            ax.set_ylabel("Weight Distance")
        ax.legend()

    plt.tight_layout()
    plt.savefig("plot_stability_vs_sparsity.png")
    plt.close()


def plot_mask_sparsity():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    # Distinct styles for overlapping lines
    style_map = {
        "prune_mag": {"marker": "o", "linestyle": "-",  "color": "C0"},
        "prune_grasp": {"marker": "s", "linestyle": "--", "color": "C1"},
    }

    for i, act in enumerate(ACTIVATIONS):
        ax = axes[i]

        for method in METHODS:
            vals = []
            for s in SPARSITIES:
                stab_path = RESULTS_ROOT / act / f"{method}_{s}" / "stability.json"
                vals.append(load_mask_sparsity(stab_path))

            style = style_map[method]
            ax.plot(
                SPARSITIES,
                vals,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                label=method,
            )

        ax.set_title(f"{act} — Mask Sparsity")
        ax.set_xlabel("Target Sparsity")
        if i == 0:
            ax.set_ylabel("Actual Mask Sparsity")
        ax.legend()

    plt.tight_layout()
    plt.savefig("plot_mask_sparsity.png")
    plt.close()


def plot_training_curves():
    for act in ACTIVATIONS:
        plt.figure(figsize=(8, 5))

        # Dense
        dense_log = RESULTS_ROOT / act / "dense" / "metrics.csv"
        if dense_log.exists():
            df = pd.read_csv(dense_log)
            plt.plot(df["train_acc"], label="Dense")

        # Pruned (use sparsity 0.8 as representative)
        for method in METHODS:
            log_path = RESULTS_ROOT / act / f"{method}_0.8" / "metrics.csv"
            if log_path.exists():
                df = pd.read_csv(log_path)
                plt.plot(df["train_acc"], label=f"{method} (0.8)")

        plt.title(f"Training Curves — {act}")
        plt.xlabel("Epoch")
        plt.ylabel("Train Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plot_training_curves_{act}.png")
        plt.close()


def main():
    print("Now Creating Plots")
    plot_accuracy_vs_sparsity()
    plot_stability_vs_sparsity()
    plot_mask_sparsity()
    plot_training_curves()
    print("All plots saved")


if __name__ == "__main__":
    main()
