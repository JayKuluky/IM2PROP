#!/usr/bin/env python3
"""
Visualize Test MAE and Training Curves Across All Experiment Runs.

Groups runs by (USE_PHASE_ATTENTION, USE_PHASE_RATIOS, USE_PHASE_FEAT) → 8 combos
× NUM_EPOCHS (30 / 50 / 80) = up to 24 logical groups.

Outputs
-------
results/fig1_test_mae_comparison.png  – grouped bar: 8 combos × epoch settings
results/fig2_learning_curves.png      – learning curve subplots per group
results/fig3_generalization_gap.png   – train-val generalization gap heatmap
results/summary_table.csv            – per-group aggregated metrics
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]   # IM2PROP/
RUNS_DIR  = ROOT / "runs"
WANDB_DIR = ROOT / "wandb"
RESULTS   = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

# ── Combo mapping: (USE_PHASE_RATIOS, USE_PHASE_ATTENTION, USE_PHASE_FEAT) → label ──
# R  A  F
# 0  0  0  Combo 1 – Baseline (RGB only)
# 0  0  1  Combo 2 – + Phase features
# 0  1  0  Combo 3 – + Attention
# 0  1  1  Combo 4 – + Attention + features
# 1  0  0  Combo 5 – + Phase ratios
# 1  0  1  Combo 6 – + Ratios + features
# 1  1  0  Combo 7 – + Ratios + attention
# 1  1  1  Combo 8 – Full IM2PROP
COMBO_MAP: dict[tuple[bool, bool, bool], str] = {
    (False, False, False): "Combo 1\nBaseline",
    (False, False, True):  "Combo 2\n+Phase feat",
    (False, True,  False): "Combo 3\n+Attention",
    (False, True,  True):  "Combo 4\n+Attn+feat",
    (True,  False, False): "Combo 5\n+Ratios",
    (True,  False, True):  "Combo 6\n+Ratio+feat",
    (True,  True,  False): "Combo 7\n+Ratio+attn",
    (True,  True,  True):  "Combo 8\nFull IM2PROP",
}
COMBO_ORDER = [
    "Combo 1\nBaseline", "Combo 2\n+Phase feat", "Combo 3\n+Attention",
    "Combo 4\n+Attn+feat", "Combo 5\n+Ratios", "Combo 6\n+Ratio+feat",
    "Combo 7\n+Ratio+attn", "Combo 8\nFull IM2PROP",
]


# ── 1. Collect per-run data ───────────────────────────────────────────────────
def load_runs() -> pd.DataFrame:
    records = []
    for config_path in sorted(RUNS_DIR.glob("*/config.json")):
        run_dir      = config_path.parent
        metrics_path = run_dir / "test_metrics.json"
        if not metrics_path.exists():
            continue

        with open(config_path) as f:
            cfg = json.load(f)
        with open(metrics_path) as f:
            met = json.load(f)

        # timestamp: run_YYYYMMDD_HHMMSS → YYYYMMDD_HHMMSS
        ts = run_dir.name[len("run_"):]

        records.append({
            "run_id":              run_dir.name,
            "timestamp":           ts,
            "NUM_EPOCHS":          int(cfg.get("NUM_EPOCHS", 30)),
            "USE_PHASE_ATTENTION": bool(cfg.get("USE_PHASE_ATTENTION", False)),
            "USE_PHASE_RATIOS":    bool(cfg.get("USE_PHASE_RATIOS",    False)),
            "USE_PHASE_FEAT":      bool(cfg.get("USE_PHASE_FEAT",      False)),
            "ENABLE_PATCHING":     bool(cfg.get("ENABLE_PATCHING",     False)),
            "test_mae":            float(met["mae"]),
            "test_mse":            float(met["mse"]),
            "test_rmse":           float(met["rmse"]),
        })

    df = pd.DataFrame(records)
    df["group_label"] = df.apply(
        lambda r: COMBO_MAP[(r.USE_PHASE_RATIOS, r.USE_PHASE_ATTENTION, r.USE_PHASE_FEAT)],
        axis=1,
    )
    return df


# ── 2. Parse output.log for per-epoch metrics ─────────────────────────────────
def parse_output_log(log_path: Path) -> dict:
    """
    Returns {epoch_num: {"train_loss": float, "val_loss": float, "val_mae": float}}
    Val metrics only appear on validation epochs (every 2 epochs).
    """
    epoch_re = re.compile(r"Epoch \[(\d+)/\d+\]")
    train_re = re.compile(r"Train Loss:\s*([\d.]+)")
    val_re   = re.compile(
        r"Val Loss:\s*([\d.]+)\s*\|\s*Val MAE:\s*([\d.]+)\s*\|\s*Val RMSE:\s*([\d.]+)"
    )

    data: dict = {}
    current = None

    for line in log_path.read_text(errors="ignore").splitlines():
        m = epoch_re.search(line)
        if m:
            current = int(m.group(1))
            data[current] = {}
            continue
        if current is None:
            continue
        m = train_re.search(line)
        if m:
            data[current]["train_loss"] = float(m.group(1))
            continue
        m = val_re.search(line)
        if m:
            data[current]["val_loss"] = float(m.group(1))
            data[current]["val_mae"]  = float(m.group(2))
            data[current]["val_rmse"] = float(m.group(3))

    return data


def load_training_curves(df: pd.DataFrame) -> dict:
    """Map run_id → parsed epoch dict from the matching wandb output.log."""
    curves = {}
    for _, row in df.iterrows():
        ts      = row["timestamp"]
        matches = list(WANDB_DIR.glob(f"run-{ts}-*/files/output.log"))
        if not matches:
            continue
        parsed = parse_output_log(matches[0])
        if parsed:
            curves[row["run_id"]] = parsed
    return curves


# ── 3. Aggregate curves per (group_label, NUM_EPOCHS) ────────────────────────
def aggregate_curves(df: pd.DataFrame, curves: dict) -> dict:
    """
    Returns {(group_label, num_epochs): {
        "train_epochs":  [[e,...], ...],   # per seed
        "train_loss":    [[l,...], ...],
        "val_epochs":    [[e,...], ...],
        "val_loss":      [[l,...], ...],
        "val_mae":       [[m,...], ...],
    }}
    """
    agg = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        rid = row["run_id"]
        if rid not in curves:
            continue
        ec  = curves[rid]
        key = (row["group_label"], int(row["NUM_EPOCHS"]))

        train_ep = sorted(e for e in ec if "train_loss" in ec[e])
        val_ep   = sorted(e for e in ec if "val_loss"   in ec[e])

        agg[key]["train_epochs"].append(train_ep)
        agg[key]["train_loss"].append([ec[e]["train_loss"] for e in train_ep])
        agg[key]["val_epochs"].append(val_ep)
        agg[key]["val_loss"].append([ec[e]["val_loss"] for e in val_ep])
        agg[key]["val_mae"].append([ec[e]["val_mae"]  for e in val_ep])

    return dict(agg)


# ── helper: mean ± std band from ragged lists ─────────────────────────────────
def _mean_std(seqs: list[list]) -> tuple[np.ndarray, np.ndarray, list]:
    """Truncate all sequences to the shortest, return mean, std, x-axis."""
    min_len = min(len(s) for s in seqs)
    arr     = np.array([s[:min_len] for s in seqs])
    return arr.mean(0), arr.std(0), list(range(1, min_len + 1))


def _val_mean_std(seqs: list[list], epoch_lists: list[list]):
    min_len = min(len(s) for s in seqs)
    arr     = np.array([s[:min_len] for s in seqs])
    x       = epoch_lists[0][:min_len]
    return arr.mean(0), arr.std(0), x


# ── Figure 1: Test MAE grouped bar chart ─────────────────────────────────────
def plot_test_mae(df: pd.DataFrame, save_path: Path):
    epoch_vals   = sorted(df["NUM_EPOCHS"].unique())
    group_labels = [g for g in COMBO_ORDER if g in df["group_label"].unique()]

    n_groups  = len(group_labels)
    n_epochs  = len(epoch_vals)
    bar_width = 0.75 / n_epochs
    x         = np.arange(n_groups)
    palette   = sns.color_palette("Set2", n_epochs)

    fig, ax = plt.subplots(figsize=(max(14, n_groups * 2.0), 7))

    # Find the true best (group, epoch) pair by minimum mean MAE
    best_key = (
        df.groupby(["group_label", "NUM_EPOCHS"])["test_mae"]
        .mean()
        .idxmin()
    )  # returns (group_label, num_epochs)

    # Collect all bars and metadata for floating labels (drawn after ylim is set)
    bar_records = []  # list of (bar_center_x, bar_top, mean_val, epoch, group_label, color)

    for i, ep in enumerate(epoch_vals):
        sub    = df[df["NUM_EPOCHS"] == ep]
        means  = [sub[sub["group_label"] == g]["test_mae"].mean() for g in group_labels]
        stds   = [sub[sub["group_label"] == g]["test_mae"].std()  for g in group_labels]
        offset = (i - n_epochs / 2 + 0.5) * bar_width
        bars   = ax.bar(
            x + offset, means, bar_width * 0.92,
            yerr=stds, capsize=3,
            label=f"{ep} epochs", color=palette[i], alpha=0.88,
            error_kw={"elinewidth": 1.2},
        )
        for bar, mean_val, std_val, gl in zip(bars, means, stds, group_labels):
            if not np.isnan(mean_val):
                bar_top = mean_val + (std_val if not np.isnan(std_val) else 0)
                bar_records.append((
                    bar.get_x() + bar.get_width() / 2,
                    bar_top,
                    mean_val,
                    ep,
                    gl,
                    palette[i],
                ))

    # Set ylim before placing floating labels
    ax.set_ylim(0, ax.get_ylim()[1] * 1.30)
    label_y = ax.get_ylim()[1] * 0.82   # fixed row for all MAE labels

    for bx, bar_top, mean_val, ep, gl, color in bar_records:
        is_best = (gl == best_key[0] and ep == best_key[1])
        # Hairline from bar top to label
        ax.plot([bx, bx], [bar_top, label_y * 0.97],
                color=color, lw=0.7, alpha=0.5, zorder=1)
        ax.text(
            bx, label_y,
            f"{mean_val:.3f}",
            ha="center", va="bottom", fontsize=8,
            color="crimson" if is_best else color,
            fontweight="bold" if is_best else "normal",
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=0, ha="center", fontsize=9)
    ax.set_ylabel("Test MAE (mean ± std across seeds)", fontsize=11)
    ax.set_title(
        "Test MAE by Configuration Group × Epoch Setting\n"
        "(Best value highlighted in bold red)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(title="NUM_EPOCHS", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Figure 2: Learning curves grid (groups × epoch settings) ─────────────────
def plot_learning_curves(
    df: pd.DataFrame, group_data: dict, save_path: Path
):
    group_labels = [g for g in COMBO_ORDER if g in df["group_label"].unique()]
    epoch_vals   = sorted(df["NUM_EPOCHS"].unique())

    n_rows = len(epoch_vals)
    n_cols = len(group_labels)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.2, n_rows * 3.0),
        squeeze=False,
    )

    for ri, ep in enumerate(epoch_vals):
        for ci, gl in enumerate(group_labels):
            ax  = axes[ri][ci]
            key = (gl, ep)

            sub_test = df[(df["group_label"] == gl) & (df["NUM_EPOCHS"] == ep)]["test_mae"]
            n_runs   = len(sub_test)

            has_curves = key in group_data and group_data[key].get("train_loss")

            if has_curves:
                gd = group_data[key]

                # Train loss
                t_mean, t_std, t_x = _mean_std(gd["train_loss"])
                ax.plot(t_x, t_mean, color="steelblue", lw=1.6, label="Train Loss")
                ax.fill_between(t_x, t_mean - t_std, t_mean + t_std,
                                alpha=0.20, color="steelblue")

                # Val loss
                v_mean, v_std, v_x = _val_mean_std(gd["val_loss"], gd["val_epochs"])
                ax.plot(v_x, v_mean, color="darkorange", lw=1.6, label="Val Loss")
                ax.fill_between(v_x, v_mean - v_std, v_mean + v_std,
                                alpha=0.20, color="darkorange")

                # Diagnosis annotation
                final_gap = float(v_mean[-1]) - float(t_mean[-1])
                if final_gap > 0.15:
                    diag, dcol = "Overfit?", "red"
                elif t_mean[-1] > 0.3 and v_mean[-1] > 0.3:
                    diag, dcol = "Underfit?", "purple"
                else:
                    diag, dcol = "OK", "green"
                ax.text(0.97, 0.97, diag, transform=ax.transAxes,
                        ha="right", va="top", fontsize=7, color=dcol,
                        fontweight="bold")
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="gray", fontsize=8)

            # Subplot title: test MAE summary
            if not sub_test.empty:
                mae_str = (f"MAE={sub_test.mean():.3f}"
                           + (f"±{sub_test.std():.3f}" if n_runs > 1 else "")
                           + f" (n={n_runs})")
            else:
                mae_str = "no test data"
            ax.set_title(mae_str, fontsize=7, pad=3)

            # Column header (combo name) on top row only
            if ri == 0:
                ax.text(0.5, 1.30, gl,
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold")

            # Row label (epoch) on leftmost column only
            if ci == 0:
                ax.set_ylabel(f"{ep} ep\nLoss", fontsize=8)
            else:
                ax.set_ylabel("")

            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.2)

    # Shared legend (top-right subplot)
    axes[0][-1].legend(fontsize=6.5, loc="upper right")

    fig.suptitle(
        "Learning Curves: Train vs Val Loss  (mean ± std across seeds)\n"
        "Annotation: OK=good fit | Overfit?=val>>train | Underfit?=both high",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Figure 3 helpers ──────────────────────────────────────────────────────────
def _build_gap_matrices(
    df: pd.DataFrame, group_data: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (gap_matrix, test_mae_matrix) indexed by group_label × NUM_EPOCHS."""
    group_labels = [g for g in COMBO_ORDER if g in df["group_label"].unique()]
    epoch_vals   = sorted(df["NUM_EPOCHS"].unique())

    gap_matrix      = pd.DataFrame(index=group_labels, columns=epoch_vals, dtype=float)
    test_mae_matrix = pd.DataFrame(index=group_labels, columns=epoch_vals, dtype=float)

    for gl in group_labels:
        for ep in epoch_vals:
            key = (gl, ep)
            sub = df[(df["group_label"] == gl) & (df["NUM_EPOCHS"] == ep)]
            if not sub.empty:
                test_mae_matrix.loc[gl, ep] = round(sub["test_mae"].mean(), 4)
            if key in group_data and group_data[key].get("train_loss"):
                gd          = group_data[key]
                final_train = np.mean([s[-1] for s in gd["train_loss"]])
                final_val   = np.mean([s[-1] for s in gd["val_loss"]])
                gap_matrix.loc[gl, ep] = round(float(final_val - final_train), 4)

    return gap_matrix, test_mae_matrix


# ── Figure 3a: Generalization gap heatmap ────────────────────────────────────
def plot_gap_heatmap(gap_matrix: pd.DataFrame, save_path: Path):
    gap_num = gap_matrix.astype(float)
    h       = max(6, len(gap_num) * 0.7 + 2)

    fig, ax = plt.subplots(figsize=(7, h))
    sns.heatmap(
        gap_num, ax=ax, annot=True, fmt=".3f",
        cmap=sns.diverging_palette(133, 10, as_cmap=True),
        center=0, linewidths=0.5,
        cbar_kws={"label": "Val Loss − Train Loss"},
    )

    # Red outline for overfitting risk cells (gap > 0.15)
    for i, gl in enumerate(gap_num.index):
        for j, ep in enumerate(gap_num.columns):
            val = gap_num.loc[gl, ep]
            if pd.notna(val) and val > 0.15:
                ax.add_patch(
                    mpatches.FancyBboxPatch(
                        (j, i), 1, 1,
                        boxstyle="round,pad=0.05",
                        linewidth=2, edgecolor="red", facecolor="none",
                        transform=ax.transData,
                    )
                )

    ax.set_title(
        "Generalization Gap  (Val Loss − Train Loss at final epoch)\n"
        "Red outline = gap > 0.15  →  overfitting risk",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("NUM_EPOCHS", fontsize=11)
    ax.set_ylabel("Config Group", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Figure 3b: Test MAE heatmap ───────────────────────────────────────────────
def plot_mae_heatmap(test_mae_matrix: pd.DataFrame, save_path: Path):
    mae_num = test_mae_matrix.astype(float)
    h       = max(6, len(mae_num) * 0.7 + 2)

    fig, ax = plt.subplots(figsize=(7, h))
    sns.heatmap(
        mae_num, ax=ax, annot=True, fmt=".3f",
        cmap="YlOrRd", linewidths=0.5,
        cbar_kws={"label": "Mean Test MAE"},
    )

    # Bold outline on the best (minimum) cell
    best_idx = np.unravel_index(
        np.nanargmin(mae_num.values), mae_num.shape
    )
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (best_idx[1], best_idx[0]), 1, 1,
            boxstyle="round,pad=0.05",
            linewidth=2.5, edgecolor="black", facecolor="none",
            transform=ax.transData,
        )
    )

    ax.set_title(
        "Mean Test MAE per Group × Epoch\nBlack outline = best (lowest) MAE",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("NUM_EPOCHS", fontsize=11)
    ax.set_ylabel("Config Group", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("IM2PROP — MAE Visualization & Overfitting Analysis")
    print("=" * 60)

    print("\n[1/5] Loading run data from runs/*/config.json + test_metrics.json ...")
    df = load_runs()
    print(f"      {len(df)} valid runs | {df['group_label'].nunique()} config groups "
          f"| epoch settings: {sorted(df['NUM_EPOCHS'].unique())}")

    print("\n[2/5] Parsing training curves from wandb output.log files ...")
    curves = load_training_curves(df)
    print(f"      Loaded curves for {len(curves)}/{len(df)} runs")

    print("\n[3/5] Aggregating curves per (group, epoch) ...")
    group_data = aggregate_curves(df, curves)
    print(f"      {len(group_data)} (group × epoch) combinations with curve data")

    # Summary CSV
    print("\n[4/5] Saving summary_table.csv ...")
    summary = (
        df.groupby(["group_label", "NUM_EPOCHS"])
        .agg(
            n_runs        =("test_mae",  "count"),
            mean_test_mae =("test_mae",  "mean"),
            std_test_mae  =("test_mae",  "std"),
            min_test_mae  =("test_mae",  "min"),
            mean_test_rmse=("test_rmse", "mean"),
        )
        .round(4)
        .reset_index()
    )
    # Sort by combo order
    combo_order_map = {g: i for i, g in enumerate(COMBO_ORDER)}
    summary["_order"] = summary["group_label"].map(combo_order_map)
    summary = summary.sort_values(["_order", "NUM_EPOCHS"]).drop(columns="_order")
    summary.to_csv(RESULTS / "summary_table.csv", index=False)
    print(summary.to_string(index=False))

    print("\n[5/5] Generating figures ...")
    plot_test_mae(df, RESULTS / "fig1_test_mae_comparison.png")
    plot_learning_curves(df, group_data, RESULTS / "fig2_learning_curves.png")
    gap_m, mae_m = _build_gap_matrices(df, group_data)
    plot_gap_heatmap(gap_m,  RESULTS / "fig3a_generalization_gap.png")
    plot_mae_heatmap(mae_m,  RESULTS / "fig3b_test_mae_heatmap.png")

    print(f"\nAll outputs saved to: {RESULTS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
