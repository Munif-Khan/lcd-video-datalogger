"""
plot.py
-------
Generate Voltage-vs-Time and Current-vs-Time plots from one or more Excel
output files produced by extract.py.

Usage
-----
  # Single file
  python plot.py --inputs readings.xlsx

  # Two files compared side-by-side and overlaid
  python plot.py --inputs run1.xlsx run2.xlsx --labels "Run 1" "Run 2"

  # Add cluster reference lines
  python plot.py --inputs run1.xlsx --vol_clusters 818 1078 --cur_clusters 241 454

  # Restrict the Y-axis range
  python plot.py --inputs run1.xlsx --vol_min 600 --vol_max 1300 --cur_min 100 --cur_max 600
"""

import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
CLUSTER_COLORS_V = ["#1565C0", "#4527A0", "#1B5E20", "#4A148C"]
CLUSTER_COLORS_C = ["#BF360C", "#880E4F", "#1B5E20", "#4A148C"]


def load(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    for col in ("frame", "voltage", "current", "time"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {path}")
    return df


def add_cluster_bands(ax, centers, band_hw, colors, unit):
    for i, (c, col) in enumerate(zip(centers, colors)):
        ax.axhspan(c - band_hw, c + band_hw, alpha=0.13, color=col)
        ax.axhline(c, color=col, lw=1.4, ls="--", alpha=0.75,
                   label=f"~{c:.0f} {unit} centre")
        ax.annotate(
            f"{c:.0f} {unit}",
            xy=(ax.get_xlim()[0], c + band_hw * 0.15),
            color=col, fontsize=9, fontweight="bold",
        )


def plot(args):
    dfs = [load(p) for p in args.inputs]
    labels = args.labels if args.labels else [f"File {i+1}" for i in range(len(dfs))]

    n = len(dfs)
    # ── Per-file subplots ────────────────────────────────────────────────────
    for metric, ylabel, ymin, ymax, clusters, cluster_cols in [
        ("voltage", "Voltage (V)", args.vol_min, args.vol_max,
         args.vol_clusters, CLUSTER_COLORS_V),
        ("current", "Current (A)", args.cur_min, args.cur_max,
         args.cur_clusters, CLUSTER_COLORS_C),
    ]:
        fig, axes = plt.subplots(n, 1, figsize=(14, 5 * n), sharex=False)
        if n == 1:
            axes = [axes]
        fig.suptitle(f"{ylabel} vs Time", fontsize=16, fontweight="bold", y=0.99)

        for ax, df, label, color in zip(axes, dfs, labels, PALETTE):
            ax.plot(df["time"], df[metric], color=color, lw=0.8, alpha=0.85, label=label)
            if clusters:
                add_cluster_bands(ax, clusters, band_hw=50,
                                  colors=cluster_cols, unit=ylabel.split()[0])
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.set_title(label, fontsize=12)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out = args.output_dir + f"/{metric}_vs_time.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")

    # ── Overlay plot (all files on same axes) ────────────────────────────────
    if n > 1:
        fig, (ax_v, ax_c) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle("Combined View – All Datasets", fontsize=16, fontweight="bold", y=0.99)

        for df, label, color in zip(dfs, labels, PALETTE):
            ax_v.plot(df["time"], df["voltage"], lw=0.7, alpha=0.75, color=color, label=label)
            ax_c.plot(df["time"], df["current"], lw=0.7, alpha=0.75, color=color, label=label)

        for ax, ylabel, ymin, ymax, clusters, cluster_cols in [
            (ax_v, "Voltage (V)", args.vol_min, args.vol_max, args.vol_clusters, CLUSTER_COLORS_V),
            (ax_c, "Current (A)", args.cur_min, args.cur_max, args.cur_clusters, CLUSTER_COLORS_C),
        ]:
            if clusters:
                add_cluster_bands(ax, clusters, band_hw=50,
                                  colors=cluster_cols, unit=ylabel.split()[0])
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3)

        ax_v.set_title("Voltage vs Time (Overlay)", fontsize=12)
        ax_c.set_title("Current vs Time (Overlay)", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out = args.output_dir + "/combined_vs_time.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Plot extraction results from one or more Excel files.")
    p.add_argument("--inputs",  nargs="+", required=True, help="Excel file(s) to plot")
    p.add_argument("--labels",  nargs="*", default=None,  help="Legend labels for each file")
    p.add_argument("--output_dir", default=".",            help="Directory to save PNG files (default: current dir)")
    p.add_argument("--vol_min",  type=float, default=None, help="Y-axis min for voltage")
    p.add_argument("--vol_max",  type=float, default=None, help="Y-axis max for voltage")
    p.add_argument("--cur_min",  type=float, default=None, help="Y-axis min for current")
    p.add_argument("--cur_max",  type=float, default=None, help="Y-axis max for current")
    p.add_argument("--vol_clusters", type=float, nargs="*", default=None,
                   help="Voltage cluster centre values to draw reference lines")
    p.add_argument("--cur_clusters", type=float, nargs="*", default=None,
                   help="Current cluster centre values to draw reference lines")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Defaults for ylim if not provided
    if args.vol_min is None: args.vol_min = 0
    if args.vol_max is None: args.vol_max = 9999
    if args.cur_min is None: args.cur_min = 0
    if args.cur_max is None: args.cur_max = 9999
    plot(args)
