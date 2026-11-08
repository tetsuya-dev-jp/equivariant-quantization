#!/usr/bin/env python3
"""Plot curated pilot diagnostics for the quantized model comparison."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = {
    "qwen": "Qwen2.5-1.5B",
    "smol": "SmolLM2-1.7B",
}
CONDITIONS = ["fp", "w8", "w4", "w3"]
WEIGHT_CONDITIONS = ["w8", "w4", "w3"]
CATEGORIES = [
    ("jointly_correct_pairs", "Jointly correct"),
    ("consistently_wrong_pairs", "Consistently wrong"),
    ("inconsistent_pairs", "Inconsistent"),
]
METRICS = [
    ("delta_accuracy_macro", "accuracy_macro", "Accuracy", "#0072B2", "o"),
    ("delta_consistency_macro", "consistency_macro", "Consistency", "#D55E00", "s"),
]
FIGURE_SIZE = (7.0, 3.0)


def load_analysis(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_all(rows):
    return [
        row
        for row in rows
        if row["split"] == "all"
        and row["task_family"] == "all"
        and row["transform"] == "all"
    ]


def collect_figure1(analysis, label):
    intervals = select_all(analysis["bootstrap_intervals"])
    selected = {}
    for metric, _, _, _, _ in METRICS:
        for condition in WEIGHT_CONDITIONS:
            matches = [
                row
                for row in intervals
                if row["reference"] == "fp"
                and row["condition"] == condition
                and row["metric"] == metric
            ]
            assert len(matches) == 1, (
                f"{label}: expected one interval for {condition}/{metric}, "
                f"found {len(matches)}"
            )
            row = matches[0]
            assert row["ci_low"] <= row["estimate"] <= row["ci_high"]
            selected[(condition, metric)] = {
                "estimate": row["estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "n_boot": row["n_boot"],
                "seed": row["seed"],
            }
    return selected


def collect_figure2(analysis, label):
    summaries = select_all(analysis["summary"])
    selected = []
    for condition in CONDITIONS:
        matches = [row for row in summaries if row["condition"] == condition]
        assert len(matches) == 1, (
            f"{label}: expected one all/all/all summary for {condition}, "
            f"found {len(matches)}"
        )
        row = matches[0]
        jointly_correct = row["jointly_correct_pairs"]
        consistently_wrong = row["consistent_but_wrong_pairs"]
        inconsistent = row["n_pairs"] - jointly_correct - consistently_wrong
        assert min(jointly_correct, consistently_wrong, inconsistent) >= 0
        assert row["n_pairs"] == jointly_correct + consistently_wrong + inconsistent
        selected.append(
            {
                "condition": condition,
                "n_pairs": row["n_pairs"],
                "jointly_correct_pairs": jointly_correct,
                "consistently_wrong_pairs": consistently_wrong,
                "inconsistent_pairs": inconsistent,
            }
        )
    return selected


def style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_differences(ax, data, panel, model_name):
    positions = {condition: len(WEIGHT_CONDITIONS) - 1 - i for i, condition in enumerate(WEIGHT_CONDITIONS)}
    offsets = {METRICS[0][0]: -0.11, METRICS[1][0]: 0.11}
    for metric, _, label, color, marker in METRICS:
        values = [100.0 * data[(condition, metric)]["estimate"] for condition in WEIGHT_CONDITIONS]
        lower = [
            100.0 * (data[(condition, metric)]["estimate"] - data[(condition, metric)]["ci_low"])
            for condition in WEIGHT_CONDITIONS
        ]
        upper = [
            100.0 * (data[(condition, metric)]["ci_high"] - data[(condition, metric)]["estimate"])
            for condition in WEIGHT_CONDITIONS
        ]
        y = [positions[condition] + offsets[metric] for condition in WEIGHT_CONDITIONS]
        ax.errorbar(
            values,
            y,
            xerr=[lower, upper],
            fmt=marker,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=5.0,
            elinewidth=1.0,
            capsize=2.5,
            label=label,
            zorder=3,
        )
    ax.axvline(0.0, color="#333333", linewidth=1.0, zorder=1)
    ax.set_yticks([positions[condition] for condition in WEIGHT_CONDITIONS])
    ax.set_yticklabels([condition.upper() for condition in WEIGHT_CONDITIONS])
    ax.set_ylim(-0.55, len(WEIGHT_CONDITIONS) - 0.45)
    ax.set_xlabel("Difference from BF16 reference (percentage points)")
    ax.set_ylabel("Quantization condition")
    style_axes(ax)
    ax.text(
        0.0,
        1.04,
        f"{panel}  {model_name}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#222222",
    )


def plot_stacks(ax, data, panel, model_name):
    x = list(range(len(CONDITIONS)))
    bottoms = [0] * len(CONDITIONS)
    colors = ["#3B6EA5", "#D08A4C", "#D9D9D9"]
    hatches = ["", "///", ".."]
    for index, (key, label) in enumerate(CATEGORIES):
        values = [row[key] for row in data]
        ax.bar(
            x,
            values,
            bottom=bottoms,
            width=0.62,
            color=colors[index],
            edgecolor="#333333",
            linewidth=0.6,
            hatch=hatches[index],
            label=label,
            zorder=3,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_xticks(x)
    ax.set_xticklabels([condition.upper() for condition in CONDITIONS])
    ax.set_xlabel("Quantization condition")
    ax.set_ylabel("Original–variant pairs (raw count; n = 512)")
    ax.set_ylim(0, 512)
    ax.set_yticks([0, 128, 256, 384, 512])
    style_axes(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax.text(
        0.0,
        1.04,
        f"{panel}  {model_name}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#222222",
    )


def save_figure(fig, output_dir, stem):
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, format="pdf", dpi=300)
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def build_figure1(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, sharey=False)
    for ax, key in zip(axes, ("qwen", "smol")):
        plot_differences(ax, results[key]["figure1"], "A" if key == "qwen" else "B", MODELS[key])
    all_limits = []
    for key in ("qwen", "smol"):
        for entry in results[key]["figure1"].values():
            all_limits.extend([100.0 * entry["ci_low"], 100.0 * entry["ci_high"]])
    lower = min(all_limits + [0.0]) - 2.0
    upper = max(all_limits + [0.0]) + 2.0
    for ax in axes:
        ax.set_xlim(lower, upper)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        fontsize=8,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.21, top=0.79, wspace=0.18)
    return save_figure(fig, output_dir, "pilot_delta_accuracy_consistency")


def build_figure2(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, sharey=True)
    for ax, key in zip(axes, ("qwen", "smol")):
        plot_stacks(ax, results[key]["figure2"], "A" if key == "qwen" else "B", MODELS[key])
    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        fontsize=8,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.21, top=0.79, wspace=0.12)
    return save_figure(fig, output_dir, "pilot_pair_outcomes")


def write_plot_data(results, input_paths, output_dir):
    payload = {
        "provenance": {
            "inputs": {key: str(path) for key, path in input_paths.items()},
            "selection": {
                "split": "all",
                "task_family": "all",
                "transform": "all",
                "reference": "fp",
            },
            "units": {
                "figure1": "percentage points relative to BF16 reference",
                "figure2": "original-variant pairs (count)",
            },
            "figure2_scope": "raw pair counts, n = 512 per model; not base-macro rates",
            "bootstrap": {
                key: {
                    "estimator": results[key]["analysis"]["bootstrap"]["estimator"],
                    "samples": results[key]["analysis"]["bootstrap"]["samples"],
                    "seed": results[key]["analysis"]["bootstrap"]["seed"],
                    "unit": results[key]["analysis"]["bootstrap"]["unit"],
                }
                for key in ("qwen", "smol")
            },
        },
        "figure1": {
            "metric_labels": {
                METRICS[0][0]: METRICS[0][2],
                METRICS[1][0]: METRICS[1][2],
            },
            "panels": {
                key: {
                    "model": MODELS[key],
                    "differences_pp": {
                        condition: {
                            metric: {
                                "estimate": 100.0 * results[key]["figure1"][(condition, metric)]["estimate"],
                                "ci_low": 100.0 * results[key]["figure1"][(condition, metric)]["ci_low"],
                                "ci_high": 100.0 * results[key]["figure1"][(condition, metric)]["ci_high"],
                                "n_boot": results[key]["figure1"][(condition, metric)]["n_boot"],
                                "seed": results[key]["figure1"][(condition, metric)]["seed"],
                            }
                            for metric, _, _, _, _ in METRICS
                        }
                        for condition in WEIGHT_CONDITIONS
                    },
                }
                for key in ("qwen", "smol")
            },
        },
        "figure2": {
            "categories": {
                key: label for key, label in CATEGORIES
            },
            "panels": {
                key: {
                    "model": MODELS[key],
                    "counts": {
                        row["condition"]: {
                            "n_pairs": row["n_pairs"],
                            "jointly_correct_pairs": row["jointly_correct_pairs"],
                            "consistently_wrong_pairs": row["consistently_wrong_pairs"],
                            "inconsistent_pairs": row["inconsistent_pairs"],
                        }
                        for row in results[key]["figure2"]
                    },
                }
                for key in ("qwen", "smol")
            },
        },
    }
    path = output_dir / "plot-data.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen",
        type=Path,
        default=Path("results/pilot/qwen/analysis/analysis.json"),
    )
    parser.add_argument(
        "--smol",
        type=Path,
        default=Path("results/pilot/smol/analysis/analysis.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/pilot"),
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "hatch.linewidth": 0.6,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    input_paths = {"qwen": args.qwen, "smol": args.smol}
    results = {}
    for key, path in input_paths.items():
        analysis = load_analysis(path)
        results[key] = {
            "analysis": analysis,
            "figure1": collect_figure1(analysis, key),
            "figure2": collect_figure2(analysis, key),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig1_pdf, fig1_png = build_figure1(results, args.output_dir)
    fig2_pdf, fig2_png = build_figure2(results, args.output_dir)
    data_path = write_plot_data(results, input_paths, args.output_dir)

    for path in (fig1_pdf, fig1_png, fig2_pdf, fig2_png, data_path):
        print(path)


if __name__ == "__main__":
    main()
