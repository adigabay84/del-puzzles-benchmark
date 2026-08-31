"""
Compute per-quadrant accuracy for the main-analysis figure.

Reads post-processed result CSVs for each puzzle/model combination, grouped
into four quadrants (Q1-Q4) along two axes: narrative type (classic vs. new)
and inference type (symmetric vs. asymmetric). For each model, computes mean
accuracy per quadrant from the "verdict" column (PASSED/FAILED) and renders
a 2x2 heatmap panel per model, color-coded by reasoning strategy. Each cell
also reports the population variance (1/m)*sum((x_i - mean)^2) of the
per-instance PASSED/FAILED (1/0) outcomes pooled across every instance in
that quadrant cell, shown as "accuracy% (+-variance)".

Outputs:
    Graphs/combined_models_accuracy.png
"""


import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from src.analysis.quadrant_files import QUARTERS


VERDICT_COLUMN = "verdict"
OUTPUT_DIR = "Graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PUZZLE_MAPPING = {
    "blue_eyed_islanders": "Blue-Eyed Islanders",
    "muddy_children": "Muddy Children",
    "wise_men": "Wise Men",
    "health_screening": "Health Screening",
    "olympic_games": "Olympic Games",
    "safety_inspection": "Safety Inspection",
    "singing_contest": "Singing Contest",
}

STRATEGY_MAPPING = {
    "Present similar capabilities": {
        "models": ["qwen", "gemini"],
        "color": "#6db8a0",       # muted sage-teal
    },
    "Benefit from classic narratives": {
        "models": ["nano"],
        "color": "#8b96cc",       # muted lavender
    },
    "Benefit from new narratives": {
    "models": ["gpt-5"],
    "color": "#f5a623",      # clearer warm orange
    "exclude": ["nano"],
    },
}

def _format_model_name(model: str) -> str:
    """Fix display capitalization: gpt -> GPT, pro -> Pro (whole-word, case-insensitive)."""
    model = re.sub(r"\bgpt\b", "GPT", model, flags=re.IGNORECASE)
    model = re.sub(r"\bpro\b", "Pro", model, flags=re.IGNORECASE)
    return model


def collect_accuracies() -> dict[str, dict[str, dict[str, float]]]:
    """
    Read every CSV listed in QUARTERS and compute per-puzzle accuracy from
    the verdict column (PASSED -> correct, FAILED -> incorrect).

    Returns {model_name: {quarter: {puzzle_name: accuracy}}}.
    Files without a verdict column are skipped; unreadable files produce a warning.
    """
    results: dict[str, dict[str, dict[str, float]]] = {}

    for quarter, paths in QUARTERS.items():
        for path in paths:
            try:
                df = pd.read_csv(path)
                if VERDICT_COLUMN not in df.columns:
                    continue

                raw_puzzle = os.path.basename(os.path.dirname(path)) or "unknown"
                puzzle_name = PUZZLE_MAPPING.get(raw_puzzle, raw_puzzle)

                if "model" in df.columns and "version" in df.columns:
                    for (model, version), sub_df in df.groupby(["model", "version"]):
                        model_name = f"{str(model).strip()} {str(version).strip()}"
                        acc = _verdict_accuracy(sub_df)
                        results.setdefault(model_name, {}).setdefault(quarter, {})[puzzle_name] = acc
                else:
                    model_name = os.path.splitext(os.path.basename(path))[0]
                    acc = _verdict_accuracy(df)
                    results.setdefault(model_name, {}).setdefault(quarter, {})[puzzle_name] = acc
            except Exception as exc:
                print(f"[warn] {quarter} / {path}: {exc}")
    return results


def _verdict_accuracy(df: pd.DataFrame) -> float:
    """Fraction of rows whose verdict is PASSED (FAILED counts as incorrect)."""
    col = df[VERDICT_COLUMN].str.strip().str.upper().map({"PASSED": True, "FAILED": False})
    return col.mean()


def collect_instance_outcomes() -> dict[str, dict[str, list[int]]]:
    """
    Read every CSV listed in QUARTERS and collect the raw per-instance
    PASSED/FAILED (1/0) outcomes, pooled across all puzzles within each
    quadrant (unlike collect_accuracies, this does not average per puzzle
    first, so quadrants with more instances contribute more data points).

    Returns {model_name: {quarter: [1, 0, 1, ...]}}.
    """
    results: dict[str, dict[str, list[int]]] = {}

    for quarter, paths in QUARTERS.items():
        for path in paths:
            try:
                df = pd.read_csv(path)
                if VERDICT_COLUMN not in df.columns:
                    continue

                if "model" in df.columns and "version" in df.columns:
                    for (model, version), sub_df in df.groupby(["model", "version"]):
                        model_name = f"{str(model).strip()} {str(version).strip()}"
                        outcomes = _verdict_outcomes(sub_df)
                        results.setdefault(model_name, {}).setdefault(quarter, []).extend(outcomes)
                else:
                    model_name = os.path.splitext(os.path.basename(path))[0]
                    outcomes = _verdict_outcomes(df)
                    results.setdefault(model_name, {}).setdefault(quarter, []).extend(outcomes)
            except Exception as exc:
                print(f"[warn] {quarter} / {path}: {exc}")
    return results


def _verdict_outcomes(df: pd.DataFrame) -> list[int]:
    """Per-row PASSED/FAILED outcomes as 1/0."""
    col = df[VERDICT_COLUMN].str.strip().str.upper().map({"PASSED": 1, "FAILED": 0})
    return col.tolist()


def get_sort_key(item: tuple[str, dict]) -> int:
    """Rank (model_name, quarters) items for panel display order; unknown models sort last."""
    model_name = item[0].lower()
    if "qwen" in model_name:
        return 1
    if "gemini" in model_name:
        return 2
    if "nano" in model_name:
        return 3
    if "gpt-5" in model_name:
        return 4
    return 99


def _filter_and_sort(all_accuracies: dict) -> list[tuple[str, dict]]:
    """Keep only models with data in all four quadrants, in display order."""
    filtered = [
        (model, quarters) for model, quarters in all_accuracies.items()
        if all(q in quarters and quarters[q] for q in ["Q1", "Q2", "Q3", "Q4"])
    ]
    return sorted(filtered, key=get_sort_key)


def _quadrant_matrix(quarters: dict[str, dict[str, float]]) -> np.ndarray:
    """
    Build the 2x2 mean-accuracy matrix for one model:
        rows = symmetric / asymmetric inference, cols = classic / new narratives.
    Missing quadrants become NaN.
    """
    def mean_or_nan(q: str) -> float:
        vals = list(quarters.get(q, {}).values())
        return float(np.mean(vals)) if vals else float("nan")

    return np.array([
        [mean_or_nan("Q1"), mean_or_nan("Q3")],
        [mean_or_nan("Q2"), mean_or_nan("Q4")],
    ])


def _quadrant_variance_matrix(quarters_outcomes: dict[str, list[int]]) -> np.ndarray:
    """
    Build the 2x2 variance matrix for one model, matching _quadrant_matrix's
    layout (rows = symmetric/asymmetric, cols = classic/new narratives).

    Each cell is the population variance (1/m)*sum((x_i - mean)^2) of the
    per-instance PASSED/FAILED (1/0) outcomes pooled across that quadrant.
    Missing quadrants become NaN.
    """
    def variance_or_nan(q: str) -> float:
        outcomes = quarters_outcomes.get(q, [])
        return float(np.var(outcomes)) if outcomes else float("nan")

    return np.array([
        [variance_or_nan("Q1"), variance_or_nan("Q3")],
        [variance_or_nan("Q2"), variance_or_nan("Q4")],
    ])


def _strategy_color(model: str) -> str | None:
    """Return the border color of the first matching strategy, or None."""
    lower = model.lower()
    for details in STRATEGY_MAPPING.values():
        is_match = any(m in lower for m in details["models"])
        is_excluded = any(ex in lower for ex in details.get("exclude", []))
        if is_match and not is_excluded:
            return details["color"]
    return None


def _draw_panel(ax, model: str, quarters: dict, variances: np.ndarray, cmap, norm, first: bool) -> None:
    """Draw one model's 2x2 accuracy heatmap panel, including title, labels, and strategy border."""
    ax.set_facecolor("#ffffff")
    data = _quadrant_matrix(quarters)

    for r in range(2):
        for c in range(2):
            val = data[r, c]
            var = variances[r, c]
            color = cmap(norm(val)) if not np.isnan(val) else "#e0e6ed"
            ax.add_patch(mpatches.FancyBboxPatch(
                (c + 0.02, 1 - r + 0.02), 0.96, 0.96,
                boxstyle="round,pad=0.0,rounding_size=0.07",
                facecolor=color, edgecolor="white", linewidth=2
            ))
            if np.isnan(val):
                text = "N/A"
            elif np.isnan(var):
                text = f"{val:.2%}"
            else:
                text = f"{val:.2%}\n(±{var:.3f})"
            ax.text(c + 0.5, 1 - r + 0.5, text, ha="center", va="center", fontsize=38, color="black",
                     linespacing=1.6)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Classic\nNarratives", "Novel\nNarratives"], fontsize=40, color="#1c1c1c")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", pad=20)
    ax.set_yticks([])

    color = _strategy_color(model)
    if color:
        ax.add_patch(mpatches.FancyBboxPatch(
            (-0.03, -0.03), 2.06, 2.06,
            boxstyle="round,pad=0.0,rounding_size=0.12",
            fill=False, edgecolor=color, lw=12, clip_on=False, zorder=10
        ))

    if first:
        ax.text(-0.25, 0.75, "Symmetric\nInference", transform=ax.transAxes, ha="center",
                va="center", rotation=90, fontsize=40, color="#1c1c1c")
        ax.text(-0.25, 0.25, "Asymmetric\nInference", transform=ax.transAxes, ha="center",
                va="center", rotation=90, fontsize=40, color="#1c1c1c")

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(_format_model_name(model), fontsize=40, weight="bold", pad=45, color="#1c1c1c", wrap=True)


def _add_colorbar_and_legend(fig, axes, cmap, norm) -> None:
    """Attach the shared accuracy colorbar and the strategy legend to the figure."""
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.6, pad=0.03, aspect=40)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["0%", "50%", "100%"])
    cbar.ax.tick_params(labelsize=35, length=0, colors="#1c1c1c")
    cbar.outline.set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor="none", edgecolor=details["color"], linewidth=8, label=strategy)
        for strategy, details in STRATEGY_MAPPING.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.05),
               ncol=3, frameon=False, prop={"size": 40},
               title="Asymmetric Inference Reasoning Strategies",
               title_fontproperties={"size": 40, "weight": "bold"})


def main() -> None:
    """Collect accuracies, render one 2x2 heatmap panel per model, and save the combined figure."""
    sorted_models = _filter_and_sort(collect_accuracies())
    num_panels = len(sorted_models)
    if num_panels == 0:
        return

    instance_outcomes = collect_instance_outcomes()

    fig, axes = plt.subplots(1, num_panels, figsize=(9.0 * num_panels, 12.5),
                             sharey=True, layout="constrained")
    fig.patch.set_facecolor("#ffffff")
    fig.get_layout_engine().set(wspace=0.1)
    if num_panels == 1:
        axes = [axes]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "paper_blues", ["#e6eff8", "#a3c4df", "#5b92bf"])
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for idx, (model, quarters) in enumerate(sorted_models):
        variances = _quadrant_variance_matrix(instance_outcomes.get(model, {}))
        _draw_panel(axes[idx], model, quarters, variances, cmap, norm, first=(idx == 0))

    _add_colorbar_and_legend(fig, axes, cmap, norm)

    plt.savefig(os.path.join(OUTPUT_DIR, "combined_models_accuracy.png"),
                dpi=100, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
