"""
Compute per-quadrant macro F1 scores table.

Reads post-processed result CSVs for each puzzle/model combination, grouped
into four quadrants along two axes: narrative type (classic vs. new) and
inference type (symmetric vs. asymmetric). For each model, macro F1 is
computed per puzzle from "correct response" vs. "model response" (over the
classes present in the ground truth; invalid responses count as wrong),
then averaged across puzzles within each quadrant.

Outputs:
  - A human-readable table to stdout
  - A ready-to-paste LaTeX table to stdout (skip with --no_latex)
"""


from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

from src.analysis.quadrant_files import QUARTERS
from src.constants import (
    DONT_KNOW_CANONICAL,
    NO_CANONICAL,
    RESPONSE_FORMAT,
    YES_CANONICAL,
)


CLASSES = [YES_CANONICAL, NO_CANONICAL, DONT_KNOW_CANONICAL]

MODEL_SORT_KEY = {
    "qwen": 1,
    "gemini": 2,
    "nano": 3,
    "gpt-5": 4,
    "claude": 5,
}

QUARTER_LABELS = {
    "Q1": "Classic Sym.",
    "Q2": "Classic Asym.",
    "Q3": "New Sym.",
    "Q4": "New Asym.",
}


def _normalize_response(s_raw) -> str:
    """Return canonical class label, or 'INVALID' if unrecognized."""
    if pd.isna(s_raw):
        return "INVALID"
    s = str(s_raw).strip()
    m = RESPONSE_FORMAT.fullmatch(s)
    if not m:
        return "INVALID"
    return m.group("ans")


def _compute_macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Macro F1 over classes present in y_true; INVALID predictions count as wrong."""
    present = [cls for cls in CLASSES if cls in set(y_true)]
    if not present:
        return 0.0
    f1s = []
    for cls in present:
        tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
        fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
        fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision + recall
        f1s.append((2 * precision * recall / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def _fmt(val: float) -> str:
    """Format to 4 decimal places by truncation, not rounding."""
    return f"{int(val * 10000) / 10000:.4f}"


def _sort_key(model_name: str) -> int:
    """Rank models for display order (qwen, gemini, nano, gpt-5, claude); unknown models sort last.
        The nano guard prevents 'gpt-5 nano' from matching the plain 'gpt-5' keyword."""
    lower = model_name.lower()
    for keyword, rank in MODEL_SORT_KEY.items():
        if keyword in lower and not ("nano" in lower and keyword == "gpt-5"):
            return rank
    return 99


def collect() -> dict[str, dict[str, list[float]]]:
    """
    Returns {model_name: {quarter: [per_puzzle_f1, ...]}}
    F1 is computed separately for each puzzle file, then averaged across puzzles per quadrant.
    """
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for quarter, paths in QUARTERS.items():
        for path in paths:
            if not Path(path).exists():
                print(f"[warn] missing: {path}")
                continue
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print(f"[warn] {path}: {exc}")
                continue

            if not {"model", "version", "model response", "correct response"}.issubset(df.columns):
                print(f"[warn] missing columns in {path}")
                continue

            for (model, version), grp in df.groupby(["model", "version"], sort=False):
                name = f"{str(model).strip()} {str(version).strip()}"
                y_true = grp["correct response"].tolist()
                y_pred = grp["model response"].apply(_normalize_response).tolist()
                data[name][quarter].append(_compute_macro_f1(y_true, y_pred))

    return data


def compute_results(data: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
    """Returns {model_name: {quarter: mean_f1_across_puzzles}}"""
    results: dict[str, dict[str, float]] = {}
    for model, quarters in data.items():
        results[model] = {}
        for q, f1_list in quarters.items():
            results[model][q] = float(np.mean(f1_list)) if f1_list else 0.0
    return results


def _print_table(results: dict[str, dict[str, float]]) -> None:
    """Print a human-readable table of per-quadrant F1 scores to stdout,
        one row per model in display order; missing quadrants shown as N/A."""
    quarters = list(QUARTERS)
    header = f"{'Model':<35}" + "".join(f"{QUARTER_LABELS[q]:>14}" for q in quarters)
    print(header)
    print("-" * len(header))
    for model in sorted(results, key=_sort_key):
        row = f"{model:<35}"
        for q in quarters:
            val = results[model].get(q)
            row += f"{_fmt(val):>14}" if val is not None else f"{'N/A':>14}"
        print(row)


def _print_latex(results: dict[str, dict[str, float]]) -> None:
    """Print a ready-to-paste LaTeX table of per-quadrant F1 scores to stdout,
        including caption and label; missing quadrants shown as '--'."""
    quarters = list(QUARTERS)
    print("\n% --- LaTeX table ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Classic Sym.} & \textbf{Classic Asym.} "
          r"& \textbf{New Sym.} & \textbf{New Asym.} \\")
    print(r"\midrule")
    for model in sorted(results, key=_sort_key):
        cells = []
        for q in quarters:
            val = results[model].get(q)
            cells.append(_fmt(val) if val is not None else "--")
        print(f"{model} & {' & '.join(cells)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{\textbf{Macro F1 scores across the four benchmark quadrants.}"
          r" For each quadrant, macro F1 is computed separately per puzzle over the two"
          r" ground-truth classes (\textit{Yes} and \textit{No}), then averaged across puzzles."
          r" Since models are queried only at the pivotal round, \textit{I don't know} is never"
          r" a correct answer and is excluded from the average.}")
    print(r"\label{tab:macro_f1}")
    print(r"\end{table}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments (--no_latex to skip the LaTeX output)."""
    p = argparse.ArgumentParser(description="Compute per-quadrant macro F1 for main-analysis models.")
    p.add_argument("--no_latex", action="store_true", help="Skip LaTeX output.")
    return p.parse_args()


def main() -> None:
    """Entry point: collect per-puzzle F1 scores, average per quadrant, and print the table(s)."""
    args = _parse_args()
    data = collect()
    results = compute_results(data)
    _print_table(results)
    if not args.no_latex:
        _print_latex(results)


if __name__ == "__main__":
    main()
