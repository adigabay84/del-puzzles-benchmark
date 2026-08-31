"""
Compute pairwise significance testing between the 5 main-analysis models.

Reads the same post-processed result CSVs as compute_quadrant_accuracy.py,
pools every per-instance PASSED/FAILED outcome across all four quadrants for
each model, and compares each pair of models with a two-proportion z-test on
their overall accuracy (instances are treated as independent Bernoulli
trials; the pooled-variance z-test is the standard test for whether two
observed proportions differ significantly).

Outputs:
    Graphs/model_significance_pvalues.csv
    A Markdown p-value table printed to stdout (renders in OpenReview reviews/rebuttals).
"""


import math
import os
import pandas as pd
from src.analysis.quadrant_files import QUARTERS


VERDICT_COLUMN = "verdict"
OUTPUT_DIR = "Graphs"
OUTPUT_CSV = "model_significance_pvalues.csv"

MODEL_SORT_KEY = {
    "qwen": 1,
    "gemini": 2,
    "nano": 3,
    "gpt-5": 4,
    "claude": 5,
}


def collect_pooled_outcomes() -> dict[str, list[int]]:
    """
    Read every CSV listed in QUARTERS and pool each model's per-instance
    PASSED/FAILED (1/0) outcomes across all four quadrants.

    Returns {model_name: [1, 0, 1, ...]}.
    """
    results: dict[str, list[int]] = {}

    for paths in QUARTERS.values():
        for path in paths:
            try:
                df = pd.read_csv(path)
                if VERDICT_COLUMN not in df.columns or "model" not in df.columns or "version" not in df.columns:
                    continue
                for (model, version), sub_df in df.groupby(["model", "version"]):
                    model_name = f"{str(model).strip()} {str(version).strip()}"
                    col = sub_df[VERDICT_COLUMN].str.strip().str.upper().map({"PASSED": 1, "FAILED": 0})
                    results.setdefault(model_name, []).extend(col.tolist())
            except Exception as exc:
                print(f"[warn] {path}: {exc}")
    return results


def _sort_key(model_name: str) -> int:
    """Rank models for display order (qwen, gemini, nano, gpt-5, claude); unknown models sort last.
        The nano guard prevents 'gpt-5 nano' from matching the plain 'gpt-5' keyword."""
    lower = model_name.lower()
    for keyword, rank in MODEL_SORT_KEY.items():
        if keyword in lower and not ("nano" in lower and keyword == "gpt-5"):
            return rank
    return 99


def filter_and_sort(pooled: dict[str, list[int]], quarters_by_model: dict[str, set[str]]) -> list[str]:
    """Keep only models with data in all four quadrants, in display order."""
    filtered = [model for model in pooled if quarters_by_model.get(model, set()) >= {"Q1", "Q2", "Q3", "Q4"}]
    return sorted(filtered, key=_sort_key)


def collect_quarters_by_model() -> dict[str, set[str]]:
    """Return {model_name: set of quarters the model has data in}, used to restrict to the 5 main models."""
    quarters_by_model: dict[str, set[str]] = {}
    for quarter, paths in QUARTERS.items():
        for path in paths:
            try:
                df = pd.read_csv(path)
                if "model" not in df.columns or "version" not in df.columns:
                    continue
                for model, version in df[["model", "version"]].drop_duplicates().itertuples(index=False):
                    model_name = f"{str(model).strip()} {str(version).strip()}"
                    quarters_by_model.setdefault(model_name, set()).add(quarter)
            except Exception as exc:
                print(f"[warn] {path}: {exc}")
    return quarters_by_model


def two_proportion_z_test(x1: int, n1: int, x2: int, n2: int) -> float:
    """
    Two-sided p-value from a pooled two-proportion z-test comparing two
    independent accuracy rates p1 = x1/n1 and p2 = x2/n2.
    Returns 1.0 if the two proportions are identical (including the
    degenerate case where the pooled proportion is 0 or 1).
    """
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    p_value = 2 * (1 - _standard_normal_cdf(abs(z)))
    return p_value


def _standard_normal_cdf(z: float) -> float:
    """Standard normal CDF via the error function (avoids a scipy dependency)."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def compute_pvalue_matrix(models: list[str], pooled: dict[str, list[int]]) -> pd.DataFrame:
    """Build the symmetric model x model p-value matrix (NaN on the diagonal)."""
    matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    for i, model_a in enumerate(models):
        outcomes_a = pooled[model_a]
        x_a, n_a = sum(outcomes_a), len(outcomes_a)
        for model_b in models[i + 1:]:
            outcomes_b = pooled[model_b]
            x_b, n_b = sum(outcomes_b), len(outcomes_b)
            p_value = two_proportion_z_test(x_a, n_a, x_b, n_b)
            matrix.loc[model_a, model_b] = p_value
            matrix.loc[model_b, model_a] = p_value
    return matrix


def _significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _format_pvalue(p_value: float) -> str:
    """Format a p-value, falling back to '<0.0001' instead of a misleading rounded '0.0000'."""
    return "<0.0001" if p_value < 0.0001 else f"{p_value:.4f}"


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in model names."""
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


def print_table(matrix: pd.DataFrame) -> None:
    """Print the p-value table as a LaTeX table (paste directly into Overleaf)."""
    n_cols = len(matrix.columns)
    col_spec = "l" + "c" * n_cols

    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\begin{tabular}{" + col_spec + "}")
    print(r"\toprule")

    header = "Model & " + " & ".join(_escape_latex(c) for c in matrix.columns) + r" \\"
    print(header)
    print(r"\midrule")

    for model_a in matrix.index:
        cells = []
        for model_b in matrix.columns:
            if model_a == model_b:
                cells.append("--")
            else:
                p_value = matrix.loc[model_a, model_b]
                cells.append(f"{_format_pvalue(p_value)}{_significance_stars(p_value)}")
        row = _escape_latex(model_a) + " & " + " & ".join(cells) + r" \\"
        print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Pairwise significance ($p$-values) between models, two-proportion $z$-test on pooled overall accuracy. "
          r"$*\,p<0.05$, $**\,p<0.01$, $***\,p<0.001$.}")
    print(r"\label{tab:model_significance}")
    print(r"\end{table}")


# def print_table(matrix: pd.DataFrame) -> None:
#     """Print the p-value table as a Markdown table (as supported in OpenReview reviews/rebuttals)."""
#     header = "| Model | " + " | ".join(matrix.columns) + " |"
#     separator = "|---|" + "---|" * len(matrix.columns)
#     print(header)
#     print(separator)
#     for model_a in matrix.index:
#         cells = []
#         for model_b in matrix.columns:
#             if model_a == model_b:
#                 cells.append("--")
#             else:
#                 p_value = matrix.loc[model_a, model_b]
#                 cells.append(f"{_format_pvalue(p_value)}{_significance_stars(p_value)}")
#         print(f"| {model_a} | " + " | ".join(cells) + " |")
#     print("\n\\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001; two-proportion z-test on pooled overall accuracy.")


def main() -> None:
    """Collect pooled per-instance outcomes, run pairwise significance tests, and export the results."""
    pooled = collect_pooled_outcomes()
    quarters_by_model = collect_quarters_by_model()
    models = filter_and_sort(pooled, quarters_by_model)
    if not models:
        return

    matrix = compute_pvalue_matrix(models, pooled)

    print_table(matrix)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    matrix.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_CSV))


if __name__ == "__main__":
    main()