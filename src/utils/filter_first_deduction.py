"""
Filter a symmetric-inference dataset down to each scenario's first deduction round.

From a generated (classic-setup) puzzle dataset containing every round per
scenario, keeps exactly one row per scenario - the earliest round at which
the queried agent can deduce its status (solver_label == 1). Scenarios that
never reach a deduction are dropped.

Optionally runs the generation step first (--run-generate invokes
`src.main generate` as a subprocess with --number_of_agents and
--boundary_types) before filtering.

Outputs: <output>.jsonl plus a .csv alongside it
(default: muddy_children_puzzles_dataset_first_deduction.jsonl / .csv)
"""


import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd


SCENARIO_KEYS = [
    "puzzle_type",
    "children_number",
    "muddy_children_number",
    "boundary_type",
    "boundary_value",
    "child_index",
    "is_child_muddy",
]


def filter_first_deduction(df: pd.DataFrame) -> pd.DataFrame:
    """
    From each scenario group keep only the first round where solver_label == 1.
    Groups that never reach solver_label == 1 are dropped entirely.
    """
    deduced = df[df["solver_label"] == 1].copy()

    # For each scenario, find the earliest round that has solver_label == 1
    first_round_idx = (
        deduced.groupby(SCENARIO_KEYS)["round_number"].idxmin()
    )

    return df.loc[first_round_idx.values].reset_index(drop=True)


def run_generate(number_of_agents: int, boundary_types: list[str], dataset_path: str):
    """generates the full dataset before filtering if requested."""
    cmd = [
        sys.executable, "-m", "src.main", "generate",
        "--number_of_agents", str(number_of_agents),
        "--boundary_types", *boundary_types,
        "--dataset_path", dataset_path,
    ]
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Keep only the first deduction round per agent scenario in the symmetric case."
    )
    parser.add_argument(
        "--input", default="test_results_old/baseline/Datasets/baseline_puzzles.jsonl",
        help=f"Path to the generated JSONL dataset"
    )
    parser.add_argument(
        "--output", default="muddy_children_puzzles_dataset_first_deduction.jsonl",
        help=f"Path for the filtered output JSONL"
    )
    parser.add_argument(
        "--run-generate", action="store_true",
        help="Run the generate command before filtering (uses --number_of_agents and --boundary_types)"
    )
    parser.add_argument(
        "--number_of_agents", type=int, default=10,
        help="Passed to generate command when --run-generate is set (default: 10)"
    )
    parser.add_argument(
        "--boundary_types", nargs="+", default=["lower"],
        help="Passed to generate command when --run-generate is set (default: lower)"
    )
    args = parser.parse_args()

    if args.run_generate:
        run_generate(
            number_of_agents=args.number_of_agents,
            boundary_types=args.boundary_types,
            dataset_path=args.input,
        )

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file '{input_path}' not found.")
        print("Run with --run-generate to generate the dataset first, or pass --input <path>.")
        sys.exit(1)

    df = pd.read_json(input_path, lines=True)

    missing = [k for k in SCENARIO_KEYS + ["solver_label", "round_number"] if k not in df.columns]
    if missing:
        print(f"Error: dataset is missing required columns: {missing}")
        sys.exit(1)

    filtered = filter_first_deduction(df)
    print(f"Rows after filtering (first deduction only): {len(filtered)}")

    output_path = Path(args.output)
    filtered.to_json(output_path, orient="records", lines=True)
    csv_path = output_path.with_suffix(".csv")
    filtered.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
