"""
One-off conversion script for symmetric-inference puzzles: re-skin an
existing dataset into a different narrative.

Rebuilds each scenario from the source records' metadata alone under the
target --puzzle_type (default: olympic_games) - same parameters and labels,
new narrative. Since the classic (symmetric) setup has no observation
matrix, no text parsing or history computation is needed.

Outputs: <dst>.jsonl and <dst>.csv (default: modified_puzzles.*)

Note: sources must be symmetric (classic-observation) puzzles.
"""


import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_generation.dataset_generator import build_problem, pipeline


def convert(src_paths: list[str], dst_path: str, puzzle_type: str):
    rows = []
    for src_path in src_paths:
        rows.extend(json.loads(line) for line in Path(src_path).open())

    problems = []
    for row in rows:
        result = build_problem(
            children_number=row["children_number"],
            muddy_children_number=row["muddy_children_number"],
            round_number=row["round_number"],
            child_index=row["child_index"],
            boundary_value=row["boundary_value"],
            boundary_type=row["boundary_type"],
            is_child_muddy=row["is_child_muddy"],
            puzzle_type=puzzle_type,
            use_special_name=False,
            matrix=None,
            use_random_observation=False,
            history=None,
            insert_solution_approach=False,
        )
        problems.extend(result)

    df = pipeline(problems, random_observation_extreme=False)
    dst = Path(dst_path)
    df.to_json(dst.with_suffix(".jsonl"), orient="records", lines=True)
    df.to_csv(dst.with_suffix(".csv"), index=False)
    print(f"Saved {len(df)} rows to {dst.with_suffix('.jsonl')} and {dst.with_suffix('.csv')}")


if __name__ == "__main__":
    import argparse
    _root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcs", nargs="*", default=[str(_root / "puzzles.jsonl")])
    parser.add_argument("--dst", default=str(_root / "modified_puzzles"))
    parser.add_argument("--puzzle_type", default="olympic_games")
    args = parser.parse_args()
    convert(args.srcs, args.dst, args.puzzle_type)
