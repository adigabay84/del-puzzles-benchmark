"""
One-off conversion script for asymmetric-inference puzzles: re-skin an
existing dataset into a different narrative.

Recovers each scenario's parameters from the source records (the observation
matrix is regex-parsed from the hypothesis text, the rest from metadata),
recomputes the knowledge history, and rebuilds the identical scenario under
the target --puzzle_type (default: olympic_games) - same logic, matrices,
and labels, new narrative.

Outputs: <dst>.jsonl and <dst>.csv (default: modified_puzzles.*)

Note: sources must be asymmetric (random-observation) puzzles, with the
matrix rendered verbatim in the hypothesis text.
"""


import json
import re
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_generation.dataset_generator import build_problem, pipeline
from src.dataset_generation.random_obs_text_generator import get_knowledge_history


def extract_matrix(hypothesis: str, n: int) -> np.ndarray:
    """Pull the NxN observation matrix out of the hypothesis text."""
    pattern = r"observation matrix:\n((?:\[[\d,\s]+\]\n?){" + str(n) + r"})"
    match = re.search(pattern, hypothesis)
    if not match:
        raise ValueError("Could not find observation matrix in hypothesis text.")
    rows = re.findall(r"\[([\d,\s]+)\]", match.group(1))
    return np.array([[int(x) for x in row.split(",")] for row in rows])


def convert(src_paths: list[str], dst_path: str, puzzle_type: str):
    rows = []
    for src_path in src_paths:
        rows.extend(json.loads(line) for line in Path(src_path).open())

    problems = []
    for row in rows:
        n = row["children_number"]
        matrix = extract_matrix(row["hypothesis"], n)
        world_mask = row["world_mask"]
        if isinstance(world_mask, str):
            import ast
            world_mask = ast.literal_eval(world_mask)
        real_world = np.array(world_mask, dtype=int)

        history = get_knowledge_history(
            number_of_children=n,
            observation_matrix=matrix.tolist(),
            real_world=real_world,
            boundary_value=row["boundary_value"],
            boundary_type=row["boundary_type"],
        )

        result = build_problem(
            children_number=n,
            muddy_children_number=row["muddy_children_number"],
            round_number=row["round_number"],
            child_index=row["child_index"],
            boundary_value=row["boundary_value"],
            boundary_type=row["boundary_type"],
            is_child_muddy=row["is_child_muddy"],
            puzzle_type=puzzle_type,
            use_special_name=False,
            matrix=matrix,
            use_random_observation=True,
            history=history,
            insert_solution_approach=False,
        )
        problems.extend(result)

    df = pipeline(problems, random_observation_extreme=False)
    dst = Path(dst_path)
    df.to_json(dst.with_suffix(".jsonl"), orient="records", lines=True)
    df.to_csv(dst.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    import argparse
    _root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--srcs", nargs="*", default=[str(_root / "puzzles.jsonl")])
    parser.add_argument("--dst", default=str(_root / "modified_puzzles"))
    parser.add_argument("--puzzle_type", default="olympic_games")
    args = parser.parse_args()
    convert(args.srcs, args.dst, args.puzzle_type)
