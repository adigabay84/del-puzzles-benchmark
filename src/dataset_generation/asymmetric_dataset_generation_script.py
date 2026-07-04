"""
Generate 100 critical-round-3 Olympic Games puzzles.

Fixed configuration: n=10 agents, queried agent 0, lower-bound boundary
with value 8, seed 1.

Sweeps all valid (muddy count, is-muddy) combinations, trying up to 10,000
random observation matrices each and keeping only scenarios whose critical
round is exactly 3, spread roughly evenly across combinations
(per-combination cap of ceil(100 / #valid combos)), deduplicated on
(premise, hypothesis).

Outputs:
  - asymmetric_inference_olympic_games.jsonl / .csv
"""


import numpy as np
from src.constants import LOWER_BOUND_TYPE, OLYMPIC_GAMES_PUZZLE, UPPER_BOUND_TYPE
from src.dataset_generation.boundary_utils import get_logic_type_and_value
from src.dataset_generation.dataset_generator import canonical_muddy_mask, build_problem, pipeline
from src.dataset_generation.random_obs_text_generator import get_knowledge_history


np.random.seed(1)
n, ci, bv = 10, 0, 8


def _critical_round(child_index, history):
    """Return the first 1-indexed round where child_index appears in history, or None."""
    for r, knowers in enumerate(history, start=1):
        if child_index in knowers:
            return r
    return None


def _scenario_is_valid(boundary_type, boundary_value, muddy_children_number,
                       children_number, is_child_muddy):
    """Mirror the validity filters used in build_all_possible_problems."""
    l_type, l_val = get_logic_type_and_value(boundary_type, boundary_value, children_number)
    if l_type == LOWER_BOUND_TYPE and l_val > muddy_children_number:
        return False
    if l_type == UPPER_BOUND_TYPE and l_val < muddy_children_number:
        return False
    if is_child_muddy and muddy_children_number == 0:
        return False
    if not is_child_muddy and muddy_children_number == children_number:
        return False
    return True


configs = []
seen = set()
combo_counts = {}
n_combos = sum(1 for m in range(n+1) for im in [True, False]
               if _scenario_is_valid(LOWER_BOUND_TYPE, bv, m, n, im))
per_combo = -(-100 // n_combos)

for muddy_n in range(n + 1):
    for is_muddy in [True, False]:
        if not _scenario_is_valid(LOWER_BOUND_TYPE, bv, muddy_n, n, is_muddy):
            continue
        real_world = np.array(canonical_muddy_mask(n, muddy_n, ci, is_muddy), dtype=int)
        for _ in range(10_000):
            if len(configs) >= 100 or combo_counts.get((muddy_n, is_muddy), 0) >= per_combo:
                break
            matrix = np.random.randint(0, 2, (n, n))
            matrix[ci][ci] = 0
            history = get_knowledge_history(n, matrix.tolist(), real_world, bv, LOWER_BOUND_TYPE)
            if _critical_round(ci, history) != 3:
                continue
            puzzles = build_problem(n, muddy_n, 3, ci, bv, LOWER_BOUND_TYPE, is_muddy,
                                    OLYMPIC_GAMES_PUZZLE, False, False, matrix,
                                    True, history)
            if not puzzles:
                continue
            key = (puzzles[0]["premise"], puzzles[0]["hypothesis"])
            if key not in seen:
                seen.add(key)
                configs.append(puzzles[0])
                combo_counts[(muddy_n, is_muddy)] = combo_counts.get((muddy_n, is_muddy), 0) + 1

df = pipeline(configs, random_observation_extreme=True)
df.to_json("asymmetric_inference_olympic_games.jsonl", orient="records", lines=True)
df.to_csv("asymmetric_inference_olympic_games.csv", index=False)
