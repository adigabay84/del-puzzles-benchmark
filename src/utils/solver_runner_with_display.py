"""
Solver runner with display of the possible-worlds epistemic simulation.

Intended for manually tracing a single scenario, running it as a script executes a
hardcoded example (n=10, a specific random observation matrix, 8 muddy agents,
lower bound 8) and prints the full round-by-round trace and final history.
"""


import numpy as np
from numpy import ndarray
from src.constants import LOWER_BOUND_TYPE, NOT_LESS_THAN_MUDDY_TYPE, UPPER_BOUND_TYPE, AT_LEAST_CLEAN_TYPE


def find_knowers(number_of_children: int, observation_matrix: ndarray, real_world: ndarray, worlds: ndarray):
    round_knowers = [] #list of agents that know their status in the current round

    for i in range(number_of_children):
        visible_indices = np.where(observation_matrix[i] == 1)[0] #access the actual array of visible indices
        matches = np.all(worlds[:, visible_indices] == real_world[visible_indices], axis=1)
        compatible_worlds = worlds[matches] #worlds that are consistent with what agent i sees

        #if in all possible worlds, the agent's status is the same - they know their own status
        if len(compatible_worlds) > 0 and len(np.unique(compatible_worlds[:, i])) == 1:
            round_knowers.append(i)

    return round_knowers


def get_knowledge_history(number_of_children: int, observation_matrix: list[list[int]], real_world: ndarray,
                          boundary_value: int, boundary_type: str):
    observation_matrix = np.array(observation_matrix)
    worlds = np.array([list(np.binary_repr(i, width=number_of_children)) for i in range(2 ** number_of_children)], dtype=int)

    # Initial Filtering based on boundary
    sums = worlds.sum(axis=1)
    if boundary_type == LOWER_BOUND_TYPE or boundary_type == NOT_LESS_THAN_MUDDY_TYPE:
        worlds = worlds[sums >= boundary_value]
    elif boundary_type == UPPER_BOUND_TYPE:
        worlds = worlds[sums <= boundary_value]
    elif boundary_type == AT_LEAST_CLEAN_TYPE:
        worlds = worlds[sums <= (number_of_children - boundary_value)]

    history = []

    print(f"\n--- STARTING SIMULATION ---")
    print(f"Number of Children: {number_of_children}")
    print(f"Boundary Type: {boundary_type}")
    print(f"Boundary Value: {boundary_value}")
    print(f"Real World: {real_world}")
    print(f"Initial Possible Worlds Count: {len(worlds)}")
    print(f"Observation Matrix:\n {observation_matrix}")

    for r in range(number_of_children + 1):
        print(f"\n>>> ROUND {r + 1} <<<")

        round_knowers = find_knowers(number_of_children=number_of_children, observation_matrix=observation_matrix,
                                     real_world=real_world, worlds=worlds)
        history.append(round_knowers)
        print(f"Agents who know this round: {round_knowers}")

        if len(worlds) <= 1:
            print("Simulation ends: All ambiguity resolved.")
            remaining_rounds = (number_of_children + 1) - len(history)
            history.extend([round_knowers] * remaining_rounds)
            break

        keep_worlds_indices = []
        eliminated_worlds_list = []

        for idx in range(len(worlds)):
            w = worlds[idx]
            # test who would have known if w was the real world?
            w_knowers = find_knowers(number_of_children=number_of_children, observation_matrix=observation_matrix,
                                     real_world=w, worlds=worlds)

            if w_knowers == round_knowers:
                keep_worlds_indices.append(idx)
            else:
                # Store eliminated world and the reason (inconsistent knowers)
                eliminated_worlds_list.append((w, w_knowers))

        # Print the eliminated worlds
        if eliminated_worlds_list:
            print(f"Eliminated {len(eliminated_worlds_list)} worlds this round:")
            for ew, k_list in eliminated_worlds_list:
                print(
                    f"  World {ew} removed. (In that world, knowers would be {k_list}, but real knowers are {round_knowers})")
        else:
            print("No worlds eliminated this round.")

        worlds = worlds[keep_worlds_indices]
        print(f"\nRemaining worlds: {len(worlds)}")
        for world in worlds:
            print(
                f"  World {world} stays.)")

    return history


if __name__ == "__main__":
    N_CHILDREN = 10

    OBS_MATRIX = [
        [0, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
    ]

    boolean_input = [True, True, True, True, True, True, True, True, False, False]
    REAL_WORLD = np.array([1 if x else 0 for x in boolean_input])

    BOUNDARY_VAL = 8
    BOUNDARY_TP = LOWER_BOUND_TYPE

    history = get_knowledge_history(
        number_of_children=N_CHILDREN,
        observation_matrix=OBS_MATRIX,
        real_world=REAL_WORLD,
        boundary_value=BOUNDARY_VAL,
        boundary_type=BOUNDARY_TP
    )

    print("\n--- FINAL HISTORY ---")
    print(history)
