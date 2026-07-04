"""
Internal epistemic solver: ground-truth labels for muddy-children puzzles.

Determines whether the queried agent knows its own status by a given round,
under either observation setup:

  - Random observation: the answer is read directly off the precomputed
    knowledge history (per-round knower sets).
  - Classic (standard) observation: the answer follows the closed-form
    formula.
"""


from src.dataset_generation.boundary_utils import get_logic_type_and_value
from src.constants import LOWER_BOUND_TYPE
from src.dataset_generation.standard_observation_text_generator import first_round_with_some_knowledge


def get_solver_label(children_number, muddy_children_number, round_number, boundary_value, boundary_type, child_index,
                     is_child_muddy, use_random_observation=False, history=None):
    """
    Compute whether a specific child is labeled as "knowing" their mud status by a given round.
    Args:
        children_number (int): Total number of children.
        muddy_children_number (int): Total number of muddy children.
        round_number (int): the reasoning round to evaluate.
        boundary_value (int): Boundary value used to derive the scenario logic.
        boundary_type (Any): Boundary type used to derive the scenario logic.
        child_index (int): Index of queried child.
        is_child_muddy (bool): Whether queried child is muddy.
        use_random_observation (bool): If True, use `history` as knower sets per round.
        history (Sequence[Collection[int]] | None): `history[r-1]` contains knower indices at round r.
    Returns:
        int: 1 if the child is labeled as knowing by `round_number`, else 0.
    Raises:
        ValueError: If `use_random_observation=True` and `round_number > len(history)`.
    """
    if use_random_observation:  #case 1: random observation matrix
        #safety check
        if round_number > len(history):
            raise ValueError("Round number should be at most the number of children + 1")

        round_knowers = history[round_number - 1]
        return 1 if child_index in round_knowers else 0

    else:   #case 2: standard observation matrix
        logic_type, logic_val = get_logic_type_and_value(boundary_type, boundary_value, children_number)

        first_iter = first_round_with_some_knowledge(logic_type, logic_val, muddy_children_number, children_number)
        if first_iter is None: #non-informative bound
            return 0

        if logic_type == LOWER_BOUND_TYPE:
            if is_child_muddy and round_number >= first_iter:
                return 1
            elif (not is_child_muddy) and round_number >= first_iter + 1:
                return 1
            else:
                return 0
        else:  # UPPER_BOUND_TYPE
            if (not is_child_muddy) and round_number >= first_iter:
                return 1
            elif is_child_muddy and round_number >= first_iter + 1:
                return 1
            else:
                return 0
