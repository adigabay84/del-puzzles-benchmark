"""
Possible-worlds epistemic simulation and NL text for the random-observation setting.

The simulation implements the standard public-announcement dynamics over a
Kripke-style possible-worlds model: candidate worlds are all 0/1 status
assignments consistent with the announced boundary; each round, an agent
"knows" its status if it is identical across all worlds matching its
observations (per the visibility matrix), and the public pattern of who
knew / didn't know then eliminates inconsistent worlds. Iteration runs for
n+1 rounds, short-circuiting once a steady state or a single world is
reached (the final knower set is repeated for the remaining rounds).

Provides:
  - find_knowers:            per-round knower detection.
  - get_knowledge_history:   full per-round knower sets (length n+1) - the
                             ground-truth "history" consumed by the solver
                             and dataset generators.
  - previous_rounds_text_random_observation: NL summary of prior rounds'
                             answers, compressing consecutive identical
                             rounds into ranges.
  - natural_language_observation_vector: NL description of which agents the
                             queried agent sees in each status.
  - format_list:             agent-index list -> natural-language phrase.

Note: world enumeration is exponential (2^n worlds).
"""


import numpy as np
from numpy import ndarray
from src.constants import LOWER_BOUND_TYPE, NOT_LESS_THAN_MUDDY_TYPE, UPPER_BOUND_TYPE, AT_LEAST_CLEAN_TYPE, rounds_range
from src.puzzles import PUZZLE_CONFIGS


def find_knowers(number_of_children: int, observation_matrix: ndarray, real_world: ndarray, worlds: ndarray):
    """
    Find which agents know their own status in the current reasoning round.
    Args:
        number_of_children (int): Total number of agents.
        observation_matrix (ndarray): Visibility matrix where row i indicates who agent i can see (1 = visible).
        real_world (ndarray): The true world assignment (length `number_of_children`, 0/1 status per agent).
        worlds (ndarray): Candidate worlds consistent with the current knowledge state
        (shape: [num_worlds, number_of_children]).
    Returns:
        list[int]: Indices of agents that can deduce their own status in this round.
    """
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
    """
    Simulate iterative public reasoning and return who knows in each round.
    Args:
        number_of_children (int): Total number of agents.
        observation_matrix (list[list[int]]): Visibility matrix where row i indicates who agent i can see (1 = visible).
        real_world (ndarray): The true world assignment (length `number_of_children`, 0/1 status per agent).
        boundary_value (int): Boundary value used to filter initial possible worlds (e.g., lower/upper bound).
        boundary_type (str): Boundary type that determines how to filter worlds (e.g., LOWER_BOUND_TYPE, UPPER_BOUND_TYPE).
    Returns:
        list[list[int]]: History of knower indices per round (length `number_of_children + 1`).
    """
    observation_matrix = np.array(observation_matrix)
    worlds = np.array([list(np.binary_repr(i, width=number_of_children)) for i in range(2 ** number_of_children)], dtype=int)

    sums = worlds.sum(axis=1)
    if boundary_type == LOWER_BOUND_TYPE or boundary_type == NOT_LESS_THAN_MUDDY_TYPE:
        worlds = worlds[sums >= boundary_value]
    elif boundary_type == UPPER_BOUND_TYPE:
        worlds = worlds[sums <= boundary_value]
    elif boundary_type == AT_LEAST_CLEAN_TYPE:
        worlds = worlds[sums <= (number_of_children - boundary_value)]

    history = [] #list of knowers in each reasoning round

    for _ in range(number_of_children + 1):
        #check each agent's status - knows or not
        round_knowers = find_knowers(number_of_children=number_of_children, observation_matrix=observation_matrix,
                                     real_world=real_world, worlds=worlds)
        history.append(round_knowers)

        if len(worlds) <= 1: #all agents know their status - no need to calculate other rounds
            remaining_rounds = (number_of_children + 1) - len(history)
            history.extend([round_knowers] * remaining_rounds)
            break

        #check which possible worlds are left after the current reasoning round
        keep_worlds_indices = []
        for idx in range(len(worlds)):
            w = worlds[idx]
            #test who would have known in world w?
            w_knowers = find_knowers(number_of_children=number_of_children, observation_matrix=observation_matrix,
                                     real_world=w, worlds=worlds)
            if w_knowers == round_knowers:  #else the world is in contradiction with the agents that know, so it gets eliminated
                keep_worlds_indices.append(idx)

        if np.array_equal(worlds[keep_worlds_indices], worlds): #no elimination of worlds - steady state, stop evaluation
            remaining_rounds = (number_of_children + 1) - len(history)
            history.extend([round_knowers] * remaining_rounds)
            break
        worlds = worlds[keep_worlds_indices]

    return history


def format_list(lst: list, cfg):
    """
    Convert a list of agent indices into a short natural-language list.
    Args:
        lst (list): Agent indices to format.
        cfg (dict): Puzzle configuration containing agent naming fields.
    Returns:
        str: Natural-language list (e.g., "child 1", "children 1 and 2", "children 1, 2, and 3").
    """
    lst = [str(x) for x in lst]
    identifier = cfg['agent_single'] if len(lst) == 1 else cfg['agent_role']

    if len(lst) == 1:
        return f"{identifier} {lst[0]}"
    elif len(lst) == 2:
        return f"{identifier} {lst[0]} and {lst[1]}"
    else:
        return f"{identifier} " + ", ".join(lst[:-1]) + ", and " + lst[-1]


def previous_rounds_text_random_observation(round_number: int, children_number: int, puzzle_type: str, history=None) -> str:
    """
    Build a natural-language summary of answers in previous rounds (random observation setting).
    Args:
        round_number (int): Current 1-indexed round number (previous rounds are [1, round_number-1]).
        children_number (int): Total number of agents.
        puzzle_type (str): Key for selecting the puzzle configuration.
        history (Sequence[list[int]] | None): `history[r-1]` contains knower indices at round r.
    Returns:
        str: A sentence (or empty string if round_number == 1) describing who knew/didn't know in prior rounds.
    """
    if round_number == 1:
        return ""

    cfg = PUZZLE_CONFIGS[puzzle_type]
    text_parts = []

    i = 0
    while i < min(len(history), round_number - 1):
        current_knowers = history[i]
        start_round = i + 1
        end_round = start_round

        while end_round < min(len(history), round_number - 1):
            if history[end_round] == current_knowers:
               end_round += 1
            else:
               break

        round_range = rounds_range(a=start_round, b=end_round)
        if not current_knowers:
            text_parts.append(f"In {round_range}, all {cfg['agent_role']} answered that they {cfg['unk_txt']}.")
        elif len(current_knowers) == children_number:
            text_parts.append(f"In {round_range}, all {cfg['agent_role']} answered that they {cfg['known_txt']}.")
        else: #some know and some don't
            knowers_str = format_list(current_knowers, cfg)
            the_rest_str = f"while the rest answered that they {cfg['unk_txt']}"

            text_parts.append(f"In {round_range}, {knowers_str} answered that they {cfg['known_txt']}, {the_rest_str}.")

        i = end_round

    return " " + " ".join(text_parts)


def natural_language_observation_vector(cfg, observation_vector: list, world_mask: list):
    """
    Generates natural language description of the agent's observation vector.
    Args:
        cfg: the puzzle configuration.
        observation_vector: the observation vector to convert to NL representation.
        world_mask: the world mask of assigned status for each agent.
    Returns:
        the string representation of the agent observation vector if it has at least 1 observation, else returns
        an empty string.
    """
    if observation_vector is not None and world_mask is not None:
        pos_indices = []
        neg_indices = []
        for i, (can_see, is_pos) in enumerate(zip(observation_vector, world_mask)):
            if can_see:  # 1 in the corresponding cell
                pos_indices.append(i) if is_pos else neg_indices.append(i)

        if not pos_indices and not neg_indices:
            return ""

        joined_desc = ""
        action_pos = cfg['pos_action_singular'] if len(pos_indices) == 1 else cfg['pos_action']
        action_neg = cfg['neg_action_singular'] if len(neg_indices) == 1 else cfg['neg_action']
        if pos_indices:
            joined_desc += f" {format_list(pos_indices, cfg)} {action_pos}"
        if neg_indices:
            if joined_desc != "":
                joined_desc += " and"
            joined_desc += f" {format_list(neg_indices, cfg)} {action_neg}"
        return "Specifically, you see that" + joined_desc
    return ""
