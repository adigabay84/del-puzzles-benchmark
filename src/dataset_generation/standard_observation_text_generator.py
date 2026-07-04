"""
Closed-form knowledge dynamics and NL text for the classic (standard) observation setting.

In the classic setup - every agent sees all others but not themselves - the
knowledge cascade has a closed form and needs no possible-worlds simulation:
after an informative announcement, knowledge first arises at round
s = |muddy_count - bound| + 1, reached first by the agents on the announced
side of the bound (muddy agents for a lower bound, clean for an upper bound),
with the remaining agents following one round later. Uninformative bounds
("at least 0", "at most n") never produce knowledge.

Provides:
  - first_round_with_some_knowledge: the round s above, or None if the
    bound is uninformative. Also used by the internal solver for labels.
  - previous_rounds_text_standard_observation: NL narrative of prior rounds'
    answers ("all answered they didn't know" / who knew at round s / "all
    knew afterwards"), with the all-muddy / all-clean edge case where
    everyone learns simultaneously.
"""


from src.dataset_generation.boundary_utils import get_logic_type_and_value, is_bound_informative
from src.constants import LOWER_BOUND_TYPE, rounds_range
from src.puzzles import PUZZLE_CONFIGS


def first_round_with_some_knowledge(boundary_type: str, boundary_value: int,
                                   muddy_children_number: int, children_number: int):
    """
    Return the first round number when some agents can infer their status, else (knowledge is never acquired) returns
    None.
    Args:
        boundary_type (str): the boundary type (lower or upper).
        boundary_value (int): Announced boundary value.
        muddy_children_number (int): the amount of muddy children.
        children_number (int): the amount of children.
    Returns:
        int | None: First knowledge round, or None if uninformative.
    """
    if not is_bound_informative(boundary_type, boundary_value, children_number):
        return None #there is no round with knowledge
    else: #lower or upper informative bound
        return abs(muddy_children_number - boundary_value) + 1


def previous_rounds_text_standard_observation(round_number: int, boundary_type: str, boundary_value: int, muddy_children_number: int,
                                              children_number: int, puzzle_type: str) -> str:
    """
    Generate narrative text describing answers given in the rounds previous to the given round number
    (rounds < round_number).
    Args:
        round_number (int): Current round number.
        boundary_type (str): the boundary type (lower or upper).
        boundary_value (int): the boundary value.
        muddy_children_number (int): the amount of muddy children.
        children_number (int): the amount of children.
        puzzle_type (str): the puzzle type.
    Returns:
        str: Description of previous rounds (possibly empty if the current round number is 1).
    """
    cfg = PUZZLE_CONFIGS[puzzle_type]

    if round_number == 1:  # No previous rounds
        return ""

    #Setup Text Variables based on puzzle Type
    all_agents = f"all {cfg['agent_role']}"

    # 2. Calculate the first round of knowledge
    logic_type, logic_val = get_logic_type_and_value(boundary_type, boundary_value, children_number)
    s = first_round_with_some_knowledge(logic_type, logic_val, muddy_children_number, children_number)

    extreme = (muddy_children_number == 0 or muddy_children_number == children_number)

    #Handle rounds before knowledge emerges
    if s is None or round_number <= s:
        return f" In {rounds_range(1, round_number - 1)} {all_agents} answered that they {cfg['unk_txt']}."

    #Construct text for rounds after s
    txt = ""

    # Text for rounds before s (if s > 1)
    if s > 1:
        txt = f" In {rounds_range(1, s - 1)} {all_agents} answered that they {cfg['unk_txt']}."

    # Text for the critical round s
    if extreme: # If all are muddy or all are clean, everyone knows at the same time
        txt += f" In {rounds_range(s, round_number - 1)} {all_agents} answered that they {cfg['known_txt']}."
        return txt
    else:
        if logic_type == LOWER_BOUND_TYPE: # Lower bound -> Positives know first (e.g. Muddy)
            knowers = cfg['pos_group']
            dont_knowers = cfg['neg_group']
        else: # Upper bound -> Negatives know first (e.g. Clean)
            knowers = cfg['neg_group']
            dont_knowers = cfg['pos_group']

        txt += (f" In round {s} {knowers} answered that they {cfg['known_txt']}, "
                f"while {dont_knowers} answered that they did not know.")

    # Handle rounds after convergence (if any)
    if round_number > s + 1:
        txt += f" In {rounds_range(s + 1, round_number - 1)} {all_agents} answered that they {cfg['known_txt']}."

    return txt
