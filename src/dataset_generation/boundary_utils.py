"""
Helpers for boundary announcements.

A "boundary" is the public announcement constraining how many agents satisfy
the positive predicate (e.g. "At least k have muddy foreheads"). Four surface
boundary types are supported - lower bound, upper bound, "not less than k
muddy", and "at least q clean" - and are mapped onto two internal logic types
(lower/upper bound on the number of muddy agents), with "at least q clean"
converted to "at most n - q muddy".
"""


from src.constants import (LOWER_BOUND_TYPE, UPPER_BOUND_TYPE, LOWER_BOUND_PREFIX, UPPER_BOUND_PREFIX, AT_LEAST_CLEAN_TYPE,
                           NOT_LESS_THAN_MUDDY_TYPE, NOT_LESS_THAN_PREFIX)


def get_logic_type_and_value(boundary_type, boundary_value, children_number):
    """
    Maps the input boundary type/value to the internal 'muddy' logic type/value.
    e.g. 'At least q clean' -> 'Upper bound (at most) n-q muddy'.
    Args:
        boundary_type: the boundary type.
        boundary_value: the boundary value.
        children_number: the children number.
    Returns:
        the boundary type and value.
    """
    if boundary_type == LOWER_BOUND_TYPE:
        return LOWER_BOUND_TYPE, boundary_value
    elif boundary_type == UPPER_BOUND_TYPE:
        return UPPER_BOUND_TYPE, boundary_value
    elif boundary_type == NOT_LESS_THAN_MUDDY_TYPE:
        return LOWER_BOUND_TYPE, boundary_value
    elif boundary_type == AT_LEAST_CLEAN_TYPE: # At least q clean <=> At most (n - q) muddy
        return UPPER_BOUND_TYPE, children_number - boundary_value
    return LOWER_BOUND_TYPE, boundary_value #safety default


def is_bound_informative(boundary_type: str, boundary_value: int, children_number: int) -> bool:
    """
    Boolean function for checking whether the given boundary is informative.
    Args:
        boundary_type: the boundary type.
        boundary_value: the boundary value.
        children_number: the number of children.
    Returns: true if the boundary is informative, false otherwise.
    """
    logic_type, logic_val = get_logic_type_and_value(boundary_type, boundary_value, children_number)

    if logic_type == LOWER_BOUND_TYPE:
        return logic_val > 0      # “at least 0” is uninformative
    if logic_type == UPPER_BOUND_TYPE:
        return logic_val < children_number  # “at most n” is uninformative
    return False


def validate_boundary_consistency(boundary_type: str, boundary_value: int, muddy_children_number: int,
                                  children_number: int) -> None:
    """
    Raise ValueError if the announced boundary would be false given the actual muddy count
    (e.g. announcing "at least 5 muddy" when only 4 children are actually muddy).
    Args:
        boundary_type: the boundary type.
        boundary_value: the boundary value.
        muddy_children_number: the actual number of muddy children in this scenario.
        children_number: the number of children.
    Raises:
        ValueError: if the boundary is inconsistent with muddy_children_number.
    """
    logic_type, logic_val = get_logic_type_and_value(boundary_type, boundary_value, children_number)

    if logic_type == LOWER_BOUND_TYPE and logic_val > muddy_children_number:
        raise ValueError(
            f"Inconsistent boundary: {boundary_type!r}={boundary_value} announces at least "
            f"{logic_val} muddy, but muddy_children_number={muddy_children_number}."
        )
    if logic_type == UPPER_BOUND_TYPE and logic_val < muddy_children_number:
        raise ValueError(
            f"Inconsistent boundary: {boundary_type!r}={boundary_value} announces at most "
            f"{logic_val} muddy, but muddy_children_number={muddy_children_number}."
        )


def choose_bound_prefix(boundary_type: str) -> str:
    """
    Returns the prefix corresponding to the given boundary type.
    Args:
        boundary_type: the boundary type.
    Returns:
        the prefix corresponding to the given boundary type.
    """
    if boundary_type == LOWER_BOUND_TYPE:
        return LOWER_BOUND_PREFIX
    elif boundary_type == UPPER_BOUND_TYPE:
        return UPPER_BOUND_PREFIX
    elif boundary_type == NOT_LESS_THAN_MUDDY_TYPE:
        return NOT_LESS_THAN_PREFIX
    elif boundary_type == AT_LEAST_CLEAN_TYPE:
        return LOWER_BOUND_PREFIX
    return LOWER_BOUND_PREFIX


def boundary_text(boundary_type: str, boundary_value: int, puzzle_config) -> str:
    """
    Generate boundary announcement text for the given parameters.
    Args:
        boundary_type: the boundary type.
        boundary_value: the boundary value.
        puzzle_config: Dictionary containing puzzle-specific terms.
    Returns: a string representing the boundary announcement.
    """
    agent_role = puzzle_config["agent_role"]
    agent_single = puzzle_config["agent_single"]
    pos_action_plural = puzzle_config["pos_action"]
    neg_action_plural = puzzle_config["neg_action"]
    pos_action_singular = puzzle_config["pos_action_singular"]
    neg_action_singular = puzzle_config["neg_action_singular"]

    prefix = choose_bound_prefix(boundary_type)

    if boundary_value == 1:
        unit = f"1 {agent_single}"
        if boundary_type == AT_LEAST_CLEAN_TYPE:
            action = neg_action_singular  # e.g. "has a clean forehead"
        else:
            action = pos_action_singular
    else:
        unit = f"{boundary_value} {agent_role}"
        # Select correct plural action
        if boundary_type == AT_LEAST_CLEAN_TYPE:
            action = neg_action_plural  # e.g. "have clean foreheads"
        else:
            action = pos_action_plural  # e.g. "have muddy foreheads"

    return f"{prefix} {unit} {action}"
