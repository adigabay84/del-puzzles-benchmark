"""
Core generator for muddy-children-style epistemic puzzle datasets.

Builds natural-language problem instances for a family of puzzle narratives
(muddy children and its story variations, from PUZZLE_CONFIGS), each
consisting of a [Premise] describing the scenario and answer format, and a
[Hypothesis] posing the queried agent's question at a specific round. Ground
truth labels come from the internal epistemic solver.

Supports two observation setups:
  - Classic ("forehead"): every agent sees all others but not themselves.
  - Random: an explicit random observation matrix, with knowledge
    history computed per matrix.

Main entry points:
  - build_problem: construct a single problem record (dict).
  - build_all_possible_problems: sweep all valid parameter combinations
                              (agents, muddy counts, boundary types/values,
                              rounds) for the requested puzzle types.
  - pipeline: DataFrame post-processing.
  - generate: full run - build, clean, and save as .jsonl + .csv.

Helpers include canonical_muddy_mask (deterministic mud assignment with the
queried agent's status fixed), premise/hypothesis/example text generation,
and get_valid_scenarios (matrix sampling and validity/solvability filtering).
"""


from pathlib import Path
from pandas import DataFrame
import pandas as pd
import numpy as np
from src.dataset_generation.random_obs_text_generator import previous_rounds_text_random_observation, \
    natural_language_observation_vector, get_knowledge_history
from src.dataset_generation.internal_solver import get_solver_label
from src.dataset_generation.boundary_utils import get_logic_type_and_value, boundary_text, choose_bound_prefix
from tqdm.auto import tqdm
from sorcery import dict_of
from src.constants import (
    LOWER_BOUND_TYPE, UPPER_BOUND_TYPE, AT_LEAST_CLEAN_TYPE, SPECIAL_NAME_MAP, NO_CANONICAL,
    YES_CANONICAL
)
from src.puzzles import PUZZLE_CONFIGS, MATRIX_CORE_TEXT
from src.dataset_generation.standard_observation_text_generator import previous_rounds_text_standard_observation


def postprocess(s):
    """
    Normalize whitespace/punctuation and ensure a terminal punctuation mark.
    Args:
        s (str): Input string.
    Returns:
    str: Cleaned string ending with one of ('.', '?', '!', ':').
        """
    s=".".join(s.split('.')).replace('. .','.')
    s=" ".join(s.split(" "))
    if not s.endswith(('.', '?', '!', ':')):
        s=s+'.'
    return s


def canonical_muddy_mask(children_number, muddy_children_number, child_index, is_child_muddy):
    """
    Build a canonical mud assignment mask with the queried child assigned status (muddy/clean).
    Args:
        children_number (int): Total number of children.
        muddy_children_number (int): Total number of muddy children.
        child_index (int): Index of queried child.
        is_child_muddy (bool): Whether queried child is muddy.
    Returns:
        list[bool]: Length `children_number` mask of mud status.
    """
    mask = [False] * children_number
    # place mud on the asked child if needed
    mask[child_index] = bool(is_child_muddy)
    remaining = muddy_children_number - (1 if is_child_muddy else 0)
    for i in range(children_number):
        if i == child_index:
            continue
        if remaining <= 0:
            break
        mask[i] = True
        remaining -= 1
    return mask


def generate_example(logic_type: str, boundary_type: str, puzzle_type: str, example_matrix: str):
    """
    Generates prompt example.
    Args:
        logic_type: the logical boundary type.
        boundary_type: the boundary type.
        puzzle_type: the puzzle type.
        example_matrix: an observation matrix representation, if the setup is classic the string is empty.
    Returns:
        the example string to incorporate in the prompt.
    """
    cfg = PUZZLE_CONFIGS[puzzle_type]

    # Calculate example boundary value
    ex_q = 2 if logic_type == LOWER_BOUND_TYPE else 0

    # Determine Example Response
    if boundary_type == AT_LEAST_CLEAN_TYPE:
        ex_q = 2
        ex_response = NO_CANONICAL
    elif logic_type == LOWER_BOUND_TYPE:  # regular lower bound + not less than
        ex_response = YES_CANONICAL
    else:  # Upper
        ex_response = NO_CANONICAL

    ex_announcement = boundary_text(boundary_type, ex_q, cfg)

    return (f"[Example] For example, if there are 2 {cfg['agent_role']}, and the {cfg['authority']} "
            f"announces that {ex_announcement}, then in round 1, {example_matrix}you should respond “{ex_response}” "
            f"for both {cfg['agent_role']}.")


def generate_premise(boundary_type: str, puzzle_type: str, use_random_observation: bool = False):
    """
    Create the natural-language [Premise] section for a problem instance.
    Args:
        boundary_type (str): Boundary type.
        puzzle_type (str): Puzzle type.
        use_random_observation (bool): Whether to use random observation matrix.
    Returns:
        str: Premise text describing the scenario and answer instructions.
    """
    cfg = PUZZLE_CONFIGS[puzzle_type]

    if use_random_observation:
        # Format the generic matrix text with specific puzzle terms
        matrix_explanation = MATRIX_CORE_TEXT.format(
            agent=cfg['agent_single'],
            status=cfg['status'],
        )
        observation_text = f"{cfg['matrix_prefix']} {matrix_explanation}"
        example_matrix = "given the following observation matrix:\n[0,1]\n[1,0]\n"
    else: # Use the classic text
        observation_text = cfg['classic_observation']
        example_matrix = ""

    full_intro = f"{cfg['setup_intro']} {observation_text}"

    bound_prefix = choose_bound_prefix(boundary_type)
    action = cfg["neg_action"] if boundary_type == AT_LEAST_CLEAN_TYPE else cfg["pos_action"]

    announcement_txt = f"{bound_prefix} q {cfg['agent_role']} {action}"
    logic_type, _ = get_logic_type_and_value(boundary_type, 0, 2)
    example = generate_example(logic_type=logic_type, boundary_type=boundary_type, puzzle_type=puzzle_type,
                               example_matrix=example_matrix)

    return (f"[Premise] The following is a problem with parameters that may vary between instances:\n"
            f"{full_intro} "
            f"The {cfg['authority']} then publicly announces that {announcement_txt}. "
            f"Then, in repeated rounds, the {cfg['authority']} asks each {cfg['agent_single']} whether "
            f"{cfg['whether_knowledge']}. In each round, the {cfg['agent_role']} may rely only on "
            f"the round number, the answers from the previous rounds, and the observed "
            f"{cfg['observed_status']} of the other {cfg['agent_role']}.\n"
            f"You are one of the {cfg['agent_role']}, and in round j, you need to answer the question "
            f"“{cfg['question'][0].upper() + cfg['question'][1:]}” based on what you see, "
            f"the round number, and the answers from the previous rounds. Answer with “Yes”, “No”, or “I don't know”. "
            f"Write your answer without additional text and, on a new line, explain your reasoning.\n"
            f"{example}\n"
            f"Answer the following question:"
            )


def generate_hypothesis(round_number: int, muddy_children_number: int, children_number: int, is_child_muddy: int, child_index: int,
                        puzzle_type: str, special_name: str = None, special_description = None, use_random_observation: bool = False,
                        observation_matrix: str = None, observation_vector: list = None, world_mask: list = None) -> str:
    """
    Generate the natural-language question for the queried child at the round in question.
    Args:
        round_number (int): Current round number.
        muddy_children_number (int): the amount of muddy children.
        children_number (int): the amount of children.
        is_child_muddy (int): Whether queried child is muddy.
        child_index (int): Index of queried child.
        puzzle_type (str): the puzzle type.
        special_name (str): real world persona name to incorporate in the puzzle.
        special_description (str): real world persona description to incorporate in the puzzle.
        use_random_observation (bool): whether to use random observation.
        observation_matrix (str): description of the observations of each agent.
        observation_vector (list): description of the observations of the asked agent.
        world_mask (list): the world mask of positive/negative agent's status.
    Returns:
        str: Hypothesis text prompting.
    """
    cfg = PUZZLE_CONFIGS[puzzle_type]

    if use_random_observation:
        identity_phrase = f" you are {cfg['agent_single']} {child_index}"
        seen_phrase = f"and observe the world as described in the following observation matrix:\n{observation_matrix}\n"
        seen_phrase += natural_language_observation_vector(cfg=cfg, observation_vector=observation_vector,
                                                           world_mask=world_mask)
    else: #standard observation matrix
        identity_phrase = ""
        if is_child_muddy: # if you are muddy, you see k-1 others
            pos_seen = max(muddy_children_number - 1, 0)
        else:
            pos_seen = muddy_children_number

        if pos_seen == 1:
            seen_phrase = f"you see 1 {cfg['pos_prop_desc']}"
        else:
            seen_phrase = f"you see {pos_seen} {cfg['pos_prop_desc_plural']}"

        if 0 < pos_seen < children_number - 1:
            seen_phrase += f" and the rest {cfg['the_rest_phrase']}"

    if special_name:
        if not use_random_observation:
            identity_phrase += f" you are "
        else:
            identity_phrase += ", "
        identity_phrase += f"{special_description} {special_name},"
    else:
        identity_phrase += ""

    base_hyp = f" In round {round_number}, assuming that{identity_phrase} {seen_phrase}; "
    task_instruction = (
        f"{cfg['question']} "
        "Answer with “Yes”, “No”, or “I don't know”. Write your answer without additional text and, on a new line, explain your reasoning"
    )

    return base_hyp + task_instruction


def generate_solution_approach():
    """Adds instructions for solving if requested."""
    solution_approach = ("[Solution Approach] Build a solver using an epistemic Kripke model (possible-worlds semantics) "
                         "and use it to solve this problem.")
    return "\n" + solution_approach



def build_problem(children_number: int, muddy_children_number: int, round_number: int, child_index: int,
                          boundary_value: int, boundary_type: str, is_child_muddy: bool, puzzle_type: str, insert_solution_approach: bool,
                          use_special_name: bool = False, matrix=None, use_random_observation: bool = False, history=None):
    """
    Construct a Problem instance with NL and SMCDEL representations for the given parameter configuration.
    Args:
        children_number: the amount of children.
        muddy_children_number: the amount of muddy children.
        round_number: the current round number.
        child_index: the index of queried child.
        boundary_value: the boundary value.
        boundary_type: the boundary type (lower/upper).
        is_child_muddy: boolean for whether queried child is muddy.
        puzzle_type: the puzzle type.
        use_special_name: boolean, need to use special name or not.
        matrix: description of the observations of each agent.
        use_random_observation: boolean, need to use random observations or not.
        history: the history of the problem (who knew in each round).
        insert_solution_approach: whether to insert solution approach.

    Returns:
        Problem: Fully instantiated problem.
    """
    cfg = PUZZLE_CONFIGS[puzzle_type]

    muddy_mask = canonical_muddy_mask(children_number=children_number, muddy_children_number=muddy_children_number,
                                      child_index=child_index, is_child_muddy=is_child_muddy)
    solver_label = get_solver_label(
        children_number=children_number, muddy_children_number=muddy_children_number, round_number=round_number,
        boundary_value=boundary_value, boundary_type=boundary_type, child_index=child_index, is_child_muddy=is_child_muddy,
        use_random_observation=use_random_observation, history=history
    )

    #setup special names if requested
    name = ""
    description = ""
    if use_special_name:
        name, description = SPECIAL_NAME_MAP[puzzle_type].get(is_child_muddy)

    obs_matrix = None
    if use_random_observation and matrix is not None:
        obs_vector = matrix[child_index].tolist()
        rows_as_strings = [str(row.tolist()) for row in matrix]
        obs_matrix = "\n".join(rows_as_strings)
        indices = f"indexed by 0-{children_number - 1}, "
        setup = "random"
    else: #classic setup - sees everyone excluding themselves
        temp_vector = np.ones(children_number, dtype=int)
        temp_vector[child_index] = 0
        obs_vector = temp_vector.tolist()
        indices = ""
        setup = "forehead"

    # create premise and hypothesis
    premise = generate_premise(boundary_type=boundary_type, puzzle_type=puzzle_type, use_random_observation=use_random_observation)

    if use_random_observation:
        prev_rounds = previous_rounds_text_random_observation(
            round_number=round_number, children_number=children_number, puzzle_type=puzzle_type, history=history)
    else:
        prev_rounds = previous_rounds_text_standard_observation(
            round_number=round_number, boundary_type=boundary_type, boundary_value=boundary_value,
            muddy_children_number=muddy_children_number, children_number=children_number, puzzle_type=puzzle_type
        )

    announcement_str = f"and the {cfg['authority']} publicly announces that {boundary_text(boundary_type, boundary_value, cfg)}."

    hypothesis_text = generate_hypothesis(round_number=round_number, muddy_children_number=muddy_children_number,
                                      children_number=children_number, is_child_muddy=is_child_muddy,
                                      puzzle_type=puzzle_type,
                                      special_name=name, special_description=description,
                                      use_random_observation=use_random_observation,
                                      observation_matrix=obs_matrix, observation_vector=obs_vector,
                                      world_mask=muddy_mask, child_index=child_index)

    solution_approach = ""
    if insert_solution_approach:
        solution_approach = generate_solution_approach()

    hypothesis = (f"[Hypothesis] Suppose there are {children_number} {cfg['agent_role']} {indices}{announcement_str}"
                  f"{prev_rounds}" + hypothesis_text + solution_approach)

    return [dict_of(
                premise=premise, hypothesis=hypothesis, setup=setup, solver_label=solver_label,
                children_number=children_number, muddy_children_number=muddy_children_number, boundary_type=boundary_type,
                boundary_value=boundary_value, n_announcements=1, hypothesis_depth=1, round_number=round_number,
                child_index=child_index, is_child_muddy=is_child_muddy, puzzle_type=puzzle_type, special_name=name,
                observation_vector=str(obs_vector) if obs_vector else "", world_mask=muddy_mask
                )]


def get_valid_scenarios(children_number: int, muddy_children_number: int, boundary_value: int, boundary_type: str,
                            use_random_observation: bool, random_observation_extreme: bool, child_index: int):
    """
    Helper function to generate valid scenario tuples (matrix, index, history, is_muddy).
    Handles matrix generation, indexing logic, and extreme-mode filtering.
    """
    scenarios = []
    iterations = 20 if random_observation_extreme else 1 #how many random matrices to generate
    is_random_mode = use_random_observation or random_observation_extreme

    for _ in range(iterations):
        matrix = None

        if is_random_mode: #one of the 2 random modes, generate a random observation matrix
            matrix = np.random.randint(0, 2, (children_number, children_number))
            if random_observation_extreme:
                matrix[child_index][child_index] = 0 #if extreme, we don't want the agent to see themselves

        for is_child_muddy in (True, False):
            if ((is_child_muddy and muddy_children_number == 0) or
                    (not is_child_muddy and muddy_children_number == children_number)):
                continue

            history = None
            if is_random_mode: #calculate History (only for random modes)
                muddy_mask = canonical_muddy_mask(muddy_children_number=muddy_children_number,
                                                  children_number=children_number,
                                                  child_index=child_index,
                                                  is_child_muddy=is_child_muddy)
                real_world = np.array(muddy_mask, dtype=int)
                history = get_knowledge_history(
                    number_of_children=children_number, observation_matrix=matrix.tolist(),
                    real_world=real_world, boundary_value=boundary_value, boundary_type=boundary_type
                )

            if random_observation_extreme:
                can_solve = False
                for r in range(1, children_number + 2):
                    label = get_solver_label(
                        children_number, muddy_children_number, r, boundary_value, boundary_type,
                        child_index, is_child_muddy, True, history
                    )
                    if r < 3 and label == 1:
                        break
                    if label == 1:
                        can_solve = True
                        break

                if not can_solve:
                    continue

            scenarios.append((matrix, history, is_child_muddy))

    return scenarios


def build_all_possible_problems(number_of_children: list[int], boundary_types: tuple[str,str], puzzle_types: list[str],
                                use_special_name: bool, child_index: int, use_random_observation: bool,
                                random_observation_extreme: bool, insert_solution_approach: bool):
    """
    Generate for each configuration all possible problems.
    Args:
        number_of_children: List of agent counts (N).
        boundary_types: List of allowed boundary types.
        puzzle_types: List of allowed puzzles (Muddy, Exam, etc.).
        use_special_name: boolean, need to use special name or not.
        child_index: Index of agent to generate problems for.
        use_random_observation: boolean, need to use random observations or not.
        random_observation_extreme: boolean, need to use random observations in an extreme way or not.
        insert_solution_approach: boolean, need to insert solution approach or not.
    """
    np.random.seed(1)
    problems = []

    for puzzle_type in puzzle_types:
        for children_number in tqdm(number_of_children, desc=f"Gen {puzzle_type}"):
            for muddy_children_number in tqdm(range(0, children_number + 1), desc=f"Muddy children (N={children_number})",
                                              leave=False):
                for boundary_type in boundary_types:
                    for boundary_value in range(0, children_number + 1):
                        l_type, l_val = get_logic_type_and_value(boundary_type, boundary_value, children_number)
                        # "At least X muddy", X must be <= actual muddy count, "At most X muddy", X must be >= actual muddy count
                        if ((l_type == LOWER_BOUND_TYPE and l_val > muddy_children_number) or
                                (l_type == UPPER_BOUND_TYPE and l_val < muddy_children_number)):
                            continue

                        if random_observation_extreme:  # we want to eliminate immediate bound announcements
                            if ((l_type == LOWER_BOUND_TYPE and l_val == children_number) or
                                    (l_type == UPPER_BOUND_TYPE and l_val == 0)):
                                continue

                        # Get valid scenarios list
                        scenarios = get_valid_scenarios(
                            children_number=children_number, muddy_children_number=muddy_children_number,
                            boundary_value=boundary_value, boundary_type=boundary_type, child_index=child_index,
                            use_random_observation=use_random_observation, random_observation_extreme=random_observation_extreme
                        )

                        is_random_mode = use_random_observation or random_observation_extreme

                        for matrix, history, is_child_muddy in scenarios:
                            for round_number in range(1, children_number + 2):
                                problems += build_problem(
                                    children_number=children_number, muddy_children_number=muddy_children_number,
                                    round_number=round_number, child_index=child_index,
                                    boundary_value=boundary_value, boundary_type=boundary_type,
                                    is_child_muddy=is_child_muddy, puzzle_type=puzzle_type, use_special_name=use_special_name,
                                    matrix=matrix, use_random_observation=is_random_mode, history=history,
                                    insert_solution_approach=insert_solution_approach
                                )
    return problems


def pipeline(problems: list, random_observation_extreme: bool) -> DataFrame:
    """
    Post process dataset - remove invalid premises, arrange text.
    Args:
        problems: the list of problems to process
        random_observation_extreme: boolean, used random observations extreme or not.
    Returns: data frame after processing.
    """
    df = pd.DataFrame(problems)

    if df.empty:
        return df

    if random_observation_extreme:
        df.drop_duplicates(subset=['premise', 'hypothesis'], inplace=True)

    #post-process the text
    for col in ['premise', 'hypothesis']:
        if col not in df:
            continue
        new_col = [postprocess(t) for t in df[col]]
        df[col] = new_col

    return df


def generate(number_of_children: list[int], boundary_types: tuple[str,str], dataset_path: str,
             puzzles: list[str], use_special_name: bool, random_observation: bool,
             random_observation_extreme: bool, child_index: int, insert_solution_approach: bool):
    """
    Generate muddy children puzzles dataset for the given parameters.
    Args:
        number_of_children: the amount of children, array with possible amounts.
        boundary_types: the boundary types, upper, lower, or both.
        dataset_path: the path for the dataset.
        puzzles: the list of puzzles.
        use_special_name: boolean, need to use special name or not.
        random_observation: boolean, need to use random observations or not.
        random_observation_extreme: boolean, need to use random observations in an extreme way or not.
        child_index: the index of agent to generate problems for.
        insert_solution_approach: boolean, need to insert solution approach or not.
    Returns: the created dataset.
    """
    # build problems
    problems = build_all_possible_problems(number_of_children=number_of_children, boundary_types=boundary_types,
                                           puzzle_types=puzzles, use_special_name=use_special_name,
                                           use_random_observation=random_observation, child_index=child_index,
                                           random_observation_extreme=random_observation_extreme,
                                           insert_solution_approach=insert_solution_approach)

    # clean and arrange dataset
    df = pipeline(problems, random_observation_extreme)

    p = Path(dataset_path)
    #save as json and csv files
    df.to_json(p.with_suffix(".jsonl"), orient="records", lines=True)
    df.to_csv(p.with_suffix(".csv"), index=False)
    return df
