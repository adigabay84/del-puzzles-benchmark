"""
Shared constants for the puzzle framework.

Central definitions used across dataset generation, testing, and analysis:

  - Boundary types (lower/upper plus the "at least q clean" and
    "not less than" surface variants) and their announcement prefixes.
  - Puzzle type identifiers for the seven narratives (three classic,
    four story variations).
  - SPECIAL_NAME_MAP: real-world persona (name, description) per puzzle
    type, keyed by the queried agent's status.
  - COLUMNS: canonical column order for the evaluation result CSVs.
  - Canonical answer strings (Yes / No / I don't know) and
    RESPONSE_FORMAT, the regex accepting exactly one canonical answer
    (tolerating surrounding whitespace, a trailing period, and curly or
    straight apostrophes in "don't").
  - rounds_range: formatting helper for round-range phrases.
"""


import re


def rounds_range(a, b):
    """
    Format a round range like 'round 3' or 'rounds 2-5'.
    Args:
        a (int): Start round (inclusive).
        b (int): End round (inclusive).
    Returns:
        str: Empty if a>b; otherwise 'round a' or 'rounds a-b'.
    """
    if a > b:
        return ""
    return f"round {a}" if a == b else f"rounds {a}-{b}"


#Bound type constants
LOWER_BOUND_TYPE = "lower"
UPPER_BOUND_TYPE = "upper"
AT_LEAST_CLEAN_TYPE = "at_least_clean"
NOT_LESS_THAN_MUDDY_TYPE = "not_less_than"

#boundary prefix
LOWER_BOUND_PREFIX = "at least"
UPPER_BOUND_PREFIX = "at most"
NOT_LESS_THAN_PREFIX = "not less than"

# Puzzle type constants
MUDDY_CHILDREN_PUZZLE = "muddy_children"
WISE_MEN_PUZZLE = "wise_men"
BLUE_EYED_ISLANDERS_PUZZLE = "blue_eyed_islanders"
OLYMPIC_GAMES_PUZZLE = "olympic_games"
SINGING_CONTEST_PUZZLE = "singing_contest"
HEALTH_SCREENING_PUZZLE = "health_screening"
SAFETY_INSPECTION_PUZZLE = "safety_inspection"

SPECIAL_NAME_MAP = {
    OLYMPIC_GAMES_PUZZLE: {
        True: ("Lizzo", "the famous singer"), # Made it
        False: ("Simone Biles", "the famous Olympic gymnast")  # Didn't make it
    },
    SINGING_CONTEST_PUZZLE: {
        True: ("Marcel Marceau", "the famous mime artist"),  # Made it (V)
        False: ("Taylor Swift", "the famous singer")  # Didn't make it (X)
    }
}

#Columns for csv files
COLUMNS = [
    "prompt", "puzzle type", "number of agents - n", "number of positive agents - k", "boundary type", "boundary Value - q",
    "round number - j", "order", "agent index", "is agent positive", "agent special name", "observation vector", "model", "version",
    "model response", "model cot", "correct response", "solver_label", "verdict", "timestamp", "temperature"
]

#expected responses
DONT_KNOW_CANONICAL = "I don't know"
YES_CANONICAL = "Yes"
NO_CANONICAL = "No"

#Regular expressions for correct answer components
SPACES = r"[ \t]*"
OPTIONAL_PERIOD = r"\.?"
YES_TOKEN = r"Yes"
NO_TOKEN = r"No"
DONT_KNOW_TOKEN = r"I don[’‘']t know"

#Regex search template for correct format response
ANSWER_CORE = rf"(?P<ans>{YES_TOKEN}|{NO_TOKEN}|{DONT_KNOW_TOKEN})"
RESPONSE_FORMAT = re.compile(
    rf"^{SPACES}{ANSWER_CORE}{SPACES}{OPTIONAL_PERIOD}{SPACES}$"
)
