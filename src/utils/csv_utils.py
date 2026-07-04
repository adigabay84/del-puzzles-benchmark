"""
CSV result logging for model evaluation runs.

Utilities for writing per-instance evaluation results to CSV incrementally
(append-per-row, so partial runs are preserved if a run crashes):

  - ensure_header:  create the file (and parent dirs) with a header row if
                    missing or empty; no-op otherwise.
  - append_row:     append one result row, ensuring the header first.
  - build_csv_row:  assemble a result row from the dataset instance
                    (puzzle parameters), the model metadata (from MODELS),
                    the model's response/CoT, the ground truth, the
                    PASSED/FAILED verdict, and a timestamp - projected onto
                    the canonical COLUMNS order, with missing fields as "".
"""


import csv
from datetime import datetime
from pathlib import Path
from pandas import Series
from src.constants import *
from src.models import MODELS


def ensure_header(file_path: Path, columns: list[str]) -> None:
    """
    Ensure a CSV file exists with the given header.
    Creates the directory (if needed) and writes a header row when the file does not exist or is empty.
    :param file_path: Path to the CSV file to create/initialize.
    :param columns: Exact column order for the CSV.
    """
    if not file_path.exists() or file_path.stat().st_size == 0:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()


def append_row(file_path: Path, columns: list[str], row: dict) -> None:
    """
    Append a row to the CSV file.
    :param file_path: the path to the CSV file.
    :param columns: the column order for the CSV.
    :param row: the row to append.
    """
    ensure_header(file_path, columns)
    with file_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(row)


def build_csv_row(prompt: str, row: Series, model_id: str, model_response: str, correct_response: str, verdict: str,
                  cot: str, temperature="default") -> dict:
    """
    Build a CSV row from the given data.
    Args:
        prompt: the prompt text.
        row: the row from the dataset to build the CSV row.
        model_id: the model id.
        model_response: the model response text.
        correct_response: the correct response text.
        verdict: the verdict text - PASSED/FAILED.
        temperature: model temperature.
        cot: the chain of thought of the response.
    Returns: the CSV row.
    """
    model_meta = MODELS[model_id]
    base = {
        "prompt": prompt,
        "puzzle type": row.get("puzzle_type"),
        "number of agents - n": row["children_number"],
        "number of positive agents - k": row["muddy_children_number"],
        "boundary type":  row["boundary_type"],
        "boundary Value - q": row["boundary_value"],
        "round number - j": row["round_number"],
        "order": row.get("hypothesis_depth", 1),
        "agent index": row["child_index"],
        "is agent positive": row["is_child_muddy"],
        "agent special name": row.get("special_name", ""),
        "observation vector": row.get("observation_vector", ""),
        "model": model_meta.model,
        "version": model_meta.version,
        "model response": model_response,
        "model cot": cot,
        "correct response": correct_response,
        "solver_label": row["solver_label"],
        "verdict": verdict,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "temperature": temperature
    }
    return {col: base.get(col, "") for col in COLUMNS}
