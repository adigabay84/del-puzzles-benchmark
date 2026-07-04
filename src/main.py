"""
Entry point for the puzzle framework: dataset generation and model evaluation.

Dispatches the two CLI subcommands (see src.cli_input_validation):

  - generate: delegates to the dataset generator with the parsed parameters.
  - test: runs every configured model (MODELS) over each dataset row -
    builds the prompt from premise + hypothesis, derives the expected answer
    from solver_label and the agent's status (Yes/No when the agent should
    know, "I don't know" otherwise), calls the model (Gemini via its native
    client, others via an OpenAI-compatible chat API, extracting CoT where
    supported), normalizes the first line of the response against
    RESPONSE_FORMAT, and appends a PASSED/FAILED verdict row to both output
    CSVs.

Outputs of test: <output_prefix>_pre_process.csv (raw response + CoT) and
<output_prefix>_post_process.csv (first response line, no CoT), written
incrementally row by row.
"""


from typing import Optional
from pandas import DataFrame
from src.utils.csv_utils import *
import pandas as pd
from src.cli_input_validation import *
from src.dataset_generation.dataset_generator import generate
from tqdm import tqdm
from src.models import NO_TEMPERATURE_MODELS


def get_first_line(text: str) -> str:
    """
    Return the first non-empty line in the model's response.
    Args:
        text: the text to get the first line of.

    Returns:
        the first line of text.
    """
    if not text:
        return ""
    for line in str(text).splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def try_extract_formatted_answer(text: str) -> Optional[str]:
    """
    Try to extract answer in the correct format, a 'Yes', 'No' or 'I don't know', with some optional additional spaces,
    a period at the end, or different apostrophe character in 'don't'.
    Args:
        text: the text to get the formatted answer from.

    Returns: the formatted answer if found, otherwise None.

    """
    if text is None:
        return None

    first_line = get_first_line(text).strip()
    if not first_line:
        return None

    m = RESPONSE_FORMAT.fullmatch(first_line)
    if not m:
        return None

    ans = m.group("ans")

    if ans == YES_CANONICAL:
        return YES_CANONICAL
    if ans == NO_CANONICAL:
        return NO_CANONICAL
    # if matched, the only remaining option is don't know token (any apostrophe variant).
    return DONT_KNOW_CANONICAL


def call_gemini_model(client, model_id: str, prompt: str, cot: bool):
    """
    Calls Gemini model and returns the answer + CoT if the model is a thinking model.
    Args:
        client: the Gemini client.
        model_id: the id of the model to call.
        prompt: the prompt to send.
        cot: boolean for extracting model's CoT.
    Returns:
        the model's answer + CoT if the model is a thinking model.
    """
    try:
        config = {
            "temperature": 0
        }

        if cot:
            config["thinking_config"] = {"include_thoughts": True}

        resp = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=config
        )

        cot_parts = []
        text_parts = []

        if hasattr(resp, 'candidates') and resp.candidates:
            for part in resp.candidates[0].content.parts:
                if hasattr(part, 'thought') and part.thought:
                    cot_parts.append(part.text)
                else:
                    text_parts.append(part.text)

            full_cot = "\n".join(cot_parts)
            full_text = "\n".join(text_parts)

            if not full_cot:
                return resp.text, ""
            return full_text, full_cot

        return resp.text, ""

    except Exception as e:
        print(f"Gemini Error: {e}")
        return "", ""


def call_model(model_id: str, model_cfg, prompt: str) -> tuple[str, str]:
    """
    Call a model and return its text output.
    Args:
        model_id: the id of the model to call.
        model_cfg: the configuration of the model.
        prompt: the prompt for the model.

    Returns:
        the text output of the model.
    """
    client = model_cfg.client
    family = getattr(model_cfg, "model", "")
    cot_enabled = getattr(model_cfg, "cot", False)

    #Gemini
    if family == "Gemini":
        return call_gemini_model(client, model_id, prompt, cot_enabled)

    #Together, OpenAI, Router
    kwargs = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
    }

    if cot_enabled:
        kwargs["extra_body"] = {"include_reasoning": True}

    if model_id not in NO_TEMPERATURE_MODELS:
        kwargs["temperature"] = 0

    try:
        resp = client.chat.completions.create(**kwargs)
        message = resp.choices[0].message
        content = message.content or ""

        # Extract CoT
        reasoning = (
                getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None)
                or ""
        )
        return content, str(reasoning) if reasoning else ""

    except Exception as e:
        print(f"Error calling model {model_id}: {e}")
        return "", ""


def build_prompt(row) -> str:
    """
    Create a prompt for a model.
    Args:
        row: the row containing the premise and hypothesis to construct the prompt with.

    Returns:
        the prompt for the model.
    """
    return f"{row['premise']}\n{row['hypothesis']}\n"


def save_to_csv_files(prompt: str, row: Series, model_id: str, raw_response: str, correct_response: str,
                      verdict: str, first_line: str, pre_csv_path: Path, post_csv_path: Path, cot: str = ""):
    """
    Save the test setup, output and verdict to the pre- and post-process csv files.
    Args:
        prompt: the prompt for the model.
        row: the row in the dataset file to extract the different parameters from.
        model_id: the id of the model.
        raw_response: the raw response from the model.
        correct_response: the correct response expected.
        verdict: the verdict of the test for the given model - PASSED/FAILED.
        first_line: the first line of the model's response.
        pre_csv_path: the path of the csv file for saving model's pre-process response.
        post_csv_path: the path of the csv file for saving model's post-process response.
        cot: the chain of thought of the model (if it's a thinking model).
    Returns: No return.
    """
    temperature = 1 if model_id in NO_TEMPERATURE_MODELS else 0

    # save raw response and data in the pre-process file
    pre_row = build_csv_row(
        prompt=prompt,
        row=row,
        model_id=model_id,
        model_response=raw_response,
        correct_response=correct_response,
        verdict=verdict,
        temperature=temperature,
        cot=cot
    )
    append_row(pre_csv_path, COLUMNS, pre_row)

    # save processed response and data in the post-process file
    post_row = build_csv_row(
        prompt=prompt,
        row=row,
        model_id=model_id,
        model_response=first_line,
        correct_response=correct_response,
        verdict=verdict,
        temperature=temperature,
        cot=""
    )
    append_row(post_csv_path, COLUMNS, post_row)


def validate_dataset(df: DataFrame):
    """
    Check if all required columns are present.
    Args:
        df: the dataset to validate.
    Returns: no return.
    """
    required_columns = {
        "premise", "hypothesis", "solver_label", "is_child_muddy",
        "children_number", "muddy_children_number", "boundary_type",
        "boundary_value", "round_number", "child_index", "puzzle_type", "special_name"
    }
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input dataset is missing required columns: {missing_cols}")


def test_models(dataset_path: str, output_prefix: str) -> None:
    """
    Run all models over the given dataset.
    Args:
        dataset_path: the path of the dataset to test the models over.
        output_prefix: the prefix of the output files.
    Returns: No return.
    """
    #create output paths
    base = Path(output_prefix)
    if base.suffix == ".csv" or base.suffix == ".jsonl":
        base = base.with_suffix("")
    pre_process_path = base.parent / f"{base.name}_pre_process.csv"
    post_process_path = base.parent / f"{base.name}_post_process.csv"

    df = pd.read_json(dataset_path, lines=True).reset_index(drop=True)
    validate_dataset(df)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Testing dataset"):
        #generate correct response
        if row["solver_label"] == 1:
            if row["is_child_muddy"]:
                correct_response = YES_CANONICAL
            else:
                correct_response = NO_CANONICAL
        else:  # solver_label == 0 - the child in question shouldn't know yet
            correct_response = DONT_KNOW_CANONICAL

        prompt = build_prompt(row)

        for model_id, model in MODELS.items():
            raw_response, cot = call_model(model_id, model, prompt)

            #process the model's response - search in the first line
            first_line = get_first_line(raw_response)
            extracted_response = try_extract_formatted_answer(raw_response)

            if extracted_response is None: #no correct format responses found
                verdict = "FAILED"
            else:
                verdict = "PASSED" if extracted_response == correct_response else "FAILED"

            save_to_csv_files(prompt=prompt, row=row, model_id=model_id, raw_response=raw_response,
                              correct_response=correct_response, verdict=verdict, first_line=first_line,
                              pre_csv_path=pre_process_path, post_csv_path=post_process_path, cot=cot)


def main():
    """
    Run requested command - generate or test, with the given command line arguments.
    Returns: No return.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if getattr(args, "command", None) == "generate": #user requested to generate dataset
        puzzles = args.puzzles
        boundary_types = args.boundary_types
        number_of_agents = args.number_of_agents
        dataset_path = args.dataset_path
        use_special_name = args.special_name
        random_observation = args.random_observation
        random_observation_extreme = args.random_observation_extreme
        agent_index = args.agent_index
        insert_solution_approach = args.insert_solution_approach

        generate(
            number_of_children=number_of_agents,
            boundary_types=boundary_types,
            dataset_path=dataset_path,
            puzzles=puzzles,
            child_index=agent_index,
            use_special_name=use_special_name,
            random_observation=random_observation,
            random_observation_extreme = random_observation_extreme,
            insert_solution_approach = args.insert_solution_approach
        )
    elif getattr(args, "command", None) == "test": #user requested to test models on a given dataset
        test_models(
            dataset_path=args.dataset_path,
            output_prefix=args.output_prefix
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
