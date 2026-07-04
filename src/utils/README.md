# `src/utils/`

Dataset post-processing and result-CSV writing utilities shared by `src.main` and the dataset
generation scripts.

## `filter_first_deduction.py`

Filters a generated symmetric-inference dataset (which by default contains every round for every
scenario) down to exactly one row per scenario: the earliest round at which the queried agent can
deduce its status (`solver_label == 1`). Scenarios that never reach a deduction within the generated
rounds are dropped entirely. A "scenario" is identified by `puzzle_type`, `children_number`,
`muddy_children_number`, `boundary_type`, `boundary_value`, `child_index`, and `is_child_muddy`.

```bash
python -m src.utils.filter_first_deduction \
    --input  path/to/full_dataset.jsonl \
    --output path/to/first_deduction_dataset.jsonl
```

Pass `--run-generate` to run `src.main generate` first and filter its output in the same command
(uses `--number_of_agents` and `--boundary_types`, both forwarded to `generate`; default `10` and
`lower`):

```bash
python -m src.utils.filter_first_deduction \
    --run-generate \
    --number_of_agents 10 \
    --boundary_types lower \
    --input  path/to/full_dataset.jsonl \
    --output path/to/first_deduction_dataset.jsonl
```

Note `--run-generate` always calls `generate` with `--puzzles muddy_children` (the CLI default) and
`--dataset_path` set to `--input`; to filter a different narrative, run `src.main generate`
separately with the desired `--puzzles` value and pass its output as `--input` without
`--run-generate`. Writes `<output>.jsonl` and a matching `<output>.csv`.

## `csv_utils.py`

Incremental CSV writers for `src.main test`'s evaluation output,:

- `ensure_header(file_path, columns)` &mdash; creates the file (and parent directories) with a
  header row if it doesn't exist yet or is empty.
- `append_row(file_path, columns, row)` &mdash; ensures the header, then appends one row.
- `build_csv_row(...)` &mdash; assembles a result row from a dataset row, the model's metadata
  (from `src.models.MODELS`), its response/chain-of-thought, the computed ground truth, and the
  `PASSED`/`FAILED` verdict, projected onto the canonical column order in `src.constants.COLUMNS`
  (missing fields become `""`).

## `solver_runner_with_display.py`

A verbose, standalone re-implementation of the possible-worlds solver
(`src.dataset_generation.random_obs_text_generator.get_knowledge_history`), intended for
observing a single scenario each time. Running it as a script executes a
hardcoded example (`n=10`, a fixed random observation matrix, 8 muddy agents, lower bound `q=8`) and
prints, round by round: which agents currently know their status, which possible worlds get
eliminated and why, and the worlds that remain.

```bash
python -m src.utils.solver_runner_with_display
```

Edit the constants at the bottom of the file (`N_CHILDREN`, `OBS_MATRIX`, `boolean_input`,
`BOUNDARY_VAL`, `BOUNDARY_TP`) to trace a different scenario.
