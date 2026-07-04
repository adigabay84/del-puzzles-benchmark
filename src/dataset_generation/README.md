# `src/dataset_generation/`

The puzzle generator: builds natural-language problem instances, computes their ground-truth labels
with an internal epistemic solver, and assembles them into `.jsonl` / `.csv` datasets. Also contains
the one-off scripts used to build and re-skin the datasets behind the experiments in `test_results/`.

See the [main README](../../README.md#reproducing-the-experiment-datasets) for the exact commands
used to reproduce those datasets end to end. This file explains what each module does and how the
pieces fit together. For what each of the 7 `--puzzles` narratives actually is (with icons), see
[`../README.md#the-seven-narratives`](../README.md#the-seven-narratives).

## The two observation settings

Every problem instance places the queried agent in one of two settings:

- **Classic / symmetric** (`--random_observation` and `--random_observation_extreme` both unset):
  every agent sees every other agent's status but not its own. Knowledge dynamics have a closed
  form, computed by `standard_observation_text_generator.first_round_with_some_knowledge`: after an
  informative announcement, knowledge first arises at round `s = |muddy_count - bound| + 1`, reached
  first by the agents on the announced side of the bound, with the rest following one round later.
- **Random / asymmetric** (`--random_observation` or `--random_observation_extreme`): each agent
  sees an explicit, arbitrary subset of the others (a 0/1 visibility matrix). There's no closed
  form, so `random_obs_text_generator.get_knowledge_history` runs an explicit possible-worlds
  simulation: starting from all status assignments consistent with the announcement, each round it
  computes which agents' status is pinned down across every world still consistent with what they
  see, then eliminates any world whose induced "who would know" pattern doesn't match what was
  actually observed. This repeats until it stabilizes or every world-agent status is determined.
  `--random_observation_extreme` additionally hides the queried agent from itself, drops
  uninformative bounds ("at least 0" / "at most n"), and keeps only scenarios that take at least 3
  rounds to resolve.

Both settings are also implemented as a standalone simulations in
`src/utils/solver_runner_with_display.py`, which prints the full round-by-round elimination trace for
a single hardcoded scenario (can be modified inside the main variables).

## Files

### `dataset_generator.py`

The core generator. Key entry points:

- **`build_problem(...)`** &mdash; constructs one problem instance (dict with `premise`,
  `hypothesis`, and all puzzle parameters) for a fully specified scenario (agent count, muddy count,
  round, boundary, which agent is queried, its true status, narrative, and observation setup). Calls
  `internal_solver.get_solver_label` for the ground-truth label, and the appropriate
  `generate_premise` / `generate_hypothesis` text builders.
- **`build_all_possible_problems(...)`** &mdash; sweeps every valid combination of agent count,
  muddy count, boundary type/value, and round number for the requested narratives, generating one
  `build_problem` call per valid combination. For the random-observation settings, samples matrices
  via `get_valid_scenarios` (20 candidate matrices per combination when
  `random_observation_extreme` is set, to find ones satisfying its extra constraints).
- **`pipeline(problems, random_observation_extreme)`** &mdash; converts the list of problem dicts to
  a `DataFrame`, deduplicates on `(premise, hypothesis)` when `random_observation_extreme` is set,
  and normalizes whitespace/terminal punctuation in the generated text.
- **`generate(...)`** &mdash; the function `src.main`'s `generate` subcommand calls: runs
  `build_all_possible_problems` + `pipeline` and writes the result to
  `<dataset_path>.jsonl` and `<dataset_path>.csv`.

Every generated dataset row has these columns: `premise`, `hypothesis`, `setup` (`"forehead"` or
`"random"`), `solver_label` (0/1 ground truth), `children_number`, `muddy_children_number`,
`boundary_type`, `boundary_value`, `n_announcements`, `hypothesis_depth`, `round_number`,
`child_index`, `is_child_muddy`, `puzzle_type`, `special_name`, `observation_vector`, `world_mask`.

### `internal_solver.py`

`get_solver_label(...)` &mdash; the single source of ground truth. For the random-observation
setting it just reads off the precomputed `history` (per-round knower sets); for the classic setting
it applies the closed-form round formula directly.

### `boundary_utils.py`

Maps the four user-facing boundary phrasings (`lower`, `upper`, `at_least_clean`,
`not_less_than`) onto the two internal logic types (lower/upper bound on the *muddy* count &mdash;
`at_least_clean` is converted to an equivalent upper bound: "at least *q* clean" &hArr; "at most
*n&minus;q* muddy"), decides whether a given bound is informative at all ("at least 0" and "at most
*n*" never produce knowledge), and renders the announcement text itself.

### `standard_observation_text_generator.py` / `random_obs_text_generator.py`

The closed-form (symmetric) and possible-worlds (asymmetric) implementations described above, plus
their matching "what happened in previous rounds" natural-language summary generators
(`previous_rounds_text_standard_observation` / `previous_rounds_text_random_observation`), consumed
by `generate_hypothesis` in `dataset_generator.py`.

### `asymmetric_dataset_generation_script.py`

A fixed-configuration script used to build the seed asymmetric dataset for the
Olympic Games narrative: `n=10` agents, queried agent 0, lower-bound announcement `q=8`, random seed
`1`. For each valid `(muddy_count, is_muddy)` combination, it samples up to 10,000 random
observation matrices (agent 0 excluded from seeing itself) and keeps scenarios whose round of inference is exactly 3 
(i.e. the queried agent needs 3 rounds of public reasoning before it can deduce
its status) capping each combination at
`ceil(100 / num_valid_combinations)` so the 100 kept scenarios are spread roughly evenly across
muddy counts. Deduplicates on `(premise, hypothesis)`.

```bash
python -m src.dataset_generation.asymmetric_dataset_generation_script
```

Writes `asymmetric_inference_olympic_games.jsonl` / `.csv` to the current working directory.

### `convert_to_different_settings_symmetric.py` / `convert_to_different_settings_asymmetric.py`

One-off "re-skinning" scripts: take an already-generated dataset and rebuild the *identical*
scenarios (same agent/muddy counts, boundary, round, observation matrices, and ground-truth labels)
under a different `--puzzle_type`, i.e. the same logic wrapped in a different story.

- **Symmetric version**: since the classic setup carries no observation matrix, it simply calls
  `build_problem` again with each source row's stored parameters and the new `puzzle_type`.
- **Asymmetric version**: the observation matrix isn't stored as a column, so it's recovered by
  regex-parsing it back out of the source row's `hypothesis` text (it's rendered verbatim there as
  `observation matrix:\n[...]\n...`), the knowledge history is recomputed from that matrix, and then
  `build_problem` is called with the new `puzzle_type`. **This means the asymmetric converter only
  works on asymmetric (random-observation) source datasets** &mdash; the matrix must actually be
  present in the hypothesis text.

Both scripts share the same CLI:

```bash
python -m src.dataset_generation.convert_to_different_settings_symmetric \
    --srcs path/to/source1.jsonl [path/to/source2.jsonl ...] \
    --dst  path/to/output_without_extension \
    --puzzle_type olympic_games   # any key in PUZZLE_CONFIGS; default: olympic_games
```

(`convert_to_different_settings_asymmetric.py` takes the identical `--srcs` / `--dst` /
`--puzzle_type` flags.) `--srcs` accepts multiple files, concatenating their rows before conversion.
Output is written to `<dst>.jsonl` and `<dst>.csv`.
