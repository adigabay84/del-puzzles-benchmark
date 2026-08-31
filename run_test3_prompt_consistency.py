# run_test3_prompt_consistency.py
"""
For each of the 14 datasets (7 puzzles × 2 inference types):
  1. Build expected prompts from the JSONL.
  2. For every model's pre_process CSV, check:
     a. Row count matches JSONL.
     b. Each prompt matches the JSONL-derived prompt exactly.
  3. Check that all models within a dataset received identical prompts.
"""
import json, csv, os, sys
from collections import defaultdict

sys.path.insert(0, '.')


def prompts_from_jsonl(path: str):
    prompts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            prompts.append(f"{row['premise']}\n{row['hypothesis']}\n")
    return prompts


def prompts_from_csv(path: str):
    prompts = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            prompts.append(row['prompt'])
    return prompts


def compare(reference: list, candidate: list, label: str):
    issues = []
    if len(reference) != len(candidate):
        issues.append(f"row count: expected {len(reference)}, got {len(candidate)}")
        return issues
    mismatches = [(i, r, c) for i, (r, c) in enumerate(zip(reference, candidate))
                  if r.strip() != c.strip()]
    if mismatches:
        issues.append(f"{len(mismatches)} prompt mismatches")
        for i, ref, cand in mismatches[:2]:   # show first 2
            issues.append(
                f"  row {i}:\n"
                f"    expected: {repr(ref[:120])}\n"
                f"    got:      {repr(cand[:120])}"
            )
    return issues


def main():
    inference_types = ['symmetric_inference', 'asymmetric_inference']
    base = 'test_results'
    total_datasets, total_issues = 0, 0

    for inf_type in inference_types:
        inf_dir = os.path.join(base, inf_type)
        if not os.path.isdir(inf_dir):
            continue

        for puzzle in sorted(os.listdir(inf_dir)):
            d = os.path.join(inf_dir, puzzle)
            if not os.path.isdir(d):
                continue

            jsonl_files = [f for f in os.listdir(d) if f.endswith('.jsonl')]
            if not jsonl_files:
                continue

            jsonl_prompts = prompts_from_jsonl(os.path.join(d, jsonl_files[0]))
            csv_files = {
                f.replace('_pre_process.csv', ''): os.path.join(d, f)
                for f in os.listdir(d) if f.endswith('_pre_process.csv')
            }
            if not csv_files:
                continue

            total_datasets += 1
            dataset_ok = True
            model_prompts = {}          # model_name → prompt list

            for model, csv_path in csv_files.items():
                csv_prompts = prompts_from_csv(csv_path)
                model_prompts[model] = csv_prompts
                issues = compare(jsonl_prompts, csv_prompts, model)
                for issue in issues:
                    print(f"  FAIL [{inf_type}/{puzzle}] {model}: {issue}")
                    dataset_ok = False
                    total_issues += 1

            # Cross-model consistency: all models must have identical prompts
            model_names = list(model_prompts.keys())
            for i in range(1, len(model_names)):
                ref_name  = model_names[0]
                cmp_name  = model_names[i]
                ref_p     = model_prompts[ref_name]
                cmp_p     = model_prompts[cmp_name]
                if len(ref_p) != len(cmp_p):
                    continue  # already caught above
                diffs = sum(1 for a, b in zip(ref_p, cmp_p) if a != b)
                if diffs:
                    print(f"  FAIL [{inf_type}/{puzzle}] {ref_name} vs {cmp_name}: "
                          f"{diffs} prompts differ")
                    dataset_ok = False
                    total_issues += 1

            label = f"{inf_type}/{puzzle}"
            n_models = len(csv_files)
            n_rows   = len(jsonl_prompts)
            print(f"{'OK  ' if dataset_ok else 'FAIL'} {label:50s} "
                  f"({n_models} models, {n_rows} rows)")

    print(f"\nDatasets checked : {total_datasets}")
    print(f"Total issues     : {total_issues}")
    if total_issues == 0:
        print("All models received identical prompts consistent with the JSONL datasets.")


if __name__ == '__main__':
    main()
