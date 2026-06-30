# Training Session Command Log

This document records the main commands used during the local trained-model
proof session and why each command was run. The examples assume project commands
run from WSL:

```bash
cd /mnt/d/Projects/code/micro-model-agent
export UV_PROJECT_ENVIRONMENT=.venv-wsl
```

## Baseline Checks

```bash
git status --short
git diff --stat
git diff --check
```

These commands check the working tree, summarize changed files, and catch
whitespace errors before running longer validation or training jobs.

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

These are the standard repository verification commands. They are run before
and after prompt/data changes so training experiments are not mixed with broken
code.

## Inspecting Failed Runs

```bash
uv run python - <<'PY'
import json
from pathlib import Path

run = Path(".micro_model_agent/training/runs/qwen-coder-7b-prompt-contract-20260618-0008")
for name in ["run.json", "synthetic-evaluation.json", "trace-evaluation.json", "promotion.json"]:
    data = json.loads((run / name).read_text())
    print("##", name)
    if name == "run.json":
        print("status", data["status"], "dataset_version", data.get("dataset_version"))
        print("train_loss", data.get("metrics", {}).get("train_loss"))
    elif "evaluation" in name:
        print("passed", data["passed"], "score", data["score"])
        print(json.dumps(data["details"].get("metrics", {}), indent=2, sort_keys=True))
        for example in data["details"].get("examples", [])[:15]:
            print("-", example.get("category"), example.get("score"), example.get("errors"))
            print("  parsed=", example.get("parsed_response"))
    else:
        print(json.dumps(data, indent=2, sort_keys=True))
PY
```

This extracts the exact failure modes from a run. It is more useful than the
top-line score because it shows whether the next iteration should target tool
selection, argument schemas, refusals, patches, or trace wording.

## Building Filtered Datasets

```bash
uv run micro-agent dataset synthesize \
  --count 350 \
  --seed 2071 \
  --output .micro_model_agent/datasets/trace_prompt_contract_trace_pool.jsonl \
  --include-category trace_final_response_training \
  --include-category trace_patch_training \
  --include-category trace_search_read_patch_training \
  --include-category trace_failure_response_training \
  --include-category trace_verification_loop_training \
  --include-category trace_unsafe_shell_refusal_training \
  --include-category trace_unsafe_path_refusal_training
```

This builds the trace-shaped portion of the training set. These examples teach
the trace evaluator response contract: final response, optional patch,
changed files, and compact tool history.

```bash
uv run micro-agent dataset synthesize \
  --count 100 \
  --seed 2072 \
  --output .micro_model_agent/datasets/trace_prompt_contract_tool_pool.jsonl \
  --include-category valid_tool_call \
  --include-category documentation_grounded \
  --include-category verification_command \
  --include-category patch_preview \
  --include-category diff_inspection
```

This builds positive safe tool-call examples. It exists to counter over-refusal
and teach the model that safe unfinished work should usually produce a typed
tool call, not a refusal or final answer.

```bash
uv run micro-agent dataset synthesize \
  --count 50 \
  --seed 2073 \
  --output .micro_model_agent/datasets/trace_prompt_contract_repair_pool.jsonl \
  --include-category safe_refusal \
  --include-category search_limit_repair \
  --include-category test_command_alias_repair \
  --include-category git_diff_command_alias_repair \
  --include-category missing_tool_name_patch_repair \
  --include-category prose_patch_repair
```

This builds a smaller repair/refusal slice. The repair examples target observed
invalid responses such as stale `command_key`, shell-shaped `git.diff.command`,
missing `tool_name`, out-of-range search limits, and prose patches.

```bash
uv run micro-agent dataset merge \
  --input .micro_model_agent/datasets/trace_prompt_contract_trace_pool.jsonl \
  --input .micro_model_agent/datasets/trace_prompt_contract_tool_pool.jsonl \
  --input .micro_model_agent/datasets/trace_prompt_contract_repair_pool.jsonl \
  --output .micro_model_agent/datasets/trace_prompt_contract_training.jsonl \
  --deduplicate-by id
```

This combines the pools into one 500-example training dataset while preserving
the selected 350/100/50 distribution.

```bash
uv run micro-agent dataset validate \
  --path .micro_model_agent/datasets/trace_prompt_contract_training.jsonl
```

Validation catches malformed dataset records, invalid tool names, invalid tool
arguments, unsafe accepted examples, missing labels, and distribution details
before training.

```bash
uv run micro-agent dataset export \
  --path .micro_model_agent/datasets/trace_prompt_contract_training.jsonl \
  --output .micro_model_agent/datasets/trace_prompt_contract_training.sft.jsonl \
  --format sft-jsonl
```

Export converts dataset records into chat-style SFT examples. During this
session, this step became especially important because it now adds
`response_contract` and sanitizes previous bad repair outputs.

## Inspecting Exported SFT Records

```bash
uv run python - <<'PY'
import json
from pathlib import Path

path = Path(".micro_model_agent/datasets/trace_prompt_contract_training.sft.jsonl")
shown = {}
for line in path.read_text().splitlines():
    record = json.loads(line)
    if record["metadata"]["kind"] == "evaluation":
        continue
    user = json.loads(record["messages"][1]["content"])
    assistant = json.loads(record["messages"][2]["content"])
    contract_type = user.get("response_contract", {}).get("type")
    key = "repair" if "previous_invalid_response" in user.get("input", {}) else contract_type
    if key not in shown:
        print("##", key)
        print(json.dumps(user, indent=2, sort_keys=True)[:1000])
        print("assistant=", json.dumps(assistant, sort_keys=True))
        shown[key] = True
    if {"tool_call", "refusal", "repair"} <= set(shown):
        break
PY
```

This spot-checks the actual training prompt surface. It is the fastest way to
confirm that safe tool-call examples forbid refusal keys, refusal examples teach
`{"refusal": "..."}`, and repair examples do not expose bad assistant-shaped
JSON for the model to copy.

## Real Adapter Training

```bash
uv run micro-agent train synthetic \
  --no-dry-run \
  --base-model=Qwen/Qwen2.5-Coder-7B-Instruct \
  --dataset=.micro_model_agent/datasets/trace_prompt_contract_training.jsonl \
  --output-dir=.micro_model_agent/training/runs/qwen-coder-7b-prompt-contract-20260618-0008 \
  --max-steps=80 \
  --batch-size=1 \
  --gradient-accumulation-steps=4 \
  --max-seq-length=1024
```

This runs a real local PEFT/LoRA training job on the 7B base model. The short
80-step run is enough to test whether a data-format change moves held-out
behavior before spending more GPU time.

## Evaluation

```bash
uv run micro-agent eval synthetic \
  --run-id qwen-coder-7b-prompt-contract-20260618-0008 \
  --base-model=Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path=.micro_model_agent/training/runs/qwen-coder-7b-prompt-contract-20260618-0008/adapter \
  --output=.micro_model_agent/training/runs/qwen-coder-7b-prompt-contract-20260618-0008/synthetic-evaluation.json
```

Synthetic evaluation checks JSON parse success, correct tool selection, valid
arguments, exact arguments, safe refusals, repair success, and accidental final
responses. During this session it exposed the over-refusal problem, then showed
that `response_contract` fixed it.

```bash
uv run micro-agent eval traces \
  --run-id qwen-coder-7b-prompt-contract-20260618-0008 \
  --base-model=Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path=.micro_model_agent/training/runs/qwen-coder-7b-prompt-contract-20260618-0008/adapter \
  --output=.micro_model_agent/training/runs/qwen-coder-7b-prompt-contract-20260618-0008/trace-evaluation.json
```

Trace evaluation checks workflow replay behavior: final-response text, patch
text, changed files, and tool history. It helped distinguish tool-history gains
from remaining exact wording and patch-diff gaps.

## Promotion Gate

```bash
uv run micro-agent promote gate \
  --run-id qwen-coder-7b-prompt-contract-20260618-0008 \
  --evaluation-report .micro_model_agent/training/runs/qwen-coder-7b-prompt-contract-20260618-0008/synthetic-evaluation.json \
  --evaluation-report .micro_model_agent/training/runs/qwen-coder-7b-prompt-contract-20260618-0008/trace-evaluation.json \
  --minimum-score 0.8
```

The promotion gate blocks adapters unless all required evaluation reports pass
the minimum score. None of the session adapters were promotable yet, but each
gate report records the evidence for the next iteration.

## Real Trace Replacement Ablations

```bash
uv run micro-agent dataset export-traces \
  --trace-path .traces/workflows.jsonl \
  --output .micro_model_agent/datasets/real_trace_tool_success_v2_review.jsonl \
  --kind evaluation \
  --workflow-status succeeded \
  --require-tool-call
```

This exports only real workflow traces that succeeded and actually exercised at
least one tool. It avoids training on failed loops, no-tool summaries, and
unreviewed trace noise.

```bash
uv run micro-agent dataset relabel \
  --path .micro_model_agent/datasets/real_trace_tool_success_v2_review.jsonl \
  --output .micro_model_agent/datasets/real_trace_tool_success_v2_curated.jsonl \
  --input-outcome needs_review \
  --input-quality unknown \
  --outcome accepted \
  --quality good \
  --reviewer-notes "Filtered succeeded real workflow trace with at least one tool call; reviewed for real-data-heavy training."
```

This turns the reviewed trace export into accepted/good training examples. The
review step matters because stored traces are operational records first and
training data only after curation.

```bash
uv run micro-agent dataset validate \
  --path .micro_model_agent/datasets/real_trace_tool_success_v2_curated.jsonl
```

Validation confirms the curated real trace examples are schema-valid before
they enter a training mix.

```bash
uv run python - <<'PY'
import copy
import json
import uuid
from pathlib import Path

real_path = Path(".micro_model_agent/datasets/real_trace_tool_success_v2_curated.jsonl")
anchor_path = Path(".micro_model_agent/datasets/trace_schema_precision_training.jsonl")
output = Path(".micro_model_agent/datasets/real_weighted_schema_anchor_training.jsonl")

real_examples = [json.loads(line) for line in real_path.read_text().splitlines() if line.strip()]
anchor_examples = [json.loads(line) for line in anchor_path.read_text().splitlines() if line.strip()]

selected = []
for copy_index in range(12):
    for example in real_examples:
        item = copy.deepcopy(example)
        original_id = item["id"]
        item["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{original_id}:real-weight:{copy_index}"))
        item["source"] = f"{item['source']}:weight-{copy_index}"
        item["metadata"] = {
            **(item.get("metadata") or {}),
            "category": "real_trace_success_training",
            "real_weight_copy": copy_index,
            "original_example_id": original_id,
        }
        selected.append(item)

anchor_categories = {
    "valid_tool_call": 5,
    "documentation_grounded": 5,
    "verification_command": 5,
    "patch_preview": 5,
    "diff_inspection": 5,
    "search_limit_repair": 5,
    "test_command_alias_repair": 5,
    "test_command_field_repair": 5,
    "git_diff_command_alias_repair": 5,
    "repo_read_directory_alias_repair": 5,
    "search_sort_alias_repair": 5,
    "invented_patch_tool_repair": 5,
    "missing_tool_name_patch_repair": 5,
    "prose_patch_repair": 5,
    "trace_patch_training": 5,
    "trace_unsafe_shell_refusal_training": 5,
    "trace_unsafe_path_refusal_training": 4,
}
used = {category: 0 for category in anchor_categories}
for example in anchor_examples:
    category = (example.get("metadata") or {}).get("category")
    if category not in anchor_categories or used[category] >= anchor_categories[category]:
        continue
    item = copy.deepcopy(example)
    item["metadata"] = {
        **(item.get("metadata") or {}),
        "anchor_reason": "schema_or_safety_gap_after_real_trace_only_ablation",
    }
    selected.append(item)
    used[category] += 1

with output.open("w", encoding="utf-8") as handle:
    for example in selected:
        handle.write(json.dumps(example, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
PY
```

This builds the real-weighted training mix used for run `0011`: 216 weighted
real trace records plus 84 schema/safety anchors. The point was to test whether
real data could dominate without dropping the schema supervision that prevented
earlier invalid JSON and argument-key failures.

```bash
uv run micro-agent dataset validate \
  --path .micro_model_agent/datasets/real_weighted_schema_anchor_training.jsonl
uv run micro-agent dataset export \
  --path .micro_model_agent/datasets/real_weighted_schema_anchor_training.jsonl \
  --output .micro_model_agent/datasets/real_weighted_schema_anchor_training.sft.jsonl \
  --format sft-jsonl
```

These commands validate the mixed dataset and inspectable SFT export before
training.

```bash
uv run micro-agent train synthetic \
  --no-dry-run \
  --base-model=Qwen/Qwen2.5-Coder-7B-Instruct \
  --dataset=.micro_model_agent/datasets/real_weighted_schema_anchor_training.jsonl \
  --output-dir=.micro_model_agent/training/runs/qwen-coder-7b-real-weighted-anchor-20260618-0011 \
  --max-steps=80 \
  --batch-size=1 \
  --gradient-accumulation-steps=4 \
  --max-seq-length=1024
```

This trained the real-weighted adapter. It completed, but evaluation showed the
real trace pool is still too narrow to duplicate heavily.

```bash
uv run micro-agent eval synthetic \
  --run-id qwen-coder-7b-real-weighted-anchor-20260618-0011 \
  --base-model=Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path=.micro_model_agent/training/runs/qwen-coder-7b-real-weighted-anchor-20260618-0011/adapter \
  --output=.micro_model_agent/training/runs/qwen-coder-7b-real-weighted-anchor-20260618-0011/synthetic-evaluation.json
uv run micro-agent eval traces \
  --run-id qwen-coder-7b-real-weighted-anchor-20260618-0011 \
  --base-model=Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path=.micro_model_agent/training/runs/qwen-coder-7b-real-weighted-anchor-20260618-0011/adapter \
  --output=.micro_model_agent/training/runs/qwen-coder-7b-real-weighted-anchor-20260618-0011/trace-evaluation.json
```

These evals scored `0.60` synthetic and `0.52` trace. The mixed run recovered
parse success and tool-history behavior versus real-only training, but repeated
real traces caused extra `repo.search` metadata fields to leak into tool-call
arguments.

```bash
uv run micro-agent loop "Preview a valid documentation title patch without applying it." \
  --available-tool repo.write_patch \
  --scripted-response '{"tool_name":"repo.write_patch","arguments":{"patch":"diff --git a/docs/training-session-command-log.md b/docs/training-session-command-log.md\n--- a/docs/training-session-command-log.md\n+++ b/docs/training-session-command-log.md\n@@ -1,5 +1,5 @@\n-# Training Session Command Log\n+# Local Training Session Command Log\n \n This document records the main commands used during the local trained-model\n proof session and why each command was run. The examples assume project commands\n run from WSL:\n","dry_run":true,"expected_changed_files":["docs/training-session-command-log.md"]},"reason":"Use repo.write_patch with a valid unified diff and dry_run=true."}' \
  --scripted-response '{"final_response":"The documentation patch preview was valid and left the file unchanged.","ok":true}'
```

This is an example of collecting a real workflow trace without changing source
files. The model response is scripted, but the loop, tool validation, dry-run
patch checking, trace capture, and final dataset export are real.

```bash
uv run micro-agent dataset export-traces \
  --trace-path .traces/workflows.jsonl \
  --output .micro_model_agent/datasets/real_trace_broadened_review.jsonl \
  --kind evaluation \
  --workflow-status succeeded \
  --require-tool-call
uv run micro-agent dataset relabel \
  --path .micro_model_agent/datasets/real_trace_broadened_review.jsonl \
  --output .micro_model_agent/datasets/real_trace_broadened_curated.jsonl \
  --input-outcome needs_review \
  --input-quality unknown \
  --outcome accepted \
  --quality good \
  --reviewer-notes "Filtered succeeded real workflow trace with at least one tool call; reviewed for broadened real-data training."
```

After broadening the trace pool, this exported and curated 22 real succeeded
tool-call traces.

```bash
uv run micro-agent train synthetic \
  --no-dry-run \
  --base-model=Qwen/Qwen2.5-Coder-7B-Instruct \
  --dataset=.micro_model_agent/datasets/real_broadened_schema_anchor_training.jsonl \
  --output-dir=.micro_model_agent/training/runs/qwen-coder-7b-real-broadened-anchor-20260619-0012 \
  --max-steps=80 \
  --batch-size=1 \
  --gradient-accumulation-steps=4 \
  --max-seq-length=1024
```

This trained the broadened-real adapter from 176 weighted real records and 184
trace/schema/safety anchors.

```bash
uv run micro-agent eval synthetic \
  --run-id qwen-coder-7b-real-broadened-anchor-20260619-0012 \
  --base-model=Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path=.micro_model_agent/training/runs/qwen-coder-7b-real-broadened-anchor-20260619-0012/adapter \
  --output=.micro_model_agent/training/runs/qwen-coder-7b-real-broadened-anchor-20260619-0012/synthetic-evaluation.json
uv run micro-agent eval traces \
  --run-id qwen-coder-7b-real-broadened-anchor-20260619-0012 \
  --base-model=Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path=.micro_model_agent/training/runs/qwen-coder-7b-real-broadened-anchor-20260619-0012/adapter \
  --output=.micro_model_agent/training/runs/qwen-coder-7b-real-broadened-anchor-20260619-0012/trace-evaluation.json
```

These evals scored `0.73` synthetic and `0.73` trace. That is the best balanced
result so far, but still below the `0.80` promotion gate. The remaining failures
are now narrow schema issues: `repo.search.limit=250`, stale
`test.run.command_key`, prompt metadata leaking as `variant_focus`, and
`repo.write_patch.max_bytes`.

```bash
uv run micro-agent dataset synthesize \
  --count 80 \
  --seed 2091 \
  --output .micro_model_agent/datasets/schema_cleanup_0013_slice.jsonl \
  --include-category search_limit_250_repair \
  --include-category test_command_key_alias_repair \
  --include-category variant_focus_argument_repair \
  --include-category write_patch_max_bytes_repair \
  --include-category search_limit_repair \
  --include-category test_command_alias_repair \
  --include-category test_command_field_repair \
  --include-category search_sort_alias_repair
uv run micro-agent dataset merge \
  --input .micro_model_agent/datasets/real_broadened_schema_anchor_training.jsonl \
  --input .micro_model_agent/datasets/schema_cleanup_0013_slice.jsonl \
  --output .micro_model_agent/datasets/real_broadened_schema_cleanup_training.jsonl \
  --deduplicate-by id
```

This built the `0013` cleanup dataset. It was intentionally narrow, but it was
still too repair-heavy: adding 80 repair examples to the 360-example `0012`
dataset shifted the adapter away from trace final-response behavior.

```bash
uv run micro-agent train synthetic \
  --no-dry-run \
  --base-model=Qwen/Qwen2.5-Coder-7B-Instruct \
  --dataset=.micro_model_agent/datasets/real_broadened_schema_cleanup_training.jsonl \
  --output-dir=.micro_model_agent/training/runs/qwen-coder-7b-real-broadened-cleanup-20260619-0013 \
  --max-steps=80 \
  --batch-size=1 \
  --gradient-accumulation-steps=4 \
  --max-seq-length=1024
```

This trained the cleanup adapter. It completed, but evals regressed to `0.68`
synthetic and `0.58` trace. The useful lesson is that schema cleanup should be
introduced with much lower weight or paired with positive safe tool-call
examples that distinguish tool request schemas from tool result fields such as
`truncated`.

```bash
uv run micro-agent dataset synthesize \
  --count 20 \
  --seed 2092 \
  --output .micro_model_agent/datasets/schema_contrast_0014_slice.jsonl \
  --include-category repo_search_request_schema_contrast \
  --include-category test_run_request_schema_contrast \
  --include-category write_patch_request_schema_contrast \
  --include-category variant_focus_request_schema_contrast
uv run micro-agent dataset merge \
  --input .micro_model_agent/datasets/real_broadened_schema_anchor_training.jsonl \
  --input .micro_model_agent/datasets/schema_contrast_0014_slice.jsonl \
  --output .micro_model_agent/datasets/real_broadened_schema_contrast_training.jsonl \
  --deduplicate-by id
```

This built `0014` from the `0012` dataset plus only 20 positive contrast
examples. It tested whether a very small schema nudge could keep `0012`'s trace
behavior while removing request/result field confusion.

```bash
uv run micro-agent train synthetic \
  --no-dry-run \
  --base-model=Qwen/Qwen2.5-Coder-7B-Instruct \
  --dataset=.micro_model_agent/datasets/real_broadened_schema_contrast_training.jsonl \
  --output-dir=.micro_model_agent/training/runs/qwen-coder-7b-real-broadened-contrast-20260619-0014 \
  --max-steps=80 \
  --batch-size=1 \
  --gradient-accumulation-steps=4 \
  --max-seq-length=1024
```

The `0014` evals scored `0.68` synthetic and `0.65` trace. This recovered some
trace behavior versus `0013`, but still regressed from `0012`. The practical
lesson is that synthetic schema nudges are not currently the bottleneck; the
next move should be broader real traces or an inference/evaluation prompt change
that makes request schemas more salient.

## Current Lessons

- Synthetic data should be reduced and filtered, not removed.
- Trace-shaped SFT examples are necessary for trace behavior.
- Raw repair `bad_output` can leak bad assistant-shaped keys into training.
- Explicit `response_contract` fixed the refusal-collapse regression.
- The next improvement target is schema precision: exact argument keys,
  avoiding unknown helper tools, and producing valid escaped JSON for patches.
- Real traces should replace synthetic examples only as the real pool broadens;
  duplicating a narrow real pool degrades generalization.
