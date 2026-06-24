"""Build the trace-weighted workspace-process training mix for run 0021."""

from __future__ import annotations

import json
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

OLD_ANCHOR_PATH = Path(".micro_model_agent/datasets/real_broadened_schema_anchor_training.jsonl")
WORKSPACE_PROCESS_PATH = Path(".micro_model_agent/datasets/workspace_process_scenarios.jsonl")
DIVERSE_PROCESS_PATH = Path(
    ".micro_model_agent/datasets/workspace_process_diverse_scenarios.jsonl"
)
OUTPUT_PATH = Path(
    ".micro_model_agent/datasets/trace_weighted_workspace_process_0021_training.jsonl"
)

TRACE_PROTECTOR_COUNTS = {
    "trace_final_response_training": 48,
    "trace_patch_training": 48,
    "trace_search_read_patch_training": 32,
    "trace_verification_loop_training": 32,
    "trace_failure_response_training": 16,
    "trace_unsafe_path_refusal_training": 8,
    "trace_unsafe_shell_refusal_training": 8,
}

POSITIVE_SCHEMA_COUNTS = {
    "valid_tool_call": 12,
    "verification_command": 9,
    "patch_preview": 9,
    "diff_inspection": 9,
    "documentation_grounded": 9,
}

REPAIR_REFUSAL_COUNTS = {
    "search_limit_repair": 2,
    "test_command_alias_repair": 2,
    "test_command_field_repair": 2,
    "git_diff_command_alias_repair": 2,
    "repo_read_directory_alias_repair": 1,
    "invented_patch_tool_repair": 1,
    "missing_tool_name_patch_repair": 1,
    "prose_patch_repair": 1,
}

DIVERSE_BOUNDARY_COUNTS = {
    "diverse_trace_replay_boundary": 8,
    "diverse_unsafe_boundary": 8,
    "diverse_ambiguity_clarification": 4,
    "diverse_decoy_read_search": 4,
}


def main() -> None:
    old_records = _read_jsonl(OLD_ANCHOR_PATH)
    process_records = _read_jsonl(WORKSPACE_PROCESS_PATH)
    diverse_process_records = _read_jsonl(DIVERSE_PROCESS_PATH)

    _require_count("old anchors", old_records, 360)
    _require_count("workspace process", process_records, 72)
    _require_count("diverse workspace process", diverse_process_records, 24)

    trace_protectors = _clone_category_counts(
        old_records,
        TRACE_PROTECTOR_COUNTS,
        run_label="0021-trace-protector",
    )
    positive_schema = _clone_category_counts(
        old_records,
        POSITIVE_SCHEMA_COUNTS,
        run_label="0021-positive-schema",
    )
    repair_refusal = _clone_category_counts(
        old_records,
        REPAIR_REFUSAL_COUNTS,
        run_label="0021-small-repair-refusal",
    )
    diverse_boundary = _clone_category_counts(
        diverse_process_records,
        DIVERSE_BOUNDARY_COUNTS,
        run_label="0021-diverse-boundary",
    )

    records = [
        *old_records,
        *process_records,
        *diverse_process_records,
        *trace_protectors,
        *positive_schema,
        *repair_refusal,
        *diverse_boundary,
    ]
    _require_count("0021 training mix", records, 732)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "\n".join(
            json.dumps(record, separators=(",", ":"), sort_keys=True) for record in records
        )
        + "\n",
        encoding="utf-8",
    )

    categories = Counter(_category(record) for record in records)
    kinds = Counter(record.get("kind") for record in records)
    sources = Counter(_mix_source(record) for record in records)
    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")
    print("Kinds:", dict(sorted(kinds.items())))
    print("Mix sources:", dict(sorted(sources.items())))
    print("Top categories:", dict(categories.most_common(30)))
    print("Trace protectors:", len(trace_protectors))
    print("Positive schema anchors:", len(positive_schema))
    print("Small repair/refusal anchors:", len(repair_refusal))
    print("Diverse boundary anchors:", len(diverse_boundary))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _require_count(name: str, records: list[dict[str, Any]], expected: int) -> None:
    if len(records) != expected:
        raise RuntimeError(f"{name} expected {expected} records, got {len(records)}")


def _clone_category_counts(
    records: list[dict[str, Any]],
    target_counts: dict[str, int],
    *,
    run_label: str,
) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[_category(record)].append(record)

    clones: list[dict[str, Any]] = []
    for category, count in target_counts.items():
        source_records = by_category.get(category, [])
        if not source_records:
            raise RuntimeError(f"category {category!r} is missing from source records")
        for index in range(count):
            source = source_records[index % len(source_records)]
            clones.append(_clone_for_run(source, category, index, run_label=run_label))
    return clones


def _clone_for_run(
    record: dict[str, Any],
    category: str,
    index: int,
    *,
    run_label: str,
) -> dict[str, Any]:
    clone = json.loads(json.dumps(record))
    source_id = str(record["id"])
    clone["id"] = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"micro-model-agent-{run_label}-{category}-{index}-{source_id}",
        )
    )
    clone["source"] = f"{clone.get('source', 'dataset')}_{run_label}"
    metadata = dict(clone.get("metadata", {}))
    metadata["reinforcement_source_id"] = source_id
    metadata["reinforcement_run"] = "0021"
    metadata["reinforcement_slice"] = run_label
    metadata["reinforcement_category"] = category
    clone["metadata"] = metadata
    return clone


def _category(record: dict[str, Any]) -> str:
    category = record.get("metadata", {}).get("category")
    return category if isinstance(category, str) else "uncategorized"


def _mix_source(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        reinforcement = metadata.get("reinforcement_slice")
        if isinstance(reinforcement, str):
            return reinforcement
        category = metadata.get("category")
        if isinstance(category, str) and category.startswith("diverse_"):
            return "diverse-process"
        if isinstance(category, str) and category in {
            "docs_cli_updates",
            "dry_run_patch_proposal",
            "focused_test_selection",
            "read_search_diagnosis",
            "repair_recovery",
            "unsafe_refusal_boundary",
        }:
            return "process-decision-labeled"
    return "old-anchor-0012"


if __name__ == "__main__":
    main()
