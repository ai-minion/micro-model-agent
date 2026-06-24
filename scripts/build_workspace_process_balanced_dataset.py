"""Build a balanced workspace-process training mix for adapter experiments."""

from __future__ import annotations

import json
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

OLD_ANCHOR_PATH = Path(".micro_model_agent/datasets/real_broadened_schema_anchor_training.jsonl")
WORKSPACE_PROCESS_PATH = Path(".micro_model_agent/datasets/workspace_process_scenarios.jsonl")
OUTPUT_PATH = Path(
    ".micro_model_agent/datasets/real_broadened_workspace_process_balanced_0018_training.jsonl"
)


def main() -> None:
    old_records = _read_jsonl(OLD_ANCHOR_PATH)
    process_records = _read_jsonl(WORKSPACE_PROCESS_PATH)
    reinforcement = _anchor_reinforcement(old_records)
    cloned_reinforcement = [
        _clone_for_reinforcement(record, index)
        for index, record in enumerate(reinforcement, start=1)
    ]
    records = old_records + process_records + cloned_reinforcement

    OUTPUT_PATH.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")
    print("Kinds:", dict(Counter(record.get("kind") for record in records)))
    print(
        "Categories:",
        dict(
            sorted(
                Counter(
                    record.get("metadata", {}).get("category", "uncategorized")
                    for record in records
                ).items()
            )
        ),
    )
    print(f"Workspace process records: {len(process_records)}")
    print(f"Anchor reinforcement records: {len(cloned_reinforcement)}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _anchor_reinforcement(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[_category(record)].append(record)

    reinforcement: list[dict[str, Any]] = []
    for category in (
        "diff_inspection",
        "documentation_grounded",
        "patch_preview",
        "valid_tool_call",
        "verification_command",
    ):
        reinforcement.extend(by_category[category])

    for category in (
        "trace_failure_response_training",
        "trace_final_response_training",
        "trace_patch_training",
        "trace_search_read_patch_training",
        "trace_verification_loop_training",
        "trace_unsafe_path_refusal_training",
        "trace_unsafe_shell_refusal_training",
        "real_trace_broadened_training",
    ):
        reinforcement.extend(by_category[category][:2])

    for category in (
        "git_diff_command_alias_repair",
        "invented_patch_tool_repair",
        "missing_tool_name_patch_repair",
        "prose_patch_repair",
        "repo_read_directory_alias_repair",
        "search_limit_repair",
        "search_sort_alias_repair",
        "test_command_alias_repair",
    ):
        reinforcement.extend(by_category[category][:2])

    if len(reinforcement) != 72:
        raise RuntimeError(f"expected 72 reinforcement records, got {len(reinforcement)}")
    return reinforcement


def _clone_for_reinforcement(record: dict[str, Any], index: int) -> dict[str, Any]:
    clone = json.loads(json.dumps(record))
    source_id = str(record["id"])
    clone["id"] = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"micro-model-agent-0018-reinforcement-{source_id}-{index}",
        )
    )
    clone["source"] = f"{clone.get('source', 'dataset')}_reinforced_0018"
    metadata = dict(clone.get("metadata", {}))
    metadata["reinforcement_source_id"] = source_id
    metadata["reinforcement_run"] = "0018"
    clone["metadata"] = metadata
    return clone


def _category(record: dict[str, Any]) -> str:
    category = record.get("metadata", {}).get("category")
    return category if isinstance(category, str) else "uncategorized"


if __name__ == "__main__":
    main()
