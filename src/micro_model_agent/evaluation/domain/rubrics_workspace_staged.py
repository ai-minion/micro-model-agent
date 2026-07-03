"""Pure rubrics for staged workspace reasoning evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from micro_model_agent.dataset.domain.value_objects import DatasetExample

DEFAULT_TOOL_PROFILE_NAME = "coding-agent-v1"

STAGE_NAMES = (
    "read_search",
    "diagnosis",
    "patch_proposal",
    "test_selection",
    "final_summary",
)


__all__ = [
    "STAGE_NAMES",
    "WorkspaceStageScore",
    "WorkspaceStagedExampleScore",
    "WorkspaceStagedRubric",
    "json_object_from_response",
    "strip_markdown_fence",
]


@dataclass(frozen=True, slots=True)
class WorkspaceStageScore:
    """Score for one stage inside a workspace evaluation example."""

    name: str
    score: float
    passed: bool
    errors: tuple[str, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "passed": self.passed,
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class WorkspaceStagedExampleScore:
    """Scorecard for one staged workspace evaluation example."""

    example_id: str
    category: str | None
    raw_response: str
    parsed_response: dict[str, Any] | None
    tool_profile: dict[str, Any]
    score: float
    parse_success: bool
    stages: tuple[WorkspaceStageScore, ...]
    errors: tuple[str, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "category": self.category,
            "score": self.score,
            "parse_success": self.parse_success,
            "stages": [stage.as_record() for stage in self.stages],
            "errors": list(self.errors),
            "raw_response": self.raw_response,
            "parsed_response": self.parsed_response,
            "tool_profile": self.tool_profile,
        }


class WorkspaceStagedRubric:
    """Score staged workspace responses without model or filesystem dependencies."""

    def __init__(
        self,
        rubric_version: str = "legacy",
        *,
        default_available_tools: Sequence[str] = (),
    ) -> None:
        if rubric_version not in {"legacy", "v2", "auto"}:
            raise ValueError("rubric_version must be legacy, v2, or auto")
        self.rubric_version = rubric_version
        self.default_available_tools = tuple(default_available_tools)

    def score_example(
        self,
        example: DatasetExample,
        raw_response: str,
    ) -> WorkspaceStagedExampleScore:
        tool_profile = _tool_profile_for_example(
            example,
            default_available_tools=self.default_available_tools,
        )
        try:
            response = json_object_from_response(raw_response)
        except ValueError as exc:
            return WorkspaceStagedExampleScore(
                example_id=str(example.id),
                category=self._category(example),
                raw_response=raw_response,
                parsed_response=None,
                tool_profile=tool_profile,
                score=0.0,
                parse_success=False,
                stages=(),
                errors=(str(exc),),
            )

        target_stages, uses_v2 = self._target_stages(example)
        if not isinstance(target_stages, dict):
            return WorkspaceStagedExampleScore(
                example_id=str(example.id),
                category=self._category(example),
                raw_response=raw_response,
                parsed_response=response,
                tool_profile=tool_profile,
                score=0.0,
                parse_success=True,
                stages=(),
                errors=("target.stages must be an object",),
            )

        stages = tuple(
            self._score_stage(
                stage_name,
                target_stages.get(stage_name),
                response.get(stage_name),
                uses_v2=uses_v2,
            )
            for stage_name in STAGE_NAMES
            if stage_name in target_stages
        )
        errors: tuple[str, ...]
        if not stages:
            errors = ("target.stages contains no known stage names",)
            score = 0.0
        else:
            errors = ()
            score = sum(stage.score for stage in stages) / len(stages)

        return WorkspaceStagedExampleScore(
            example_id=str(example.id),
            category=self._category(example),
            raw_response=raw_response,
            parsed_response=response,
            tool_profile=tool_profile,
            score=score,
            parse_success=True,
            stages=stages,
            errors=errors,
        )

    def parse_success_rate(self, scores: list[WorkspaceStagedExampleScore]) -> float:
        return sum(1.0 for score in scores if score.parse_success) / len(scores)

    def stage_metrics(self, scores: list[WorkspaceStagedExampleScore]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for stage_name in STAGE_NAMES:
            stage_scores = [
                stage
                for score in scores
                for stage in score.stages
                if stage.name == stage_name
            ]
            if not stage_scores:
                continue
            metrics[f"{stage_name}_score"] = (
                sum(stage.score for stage in stage_scores) / len(stage_scores)
            )
            metrics[f"{stage_name}_pass_rate"] = (
                sum(1.0 for stage in stage_scores if stage.passed) / len(stage_scores)
            )
        return metrics

    def category_metrics(
        self,
        scores: list[WorkspaceStagedExampleScore],
    ) -> dict[str, dict[str, float]]:
        grouped_scores: dict[str, list[WorkspaceStagedExampleScore]] = {}
        for score in scores:
            grouped_scores.setdefault(score.category or "uncategorized", []).append(score)

        return {
            category: {
                "example_count": float(len(category_scores)),
                "score": sum(score.score for score in category_scores) / len(category_scores),
                "parse_success_rate": self.parse_success_rate(category_scores),
                **self.stage_metrics(category_scores),
            }
            for category, category_scores in sorted(grouped_scores.items())
        }

    def _target_stages(self, example: DatasetExample) -> tuple[object, bool]:
        if self.rubric_version in {"v2", "auto"}:
            v2_stages = example.target.get("stage_rubric_v2")
            if isinstance(v2_stages, dict):
                return v2_stages, True
            if self.rubric_version == "v2":
                return None, True
        return example.target.get("stages"), False

    def _score_stage(
        self,
        stage_name: str,
        raw_expected: object,
        raw_actual: object,
        *,
        uses_v2: bool = False,
    ) -> WorkspaceStageScore:
        if not isinstance(raw_expected, dict):
            return WorkspaceStageScore(
                name=stage_name,
                score=0.0,
                passed=False,
                errors=("expected stage must be an object",),
            )
        expected = cast(dict[str, Any], raw_expected)
        if uses_v2:
            return self._score_stage_v2(stage_name, expected, raw_actual)

        actual = raw_actual if isinstance(raw_actual, dict) else {}
        actual_text = self._text_for_matching(actual)
        checks: list[bool] = []
        errors: list[str] = []

        self._check_required_terms(
            expected.get("required_terms"),
            actual_text,
            f"{stage_name}.required_terms",
            checks,
            errors,
        )
        self._check_forbidden_terms(
            expected.get("forbidden_terms"),
            actual_text,
            f"{stage_name}.forbidden_terms",
            checks,
            errors,
        )

        if stage_name == "read_search":
            self._check_required_terms(
                expected.get("required_files"),
                actual_text,
                "read_search.required_files",
                checks,
                errors,
            )
            self._check_required_terms(
                expected.get("required_queries"),
                actual_text,
                "read_search.required_queries",
                checks,
                errors,
            )
            self._check_forbidden_terms(
                expected.get("forbidden_files"),
                actual_text,
                "read_search.forbidden_files",
                checks,
                errors,
            )
        elif stage_name == "patch_proposal":
            self._check_required_terms(
                expected.get("required_changed_files"),
                actual_text,
                "patch_proposal.required_changed_files",
                checks,
                errors,
            )
            self._check_required_terms(
                expected.get("patch_contains"),
                actual_text,
                "patch_proposal.patch_contains",
                checks,
                errors,
            )
            self._check_forbidden_terms(
                expected.get("forbidden_patch_terms"),
                actual_text,
                "patch_proposal.forbidden_patch_terms",
                checks,
                errors,
            )
        elif stage_name == "test_selection":
            self._check_required_terms(
                expected.get("required_commands"),
                actual_text,
                "test_selection.required_commands",
                checks,
                errors,
            )
            self._check_forbidden_terms(
                expected.get("forbidden_commands"),
                actual_text,
                "test_selection.forbidden_commands",
                checks,
                errors,
            )
        elif stage_name == "final_summary":
            self._check_required_terms(
                expected.get("required_summary_terms"),
                actual_text,
                "final_summary.required_summary_terms",
                checks,
                errors,
            )
            self._check_forbidden_terms(
                expected.get("forbidden_summary_terms"),
                actual_text,
                "final_summary.forbidden_summary_terms",
                checks,
                errors,
            )

        if not isinstance(raw_actual, dict):
            checks.append(False)
            errors.append(f"{stage_name} response must be an object")

        score = sum(1.0 for check in checks if check) / len(checks) if checks else 0.0
        return WorkspaceStageScore(
            name=stage_name,
            score=score,
            passed=score == 1.0,
            errors=tuple(errors),
        )

    def _score_stage_v2(
        self,
        stage_name: str,
        expected: dict[str, Any],
        raw_actual: object,
    ) -> WorkspaceStageScore:
        actual = raw_actual if isinstance(raw_actual, dict) else {}
        actual_text = self._text_for_matching(actual)
        checks: list[bool] = []
        errors: list[str] = []

        if not isinstance(raw_actual, dict):
            checks.append(False)
            errors.append(f"{stage_name} response must be an object")

        self._check_forbidden_terms(
            expected.get("forbidden_terms"),
            actual_text,
            f"{stage_name}.forbidden_terms",
            checks,
            errors,
        )

        if stage_name == "read_search":
            self._check_structured_items(
                expected.get("required_files"),
                actual.get("files"),
                "read_search.files",
                checks,
                errors,
            )
            self._check_structured_absent_items(
                expected.get("forbidden_files"),
                actual.get("files"),
                "read_search.files",
                checks,
                errors,
            )
            self._check_structured_items(
                expected.get("required_queries"),
                actual.get("queries"),
                "read_search.queries",
                checks,
                errors,
                allow_text_match=True,
            )
        elif stage_name == "diagnosis":
            self._check_decision_labels(
                expected.get("root_cause_codes"),
                actual,
                ("root_cause_code", "root_cause", "diagnosis"),
                actual_text,
                "diagnosis.root_cause_codes",
                checks,
                errors,
            )
            self._check_decision_labels(
                expected.get("expected_actions"),
                actual,
                ("expected_action", "action", "next_action"),
                actual_text,
                "diagnosis.expected_actions",
                checks,
                errors,
            )
            self._check_structured_items(
                expected.get("required_evidence_files"),
                actual.get("evidence_files"),
                "diagnosis.evidence_files",
                checks,
                errors,
            )
        elif stage_name == "patch_proposal":
            self._check_structured_items(
                expected.get("expected_changed_files"),
                actual.get("changed_files"),
                "patch_proposal.changed_files",
                checks,
                errors,
            )
            self._check_structured_absent_items(
                expected.get("forbidden_changed_files"),
                actual.get("changed_files"),
                "patch_proposal.changed_files",
                checks,
                errors,
            )
            self._check_decision_labels(
                expected.get("expected_actions"),
                actual,
                ("action", "expected_action"),
                actual_text,
                "patch_proposal.expected_actions",
                checks,
                errors,
            )
            if expected.get("dry_run_required") is True:
                dry_run = actual.get("dry_run") is True or self._contains(actual_text, "dry-run")
                not_applied = not any(
                    self._contains(actual_text, term)
                    for term in ("applied patch", "committed", "executed")
                )
                checks.append(dry_run and not_applied)
                if not dry_run:
                    errors.append("patch_proposal.dry_run missing true or dry-run wording")
                if not not_applied:
                    errors.append("patch_proposal.dry_run included applied/executed claim")
            if expected.get("concrete_patch_required") is True:
                patch_text = self._first_string(
                    actual.get("dry_run_patch"),
                    actual.get("patch"),
                    actual.get("diff"),
                )
                checks.append(bool(patch_text.strip()))
                if not patch_text.strip():
                    errors.append("patch_proposal.concrete_patch_required missing patch text")
        elif stage_name == "test_selection":
            self._check_decision_labels(
                expected.get("command_policies"),
                actual,
                ("command_policy", "policy"),
                actual_text,
                "test_selection.command_policies",
                checks,
                errors,
            )
            self._check_structured_items(
                expected.get("required_commands"),
                actual.get("commands"),
                "test_selection.commands",
                checks,
                errors,
                allow_text_match=True,
            )
            self._check_structured_absent_items(
                expected.get("forbidden_commands"),
                actual.get("commands"),
                "test_selection.commands",
                checks,
                errors,
            )
        elif stage_name == "final_summary":
            self._check_decision_labels(
                expected.get("expected_statuses"),
                actual,
                ("status", "outcome"),
                actual_text,
                "final_summary.expected_statuses",
                checks,
                errors,
            )
            self._check_structured_items(
                expected.get("required_files"),
                actual.get("changed_files"),
                "final_summary.changed_files",
                checks,
                errors,
            )
            self._check_structured_items(
                expected.get("required_tests"),
                actual.get("tests"),
                "final_summary.tests",
                checks,
                errors,
                allow_text_match=True,
            )
            if expected.get("dry_run_truthful") is True:
                not_applied = not any(
                    self._contains(actual_text, term)
                    for term in ("applied", "promoted", "executed", "deployed")
                )
                checks.append(not_applied)
                if not not_applied:
                    errors.append("final_summary.dry_run_truthful included completion claim")

        score = sum(1.0 for check in checks if check) / len(checks) if checks else 0.0
        return WorkspaceStageScore(
            name=stage_name,
            score=score,
            passed=score == 1.0,
            errors=tuple(errors),
        )

    def _check_structured_items(
        self,
        raw_expected: object,
        raw_actual: object,
        label: str,
        checks: list[bool],
        errors: list[str],
        *,
        allow_text_match: bool = False,
    ) -> None:
        expected = self._string_list(raw_expected)
        if not expected:
            return
        actual_items = self._string_list(raw_actual)
        actual_text = self._text_for_matching(raw_actual)
        matched = [
            item
            for item in expected
            if item in actual_items or (allow_text_match and self._contains(actual_text, item))
        ]
        checks.append(len(matched) == len(expected))
        for item in expected:
            if item not in matched:
                errors.append(f"{label} missing {item!r}")

    def _check_structured_absent_items(
        self,
        raw_forbidden: object,
        raw_actual: object,
        label: str,
        checks: list[bool],
        errors: list[str],
    ) -> None:
        forbidden = self._string_list(raw_forbidden)
        if not forbidden:
            return
        actual_items = self._string_list(raw_actual)
        present = [item for item in forbidden if item in actual_items]
        checks.append(not present)
        for item in present:
            errors.append(f"{label} included forbidden {item!r}")

    def _check_decision_labels(
        self,
        raw_expected: object,
        actual: dict[str, Any],
        field_names: tuple[str, ...],
        actual_text: str,
        label: str,
        checks: list[bool],
        errors: list[str],
    ) -> None:
        expected = self._string_list(raw_expected)
        if not expected:
            return

        actual_labels: list[str] = []
        for field_name in field_names:
            actual_labels.extend(self._string_list(actual.get(field_name)))
        matched = [
            item
            for item in expected
            if any(self._decision_label_matches(value, item) for value in actual_labels)
            or self._decision_label_matches(actual_text, item)
        ]
        checks.append(len(matched) == len(expected))
        for item in expected:
            if item not in matched:
                errors.append(f"{label} missing {item!r}")

    def _decision_label_matches(self, actual: str, expected: str) -> bool:
        if self._contains(actual, expected):
            return True
        expected_words = [word for word in expected.replace("_", " ").split() if word]
        if not expected_words:
            return False
        return all(self._contains(actual, word) for word in expected_words)

    def _first_string(self, *values: object) -> str:
        for value in values:
            if isinstance(value, str):
                return value
        return ""

    def _check_required_terms(
        self,
        raw_terms: object,
        actual_text: str,
        label: str,
        checks: list[bool],
        errors: list[str],
    ) -> None:
        terms = self._string_list(raw_terms)
        if not terms:
            return
        matched = [term for term in terms if self._contains(actual_text, term)]
        checks.append(len(matched) == len(terms))
        for term in terms:
            if term not in matched:
                errors.append(f"{label} missing {term!r}")

    def _check_forbidden_terms(
        self,
        raw_terms: object,
        actual_text: str,
        label: str,
        checks: list[bool],
        errors: list[str],
    ) -> None:
        terms = self._string_list(raw_terms)
        if not terms:
            return
        present = [term for term in terms if self._contains(actual_text, term)]
        checks.append(not present)
        for term in present:
            errors.append(f"{label} included forbidden {term!r}")

    def _text_for_matching(self, value: object) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True)

    def _contains(self, haystack: str, needle: str) -> bool:
        return needle.casefold() in haystack.casefold()

    def _string_list(self, value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str) and item]

    def _category(self, example: DatasetExample) -> str | None:
        category = example.metadata.get("category")
        return category if isinstance(category, str) else None


def json_object_from_response(raw_response: str) -> dict[str, Any]:
    """Extract the first JSON object from plain text or fenced markdown."""

    payload = strip_markdown_fence(raw_response.strip())
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(payload)
    except json.JSONDecodeError:
        first_brace = payload.find("{")
        if first_brace < 0:
            raise ValueError("model response must be a JSON object") from None
        try:
            parsed, _ = decoder.raw_decode(payload[first_brace:])
        except json.JSONDecodeError as exc:
            raise ValueError(str(exc)) from None

    if not isinstance(parsed, dict):
        raise ValueError("model response must be a JSON object")
    return cast(dict[str, Any], parsed)


def strip_markdown_fence(response: str) -> str:
    """Remove markdown code fences around JSON when present."""

    if not response.startswith("```"):
        return response

    lines = response.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _tool_profile_for_example(
    example: DatasetExample,
    *,
    default_available_tools: Sequence[str] = (),
) -> dict[str, Any]:
    available_tools = _available_tools(example, default_available_tools)
    profile_name = _metadata_profile_name(example)
    if profile_name is None:
        profile_name = DEFAULT_TOOL_PROFILE_NAME if default_available_tools else "custom"

    return {
        "name": profile_name,
        "tool_schema_version": example.tool_schema_version,
        "available_tools": list(available_tools),
        "tools_used": list(_tools_used(example)),
    }


def _available_tools(
    example: DatasetExample,
    default_available_tools: Sequence[str],
) -> tuple[str, ...]:
    input_tools = example.input.get("available_tools")
    if isinstance(input_tools, list) and all(isinstance(tool, str) for tool in input_tools):
        return _dedupe(input_tools)

    metadata_profile = example.metadata.get("tool_profile")
    if isinstance(metadata_profile, dict):
        metadata_tools = metadata_profile.get("available_tools")
        if isinstance(metadata_tools, list) and all(
            isinstance(tool, str) for tool in metadata_tools
        ):
            return _dedupe(metadata_tools)

    return _dedupe(default_available_tools)


def _tools_used(example: DatasetExample) -> tuple[str, ...]:
    tools: list[str] = []
    target_tool = example.target.get("tool_name")
    if isinstance(target_tool, str):
        tools.append(target_tool)

    history = example.input.get("tool_history")
    if isinstance(history, list):
        for item in history:
            if not isinstance(item, dict):
                continue
            tool_call = item.get("tool_call")
            if isinstance(tool_call, dict) and isinstance(tool_call.get("tool_name"), str):
                tools.append(tool_call["tool_name"])

    return _dedupe(tools)


def _metadata_profile_name(example: DatasetExample) -> str | None:
    metadata_profile = example.metadata.get("tool_profile")
    if not isinstance(metadata_profile, dict):
        return None
    name = metadata_profile.get("name")
    return name if isinstance(name, str) and name else None


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
