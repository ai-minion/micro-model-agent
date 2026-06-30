"""Staged workspace reasoning evaluation for coding-agent behavior."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from micro_model_agent.application.ports import ModelProvider
from micro_model_agent.domain.contracts import EvaluationResult
from micro_model_agent.domain.datasets import DatasetExample
from micro_model_agent.infrastructure.dataset_metadata import tool_profile_for_example
from micro_model_agent.infrastructure.evaluation_response_parsing import (
    json_object_from_response,
)
from micro_model_agent.infrastructure.tools.catalog import TOOL_ARGUMENT_CONTRACTS

STAGE_NAMES = (
    "read_search",
    "diagnosis",
    "patch_proposal",
    "test_selection",
    "final_summary",
)

WORKSPACE_STAGED_SYSTEM_PROMPT = (
    "You are MicroModelAgent evaluating a dry-run coding task. "
    "Respond with exactly one JSON object and no markdown. "
    "Do not claim that patches were applied. "
    "Use these top-level keys: read_search, diagnosis, patch_proposal, "
    "test_selection, final_summary. "
    "read_search should name files and searches needed. "
    "diagnosis should explain the likely cause and plan before patching. "
    "patch_proposal should include changed_files and a dry-run patch sketch. "
    "test_selection should list focused commands. "
    "final_summary should summarize the dry-run proposal, files, tests, and risks."
)


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


class WorkspaceStagedEvaluationSuite:
    """Evaluate workspace reasoning in staged, dry-run-only responses."""

    def __init__(self, pass_threshold: float = 0.8, rubric_version: str = "legacy") -> None:
        if pass_threshold < 0.0 or pass_threshold > 1.0:
            raise ValueError("pass_threshold must be between 0.0 and 1.0")
        if rubric_version not in {"legacy", "v2", "auto"}:
            raise ValueError("rubric_version must be legacy, v2, or auto")
        self.pass_threshold = pass_threshold
        self.rubric_version = rubric_version

    async def evaluate_model(
        self,
        model_provider: ModelProvider,
        examples: list[DatasetExample],
    ) -> EvaluationResult:
        """Score model completions against staged workspace expectations."""

        if not examples:
            return EvaluationResult(
                passed=False,
                summary="staged workspace eval has no examples",
                score=0.0,
                details={"example_count": 0, "errors": ["dataset contains no examples"]},
            )

        scores: list[WorkspaceStagedExampleScore] = []
        for example in examples:
            raw_response = await model_provider.complete(self._prompt_for_example(example))
            scores.append(self._score_example(example, raw_response))

        overall_score = sum(score.score for score in scores) / len(scores)
        stage_metrics = self._stage_metrics(scores)
        return EvaluationResult(
            passed=overall_score >= self.pass_threshold,
            summary=(
                f"staged workspace eval scored {overall_score:.2f} "
                f"over {len(scores)} example(s)"
            ),
            score=overall_score,
            details={
                "example_count": len(scores),
                "pass_threshold": self.pass_threshold,
                "rubric_version": self.rubric_version,
                "metrics": {
                    "parse_success_rate": self._parse_success_rate(scores),
                    **stage_metrics,
                },
                "category_metrics": self._category_metrics(scores),
                "examples": [score.as_record() for score in scores],
            },
        )

    def _prompt_for_example(self, example: DatasetExample) -> str:
        return (
            f"<|system|>\n{WORKSPACE_STAGED_SYSTEM_PROMPT}\n"
            f"<|user|>\n{json.dumps(workspace_staged_prompt_payload(example), sort_keys=True)}\n"
            "<|assistant|>\n"
        )

    def _score_example(
        self,
        example: DatasetExample,
        raw_response: str,
    ) -> WorkspaceStagedExampleScore:
        tool_profile = tool_profile_for_example(
            example,
            default_available_tools=list(TOOL_ARGUMENT_CONTRACTS),
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

    def _parse_success_rate(self, scores: list[WorkspaceStagedExampleScore]) -> float:
        return sum(1.0 for score in scores if score.parse_success) / len(scores)

    def _stage_metrics(self, scores: list[WorkspaceStagedExampleScore]) -> dict[str, float]:
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

    def _category_metrics(
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
                "parse_success_rate": self._parse_success_rate(category_scores),
                **self._stage_metrics(category_scores),
            }
            for category, category_scores in sorted(grouped_scores.items())
        }

    def _category(self, example: DatasetExample) -> str | None:
        category = example.metadata.get("category")
        return category if isinstance(category, str) else None


def build_workspace_staged_review_records(
    *,
    examples: list[DatasetExample],
    reports: list[tuple[Path, EvaluationResult]],
    simple_failure_threshold: float = 0.4,
    auto_accept_threshold: float = 0.95,
) -> list[dict[str, Any]]:
    """Build JSON-ready review records from staged eval reports."""

    report_examples = [_examples_by_id(report) for _, report in reports]
    records: list[dict[str, Any]] = []
    for example in examples:
        result_records = []
        for (report_path, report), examples_by_id in zip(
            reports,
            report_examples,
            strict=True,
        ):
            result = examples_by_id.get(str(example.id))
            result_records.append(
                {
                    "report_path": str(report_path),
                    "run_id": _metadata_value(report, "run_id"),
                    "provider": _metadata_value(report, "provider"),
                    "model": _metadata_value(report, "model"),
                    "base_model": _metadata_value(report, "base_model"),
                    "adapter_path": _metadata_value(report, "adapter_path"),
                    "score": result.get("score") if result else None,
                    "parse_success": result.get("parse_success") if result else False,
                    "stages": result.get("stages") if result else [],
                    "raw_response": result.get("raw_response") if result else "",
                    "parsed_response": result.get("parsed_response") if result else None,
                    "missing_from_report": result is None,
                }
            )

        auto_triage = _auto_triage_record(
            result_records,
            simple_failure_threshold=simple_failure_threshold,
            auto_accept_threshold=auto_accept_threshold,
        )
        records.append(
            {
                "example_id": str(example.id),
                "category": _category_for_review(example),
                "goal": example.input.get("goal", ""),
                "workspace_files": example.input.get("workspace_files", {}),
                "candidate_files": example.input.get("candidate_files", []),
                "observations": example.input.get("observations", []),
                "constraints": example.input.get("constraints", []),
                "expected_stages": example.target.get("stages", {}),
                "gold_response": example.target.get("gold_response"),
                "model_results": result_records,
                "auto_triage": auto_triage,
                "review": {
                    "decision": "unreviewed",
                    "notes": None,
                },
            }
        )
    return records


class LocalWorkspaceStagedReviewBuilder:
    """Adapter for building staged workspace review records."""

    def build_workspace_staged_review_records(
        self,
        *,
        examples: list[DatasetExample],
        reports: list[tuple[Path, EvaluationResult]],
        simple_failure_threshold: float = 0.4,
        auto_accept_threshold: float = 0.95,
    ) -> list[dict[str, Any]]:
        """Build JSON-ready staged workspace review records."""

        return build_workspace_staged_review_records(
            examples=examples,
            reports=reports,
            simple_failure_threshold=simple_failure_threshold,
            auto_accept_threshold=auto_accept_threshold,
        )


class LocalWorkspaceStagedReviewQueueWriter:
    """Filesystem adapter for staged workspace review queues."""

    def write_workspace_staged_review_records(
        self,
        path: Path,
        records: list[dict[str, Any]],
    ) -> None:
        """Write staged workspace review records as JSONL."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, sort_keys=True))
                file.write("\n")


def _examples_by_id(report: EvaluationResult) -> dict[str, dict[str, Any]]:
    examples = report.details.get("examples")
    if not isinstance(examples, list):
        return {}
    examples_by_id: dict[str, dict[str, Any]] = {}
    for example in examples:
        if not isinstance(example, dict):
            continue
        example_id = example.get("example_id")
        if isinstance(example_id, str):
            examples_by_id[example_id] = example
    return examples_by_id


def _metadata_value(report: EvaluationResult, key: str) -> object:
    metadata = report.details.get("evaluation_metadata")
    if not isinstance(metadata, dict):
        return None
    return metadata.get(key)


def _auto_triage_record(
    result_records: list[dict[str, Any]],
    *,
    simple_failure_threshold: float,
    auto_accept_threshold: float,
) -> dict[str, Any]:
    scores = [
        score
        for result in result_records
        if isinstance((score := result.get("score")), int | float)
    ]
    if not scores:
        return {
            "decision": "needs_human_review",
            "reason": "no scorable model results were found",
            "best_score": None,
        }

    best_score = max(float(score) for score in scores)
    if best_score >= auto_accept_threshold:
        return {
            "decision": "auto_accept_candidate",
            "reason": f"best score {best_score:.2f} reached auto-accept threshold",
            "best_score": best_score,
        }
    if best_score <= simple_failure_threshold:
        return {
            "decision": "auto_reject_simple_failure",
            "reason": f"best score {best_score:.2f} was at or below simple-failure threshold",
            "best_score": best_score,
        }
    return {
        "decision": "needs_human_review",
        "reason": f"best score {best_score:.2f} needs judgment",
        "best_score": best_score,
    }


def _category_for_review(example: DatasetExample) -> str | None:
    category = example.metadata.get("category")
    return category if isinstance(category, str) else None


def workspace_staged_prompt_payload(example: DatasetExample) -> dict[str, object]:
    """Build the staged workspace prompt payload shared by eval and SFT export."""

    return {
        "goal": example.input.get("goal", ""),
        "repository_context": example.input.get("repository_context", {}),
        "available_tools": tool_profile_for_example(
            example,
            default_available_tools=list(TOOL_ARGUMENT_CONTRACTS),
        )["available_tools"],
        "workspace_files": example.input.get("workspace_files", {}),
        "candidate_files": example.input.get("candidate_files", []),
        "observations": example.input.get("observations", []),
        "constraints": example.input.get("constraints", []),
    }


def is_workspace_staged_example(example: DatasetExample) -> bool:
    """Return whether an evaluation example uses the staged workspace surface."""

    return isinstance(example.input.get("workspace_files"), dict) or isinstance(
        example.target.get("gold_response"),
        dict,
    )
