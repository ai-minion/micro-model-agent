"""Implementation of the repo.write_patch tool.

Patch writing is intentionally cautious: it validates changed paths, checks the
patch with git, supports dry runs, and can require explicit approval before
modifying files.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from micro_model_agent.repository_ops.infrastructure.contracts import (
    RepoWritePatchRequest,
    RepoWritePatchResult,
    ToolError,
)
from micro_model_agent.repository_ops.infrastructure.paths import (
    RepositoryPathError,
    RepositoryRoot,
)


class RepoWritePatchTool:
    """Safely preview, validate, and optionally apply unified diffs."""

    def __init__(
        self,
        repository_root: str | Path,
        git_executable: str = "git",
        timeout_seconds: int = 30,
    ) -> None:
        self.repository = RepositoryRoot(repository_root)
        self.git_executable = git_executable
        self.timeout_seconds = timeout_seconds

    def run(self, request: RepoWritePatchRequest) -> RepoWritePatchResult:
        patch = self._normalized_patch(request.patch)
        changed_files, path_errors = self._changed_files(patch)
        errors = path_errors

        if not errors:
            # If the caller expected a specific set of files, reject patches that
            # touch anything else.
            expected_errors = self._expected_changed_file_errors(
                expected=request.expected_changed_files,
                actual=changed_files,
            )
            errors.extend(expected_errors)

        if errors:
            return RepoWritePatchResult(
                ok=False,
                dry_run=request.dry_run,
                applied=False,
                preview=patch,
                changed_files=changed_files,
                errors=errors,
            )

        # git apply --check validates the patch without changing files.
        check = self._git_apply(patch, check=True)
        if check.returncode != 0:
            return RepoWritePatchResult(
                ok=False,
                dry_run=request.dry_run,
                applied=False,
                preview=patch,
                changed_files=changed_files,
                errors=[
                    ToolError(
                        code="patch_check_failed",
                        message=check.stderr.strip() or "git apply --check failed",
                        details={"returncode": check.returncode},
                    )
                ],
            )

        if request.dry_run:
            # A dry run stops after validation and returns the patch preview.
            return RepoWritePatchResult(
                ok=True,
                dry_run=True,
                applied=False,
                preview=patch,
                changed_files=changed_files,
            )

        if request.require_approval:
            # This is a safety rail for model-generated patches. The caller must
            # opt out of approval before real writes can happen.
            return RepoWritePatchResult(
                ok=False,
                dry_run=False,
                applied=False,
                preview=patch,
                changed_files=changed_files,
                errors=[
                    ToolError(
                        code="approval_required",
                        message="patch application requires explicit approval",
                        details={"changed_files": changed_files},
                    )
                ],
            )

        apply_result = self._git_apply(patch, check=False)
        if apply_result.returncode != 0:
            return RepoWritePatchResult(
                ok=False,
                dry_run=False,
                applied=False,
                preview=patch,
                changed_files=changed_files,
                errors=[
                    ToolError(
                        code="patch_apply_failed",
                        message=apply_result.stderr.strip() or "git apply failed",
                        details={"returncode": apply_result.returncode},
                    )
                ],
            )

        return RepoWritePatchResult(
            ok=True,
            dry_run=False,
            applied=True,
            preview=patch,
            changed_files=changed_files,
        )

    def _normalized_patch(self, patch: str) -> str:
        """Normalize patch: fix @@ hunk headers and ensure trailing newline.

        Rewrites every hunk header to have correct line counts derived from the
        hunk content.  Models frequently emit headers like "@@ -16,0 +17 @@"
        (missing the new-file count) which causes git apply to silently truncate
        the hunk.  Bare "@@" headers (no numbers at all) are also rewritten.
        """
        import re

        lines = patch.split("\n")
        result = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if re.match(r"^@@", line):
                # Scan ahead to count removed, added, and context lines in hunk.
                removed = added = context = 0
                j = i + 1
                while j < len(lines):
                    if re.match(r"^@@|^diff |^index |^--- |^\+\+\+ ", lines[j]):
                        break
                    if lines[j].startswith("-"):
                        removed += 1
                    elif lines[j].startswith("+"):
                        added += 1
                    elif lines[j].startswith(" "):
                        context += 1
                    # lines starting with "\" (no newline marker) are skipped
                    j += 1
                old_count = removed + context
                new_count = added + context
                m = re.match(r"^@@ (-\d+)(?:,\d+)? (\+\d+)(?:,\d+)? @@", line)
                if m:
                    old_minus = m.group(1)  # e.g. "-16"
                    old_plus = m.group(2)   # e.g. "+17"
                    result.append(f"@@ {old_minus},{old_count} {old_plus},{new_count} @@")
                else:
                    result.append(f"@@ -1,{old_count} +1,{new_count} @@")
            else:
                result.append(line)
            i += 1
        patch = "\n".join(result)
        return patch if patch.endswith("\n") else f"{patch}\n"

    def _changed_files(self, patch: str) -> tuple[list[str], list[ToolError]]:
        """Extract changed file paths from unified diff headers."""

        paths: set[str] = set()
        errors: list[ToolError] = []

        for line in patch.splitlines():
            if not (line.startswith("--- ") or line.startswith("+++ ")):
                continue
            raw_path = self._clean_diff_path(line[4:])
            if raw_path == "/dev/null":
                continue
            try:
                resolved = self.repository.resolve_file(raw_path)
                paths.add(self.repository.relative_path(resolved))
            except RepositoryPathError as exc:
                errors.append(
                    ToolError(
                        code="unsafe_patch_path",
                        message=str(exc),
                        details={"path": raw_path},
                    )
                )

        if not paths and not errors:
            errors.append(
                ToolError(
                    code="no_changed_files",
                    message="patch does not contain changed file headers",
                )
            )
        return sorted(paths), errors

    def _expected_changed_file_errors(
        self,
        expected: list[str],
        actual: list[str],
    ) -> list[ToolError]:
        """Compare requested changed files with the files found in the patch."""

        if not expected:
            return []

        safe_expected: set[str] = set()
        errors: list[ToolError] = []
        for path in expected:
            try:
                resolved = self.repository.resolve_file(path)
                safe_expected.add(self.repository.relative_path(resolved))
            except RepositoryPathError as exc:
                errors.append(
                    ToolError(
                        code="unsafe_expected_path",
                        message=str(exc),
                        details={"path": path},
                    )
                )

        if errors:
            return errors

        actual_set = set(actual)
        if safe_expected != actual_set:
            errors.append(
                ToolError(
                    code="changed_files_mismatch",
                    message="patch changed files do not match expected_changed_files",
                    details={
                        "expected": sorted(safe_expected),
                        "actual": sorted(actual_set),
                    },
                )
            )
        return errors

    def _clean_diff_path(self, raw_path: str) -> str:
        """Strip diff prefixes and timestamps from a file header path."""

        path = raw_path.strip().split("\t", maxsplit=1)[0].split(" ", maxsplit=1)[0]
        if path.startswith("a/") or path.startswith("b/"):
            return path[2:]
        return path

    def _git_apply(self, patch: str, check: bool) -> subprocess.CompletedProcess[str]:
        """Run git apply or git apply --check with the patch on stdin."""

        args = [self.git_executable, "apply", "--whitespace=nowarn", "--unidiff-zero"]
        if check:
            args.append("--check")
        env = os.environ.copy()
        ceiling = str(self.repository.root.parent)
        existing_ceiling = env.get("GIT_CEILING_DIRECTORIES")
        env["GIT_CEILING_DIRECTORIES"] = (
            f"{existing_ceiling}{os.pathsep}{ceiling}" if existing_ceiling else ceiling
        )
        try:
            return subprocess.run(
                args,
                cwd=self.repository.root,
                env=env,
                input=patch,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                shell=False,
                check=False,
            )
        except FileNotFoundError as exc:
            return subprocess.CompletedProcess(
                args=args,
                returncode=127,
                stdout="",
                stderr=str(exc),
            )
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(
                args=args,
                returncode=124,
                stdout=_process_output_text(exc.stdout),
                stderr=f"git apply timed out after {self.timeout_seconds} seconds",
            )


def _process_output_text(output: bytes | str | None) -> str:
    """Normalize subprocess timeout output to text."""

    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output
