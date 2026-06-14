"""Tests for the repo.write_patch tool."""

from __future__ import annotations

from pathlib import Path

from micro_model_agent.infrastructure.tools.contracts import RepoWritePatchRequest
from micro_model_agent.infrastructure.tools.repo_write_patch import RepoWritePatchTool


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _patch(old: str = "return 1", new: str = "return 2") -> str:
    return (
        "diff --git a/example.py b/example.py\n"
        "index 041b5f7..be082e7 100644\n"
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def value():\n"
        f"-    {old}\n"
        f"+    {new}\n"
    )


def test_repo_write_patch_dry_run_validates_without_applying(tmp_path: Path) -> None:
    source = tmp_path / "example.py"
    _write_text(source, "def value():\n    return 1\n")

    result = RepoWritePatchTool(tmp_path).run(
        RepoWritePatchRequest(patch=_patch(), dry_run=True)
    )

    assert result.ok is True
    assert result.applied is False
    assert result.changed_files == ["example.py"]
    assert source.read_text(encoding="utf-8") == "def value():\n    return 1\n"


def test_repo_write_patch_accepts_missing_final_patch_newline(tmp_path: Path) -> None:
    source = tmp_path / "example.py"
    _write_text(source, "def value():\n    return 1\n")

    result = RepoWritePatchTool(tmp_path).run(
        RepoWritePatchRequest(patch=_patch().rstrip("\n"), dry_run=True)
    )

    assert result.ok is True
    assert result.applied is False
    assert result.preview.endswith("\n")
    assert result.changed_files == ["example.py"]
    assert source.read_text(encoding="utf-8") == "def value():\n    return 1\n"


def test_repo_write_patch_requires_approval_before_real_apply(tmp_path: Path) -> None:
    source = tmp_path / "example.py"
    _write_text(source, "def value():\n    return 1\n")

    result = RepoWritePatchTool(tmp_path).run(
        RepoWritePatchRequest(patch=_patch(), dry_run=False, require_approval=True)
    )

    assert result.ok is False
    assert result.applied is False
    assert result.errors[0].code == "approval_required"
    assert source.read_text(encoding="utf-8") == "def value():\n    return 1\n"


def test_repo_write_patch_applies_when_approval_disabled(tmp_path: Path) -> None:
    source = tmp_path / "example.py"
    _write_text(source, "def value():\n    return 1\n")

    result = RepoWritePatchTool(tmp_path).run(
        RepoWritePatchRequest(patch=_patch(), dry_run=False, require_approval=False)
    )

    assert result.ok is True
    assert result.applied is True
    assert source.read_text(encoding="utf-8") == "def value():\n    return 2\n"


def test_repo_write_patch_honors_nested_repository_root(tmp_path: Path) -> None:
    parent = tmp_path / "outer"
    repository_root = parent / "nested"
    _write_text(repository_root / "example.py", "def value():\n    return 1\n")

    import subprocess

    subprocess.run(["git", "init"], cwd=parent, check=True, capture_output=True)

    result = RepoWritePatchTool(repository_root).run(
        RepoWritePatchRequest(patch=_patch(), dry_run=False, require_approval=False)
    )

    assert result.ok is True
    assert result.applied is True
    assert (repository_root / "example.py").read_text(encoding="utf-8") == (
        "def value():\n    return 2\n"
    )


def test_repo_write_patch_rejects_expected_changed_file_mismatch(tmp_path: Path) -> None:
    _write_text(tmp_path / "example.py", "def value():\n    return 1\n")

    result = RepoWritePatchTool(tmp_path).run(
        RepoWritePatchRequest(
            patch=_patch(),
            expected_changed_files=["other.py"],
        )
    )

    assert result.ok is False
    assert result.errors[0].code == "changed_files_mismatch"


def test_repo_write_patch_rejects_unsafe_patch_paths(tmp_path: Path) -> None:
    patch = (
        "diff --git a/../outside.py b/../outside.py\n"
        "--- a/../outside.py\n"
        "+++ b/../outside.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )

    result = RepoWritePatchTool(tmp_path).run(RepoWritePatchRequest(patch=patch))

    assert result.ok is False
    assert result.errors[0].code == "unsafe_patch_path"


def test_repo_write_patch_reports_patch_check_failure(tmp_path: Path) -> None:
    _write_text(tmp_path / "example.py", "def value():\n    return 1\n")

    result = RepoWritePatchTool(tmp_path).run(
        RepoWritePatchRequest(patch=_patch(old="missing line", new="return 2"))
    )

    assert result.ok is False
    assert result.applied is False
    assert result.errors[0].code == "patch_check_failed"


def test_repo_write_patch_can_create_new_file(tmp_path: Path) -> None:
    patch = (
        "diff --git a/new_file.py b/new_file.py\n"
        "new file mode 100644\n"
        "index 0000000..c642582\n"
        "--- /dev/null\n"
        "+++ b/new_file.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def created():\n"
        "+    return True\n"
    )

    result = RepoWritePatchTool(tmp_path).run(
        RepoWritePatchRequest(patch=patch, dry_run=False, require_approval=False)
    )

    assert result.ok is True
    assert result.applied is True
    assert result.changed_files == ["new_file.py"]
    assert (tmp_path / "new_file.py").read_text(encoding="utf-8") == (
        "def created():\n    return True\n"
    )
