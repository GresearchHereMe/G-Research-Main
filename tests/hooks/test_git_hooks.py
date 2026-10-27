"""Tests for .githooks/pre-commit and .githooks/pre-push (human-side guard)."""

import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import GITHOOKS_DIR


def _find_sh() -> str | None:
    """Locate a POSIX sh; on Windows fall back to the one bundled with Git for Windows."""
    sh = shutil.which("sh")
    if sh:
        return sh
    git = shutil.which("git")
    if git:
        bundled = Path(git).resolve().parents[1] / "bin" / "sh.exe"
        if bundled.exists():
            return str(bundled)
    return None


SH = _find_sh()


def _use_repo_hooks(repo, git):
    git(repo, "config", "core.hooksPath", str(GITHOOKS_DIR))


def test_pre_commit_blocks_main(temp_repo, git):
    _use_repo_hooks(temp_repo, git)
    result = git(temp_repo, "commit", "--allow-empty", "-m", "x")
    assert result.returncode != 0
    assert "main" in result.stderr


def test_pre_commit_allows_feature_branch(temp_repo, git):
    _use_repo_hooks(temp_repo, git)
    git(temp_repo, "checkout", "-q", "-b", "feat/x")
    result = git(temp_repo, "commit", "--allow-empty", "-m", "x")
    assert result.returncode == 0, result.stderr


def _pre_push(stdin: str) -> subprocess.CompletedProcess:
    # Run via sh explicitly: Windows cannot exec a shebang script directly.
    if SH is None:
        pytest.skip("no POSIX sh available")
    return subprocess.run(
        [SH, str(GITHOOKS_DIR / "pre-push"), "origin", "https://example.invalid/r.git"],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


SHA_A = "a" * 40
SHA_Z = "0" * 40


def test_pre_push_blocks_protected_remote_ref():
    line = f"refs/heads/feat/x {SHA_A} refs/heads/main {SHA_Z}\n"
    result = _pre_push(line)
    assert result.returncode != 0
    assert "main" in result.stderr


def test_pre_push_blocks_master():
    line = f"refs/heads/master {SHA_A} refs/heads/master {SHA_Z}\n"
    assert _pre_push(line).returncode != 0


def test_pre_push_allows_feature_ref():
    line = f"refs/heads/feat/x {SHA_A} refs/heads/feat/x {SHA_Z}\n"
    result = _pre_push(line)
    assert result.returncode == 0, result.stderr


def test_pre_push_allows_empty_input():
    assert _pre_push("").returncode == 0
