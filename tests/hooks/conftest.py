"""Shared fixtures for repo-guard hook tests."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = REPO_ROOT / ".claude" / "hooks" / "guard_main.py"
GITHOOKS_DIR = REPO_ROOT / ".githooks"


def _load_guard():
    spec = importlib.util.spec_from_file_location("guard_main", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def guard():
    return _load_guard()


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """A throwaway repo on `main` with one commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "README.md").write_text("x\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


@pytest.fixture
def git():
    return _git
