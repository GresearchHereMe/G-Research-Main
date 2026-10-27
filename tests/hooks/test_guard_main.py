"""Tests for .claude/hooks/guard_main.py (Claude Code PreToolUse guard)."""

import json
import os
import subprocess
import sys

import pytest
from conftest import GUARD_PATH


def on_branch(name):
    return lambda _cwd: name


ON_MAIN = on_branch("main")
ON_FEATURE = on_branch("feat/backtester")


# --- commits -----------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        'git commit -m "x"',
        "git add . && git commit -m x",
        "git status; git commit -am x",
        "git -c user.name=a commit -m x",
    ],
)
def test_commit_on_main_blocked(guard, cmd):
    assert guard.evaluate(cmd, "/repo", ON_MAIN) is not None


def test_commit_on_feature_allowed(guard):
    assert guard.evaluate('git commit -m "feat: x"', "/repo", ON_FEATURE) is None


def test_commit_message_mentioning_main_allowed(guard):
    cmd = 'git commit -m "docs: never push to main; never gh pr merge"'
    assert guard.evaluate(cmd, "/repo", ON_FEATURE) is None


def test_commit_no_verify_blocked(guard):
    assert guard.evaluate("git commit --no-verify -m x", "/r", ON_FEATURE)
    assert guard.evaluate("git commit -n -m x", "/r", ON_FEATURE)


def test_cd_then_commit_uses_new_cwd(guard):
    lookup = lambda cwd: "main" if cwd == "/other" else "feat/x"
    assert guard.evaluate("cd /other && git commit -m x", "/repo", lookup)
    assert guard.evaluate("git commit -m x", "/repo", lookup) is None


def test_git_dash_c_path_uses_that_repo(guard):
    lookup = lambda cwd: "main" if cwd == "/other" else "feat/x"
    assert guard.evaluate("git -C /other commit -m x", "/repo", lookup)


# --- pushes ------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "git push origin main",
        "git push origin HEAD:main",
        "git push origin feat/x:refs/heads/main",
        "git push -u origin master",
        "git push origin +main",
        "git push --all",
        "git push --mirror origin",
        "git push origin --delete main",
    ],
)
def test_push_to_protected_blocked(guard, cmd):
    assert guard.evaluate(cmd, "/repo", ON_FEATURE) is not None


@pytest.mark.parametrize(
    "cmd", ["git push", "git push origin", "git push origin HEAD", "git push -o ci.skip origin"]
)
def test_implicit_push_from_main_blocked(guard, cmd):
    assert guard.evaluate(cmd, "/repo", ON_MAIN) is not None


@pytest.mark.parametrize(
    "cmd",
    [
        "git push",
        "git push -u origin feat/backtester",
        "git push origin HEAD",
        "git push --force-with-lease origin feat/backtester",
    ],
)
def test_push_feature_allowed(guard, cmd):
    assert guard.evaluate(cmd, "/repo", ON_FEATURE) is None


@pytest.mark.parametrize(
    "cmd",
    [
        "git push --force origin feat/x",
        "git push -f origin feat/x",
        "git push -uf origin feat/x",
        "git push --no-verify origin feat/x",
    ],
)
def test_force_or_no_verify_push_blocked(guard, cmd):
    assert guard.evaluate(cmd, "/repo", ON_FEATURE) is not None


# --- merges / branch ops -----------------------------------------------------


def test_merge_into_main_blocked(guard):
    assert guard.evaluate("git merge feat/x", "/repo", ON_MAIN)


def test_merge_abort_on_main_allowed(guard):
    assert guard.evaluate("git merge --abort", "/repo", ON_MAIN) is None


def test_merge_main_into_feature_allowed(guard):
    assert guard.evaluate("git merge origin/main", "/repo", ON_FEATURE) is None


def test_delete_main_branch_blocked(guard):
    assert guard.evaluate("git branch -D main", "/repo", ON_FEATURE)


def test_pull_ff_only_on_main_allowed(guard):
    assert guard.evaluate("git pull --ff-only", "/repo", ON_MAIN) is None


def test_switch_to_main_allowed(guard):
    assert guard.evaluate("git checkout main", "/repo", ON_FEATURE) is None


# --- gh ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "gh pr merge 12 --squash",
        "gh pr merge --auto",
        "gh pr review 12 --approve",
        "gh pr review -a",
        "gh api -X DELETE repos/o/r/branches/main/protection",
        "gh api --method PUT repos/o/r/pulls/3/merge",
        "gh api repos/o/r/rulesets/1 -X DELETE",
    ],
)
def test_gh_merge_approve_protection_blocked(guard, cmd):
    assert guard.evaluate(cmd, "/repo", ON_FEATURE) is not None


@pytest.mark.parametrize(
    "cmd",
    [
        "gh pr create --fill",
        "gh pr view 12",
        "gh pr review 12 --comment -b 'lgtm'",
        "gh api repos/o/r/branches/main/protection",
    ],
)
def test_gh_safe_commands_allowed(guard, cmd):
    assert guard.evaluate(cmd, "/repo", ON_FEATURE) is None


# --- nesting / robustness ----------------------------------------------------


def test_bash_dash_c_nested_command_checked(guard):
    assert guard.evaluate("bash -c 'git push origin main'", "/repo", ON_FEATURE)


def test_newline_separated_commands_checked(guard):
    assert guard.evaluate("git status\ngit push origin main", "/repo", ON_FEATURE)


def test_env_prefix_ignored(guard):
    assert guard.evaluate("GIT_TRACE=1 git push origin main", "/repo", ON_FEATURE)


def test_unrelated_commands_allowed(guard):
    assert guard.evaluate("ls -la && pytest -q", "/repo", ON_MAIN) is None


def test_unparseable_command_falls_back_to_regex(guard):
    assert guard.evaluate("gh pr merge 1 'unterminated", "/repo", ON_FEATURE)


# --- end-to-end via stdin ----------------------------------------------------


def _run_hook(payload: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GUARD_PATH)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )


def test_hook_blocks_with_exit_2(temp_repo):
    payload = {
        "tool_name": "Bash",
        "cwd": str(temp_repo),
        "tool_input": {"command": "git commit --allow-empty -m x"},
    }
    result = _run_hook(payload)
    assert result.returncode == 2
    assert "main" in result.stderr


def test_hook_allows_with_exit_0(temp_repo, git):
    git(temp_repo, "checkout", "-q", "-b", "feat/x")
    payload = {
        "tool_name": "Bash",
        "cwd": str(temp_repo),
        "tool_input": {"command": "git commit --allow-empty -m x"},
    }
    assert _run_hook(payload).returncode == 0


def test_hook_ignores_non_bash_tools():
    assert _run_hook({"tool_name": "Read", "tool_input": {}}).returncode == 0


def test_hook_survives_garbage_stdin():
    result = subprocess.run(
        [sys.executable, str(GUARD_PATH)],
        input="not json",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "guard_main" in result.stderr


# --- regressions from review -------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "# sync with remote\ngit push origin main",
        "git push origin main \\\n  --quiet",
        "git checkout main && git commit -m x",
        "git switch main && git merge feat/x",
        "bash -lc 'git push origin main'",
        "eval 'git push origin main'",
        "env -i git push origin main",
        "timeout 60 git push origin main",
        "nice -n 5 git push origin main",
        "sudo -u bot git push origin main",
        "echo `git push origin main`",
        "x=$(git push origin main)",
        "git push origin heads/main",
        "git branch -f main HEAD~1",
        "git update-ref refs/heads/main HEAD",
        "gh api -XPUT repos/o/r/pulls/3/merge",
        "gh api -X PATCH repos/o/r/git/refs/heads/main -f sha=abc",
        "gh api graphql -f query='mutation { mergePullRequest(input:{}) { clientMutationId } }'",
        "gh api graphql -f query='mutation { enablePullRequestAutoMerge(input:{}) { x } }'",
    ],
)
def test_review_bypasses_blocked(guard, cmd):
    assert guard.evaluate(cmd, "/repo", ON_FEATURE) is not None


@pytest.mark.parametrize(
    "cmd",
    [
        "git cherry-pick abc123",
        "git revert HEAD",
        "git rebase feat/x",
        "git pull origin feat/x",
        "git am patch.mbox",
    ],
)
def test_history_writes_on_main_blocked(guard, cmd):
    assert guard.evaluate(cmd, "/repo", ON_MAIN) is not None


@pytest.mark.parametrize(
    "cmd",
    [
        "git reset --hard origin/main",
        "git rebase --abort",
        "git fetch origin && git pull --ff-only origin main",
    ],
)
def test_main_sync_commands_allowed(guard, cmd):
    assert guard.evaluate(cmd, "/repo", ON_MAIN) is None


@pytest.mark.parametrize(
    "cmd",
    [
        "git checkout -b feat/new && git commit -m x",
        "git switch -c feat/new && git push -u origin feat/new",
        "git rebase origin/main",
        "git cherry-pick abc123",
        'git commit -m "$(cat <<\'EOF\'\nfeat: x\n\ngit push origin main is blocked\nEOF\n)"',
        "echo 'git push origin main' > notes.txt",
    ],
)
def test_feature_work_not_false_positive(guard, cmd):
    assert guard.evaluate(cmd, "/repo", ON_FEATURE) is None


def test_checkout_new_branch_from_main_then_commit_allowed(guard):
    cmd = "git checkout -b feat/new && git commit -m x"
    assert guard.evaluate(cmd, "/repo", ON_MAIN) is None


def test_unknown_branch_fails_closed_for_commit(guard):
    assert guard.evaluate("git commit -m x", "/repo", lambda _c: None) is not None


def test_unparseable_git_command_fails_closed(guard):
    assert guard.evaluate("git commit -m 'unterminated", "/repo", ON_FEATURE)


def test_dash_c_tilde_expanded(guard):
    home_repo = os.path.expanduser("~/repo")
    lookup = lambda cwd: "main" if cwd == home_repo else "feat/x"
    assert guard.evaluate("git -C ~/repo commit -m x", "/elsewhere", lookup)
