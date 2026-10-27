#!/usr/bin/env python3
"""Claude Code PreToolUse guard for Bash: keep agents off `main`.

Blocks (exit 2, reason on stderr):
  * history writes on a protected branch: commit, merge, cherry-pick, revert, am, rebase,
    pull without --ff-only (branch tracked through `checkout`/`switch` in the same command)
  * `git push` that targets a protected branch (explicit, implicit, --all, --mirror, HEAD)
  * force pushes (`-f`, `--force`, `+ref`) and `--no-verify` on commit/push
  * deleting, renaming, force-moving or update-ref'ing a protected branch
  * `gh pr merge`, `gh pr review --approve`, mutating `gh api` calls on protection,
    rulesets, merges or protected refs, and graphql merge/approve mutations
  * unparseable commands that mention git/gh (fail closed)

Defense in depth only: GitHub branch protection is the real gate. See RULES.md.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable

PROTECTED = frozenset({"main", "master"})
OPERATORS = frozenset({"&&", "||", ";", "|", "&", "(", ")", ";;", "|&"})
SEPARATOR_CHARS = frozenset("();<>|&")
SHELLS = frozenset({"bash", "sh", "zsh"})
WRAPPERS = frozenset(
    {"sudo", "command", "env", "exec", "time", "nohup", "timeout", "nice", "xargs", "setsid"}
)
WRAPPER_OPTS_WITH_VALUE = frozenset({"-u", "-g", "-n", "-s", "-k", "-I", "-L", "-P"})
ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
DURATION = re.compile(r"\d+(\.\d+)?[smhd]?")
GIT_OPTS_WITH_VALUE = frozenset({"-C", "-c", "--git-dir", "--work-tree", "--namespace"})
PUSH_OPTS_WITH_VALUE = frozenset({"-o", "--push-option", "--repo", "--receive-pack", "--exec"})
MAIN_HISTORY_WRITERS = frozenset({"commit", "merge", "cherry-pick", "revert", "am", "rebase", "pull"})
ESCAPE_FLAGS = frozenset({"--abort", "--quit", "--skip", "--ff-only"})
BRANCH_MUTATING_FLAGS = frozenset(
    {"-d", "-D", "--delete", "-m", "-M", "--move", "-f", "--force", "-c", "-C", "--copy"}
)
GH_API_GUARDED_PATH = re.compile(
    r"/(protection|rulesets)(/|$)|/pulls/\d+/merge$|/merges$|/git/refs/heads/(main|master)$"
)
GH_GRAPHQL_GUARDED = re.compile(
    r"mergePullRequest|enablePullRequestAutoMerge|BranchProtectionRule|\bAPPROVE\b"
)
GH_API_WRITE_FLAGS = frozenset({"-f", "-F", "--field", "--raw-field", "--input"})
MENTIONS_GIT = re.compile(r"\b(git|gh)\b")

BranchLookup = Callable[[str], str | None]


def current_branch(cwd: str) -> str | None:
    """Return the checked-out branch in `cwd`, or None when unknown."""
    try:
        out = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def tokenize(command: str) -> list[list[str]]:
    """Split a shell command into simple-command segments of argv tokens.

    Backticks and `$(` outside quotes become separators so substituted commands are checked.
    """
    text = command.replace("\\\n", " ").replace("`", " ; ").replace("$(", " ; ( ")
    lexer = shlex.shlex(text.replace("\n", " ; "), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    segments: list[list[str]] = [[]]
    for token in lexer:
        if token in OPERATORS or set(token) <= SEPARATOR_CHARS:
            segments.append([])
        else:
            segments[-1].append(token)
    return [seg for seg in segments if seg]


def strip_prefix(argv: list[str]) -> list[str]:
    """Drop VAR=value assignments and wrappers (sudo -u x, timeout 60, nice -n 5, env -i ...)."""
    idx = 0
    while idx < len(argv):
        if ASSIGNMENT.fullmatch(argv[idx]):
            idx += 1
        elif argv[idx] in WRAPPERS:
            idx += 1
            while idx < len(argv) and (argv[idx].startswith("-") or DURATION.fullmatch(argv[idx])):
                idx += 2 if argv[idx] in WRAPPER_OPTS_WITH_VALUE else 1
        else:
            break
    return argv[idx:]


def short_flags(args: list[str]) -> set[str]:
    """Collect single-letter flags, expanding clusters like `-uf`."""
    letters: set[str] = set()
    for arg in args:
        if re.fullmatch(r"-[A-Za-z]+", arg):
            letters.update(arg[1:])
    return letters


def ref_name(ref: str) -> str:
    return ref.lstrip("+").removeprefix("refs/").removeprefix("heads/")


def positionals(args: list[str], opts_with_value: frozenset[str]) -> list[str]:
    out: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif arg in opts_with_value:
            skip = True
        elif not arg.startswith("-"):
            out.append(arg)
    return out


def split_git(args: list[str], cwd: str) -> tuple[str | None, list[str], str]:
    """Return (subcommand, subcommand args, effective repo dir) for `git ...`."""
    idx = 0
    while idx < len(args) and args[idx].startswith("-"):
        opt = args[idx]
        if opt in GIT_OPTS_WITH_VALUE and idx + 1 < len(args):
            if opt == "-C":
                cwd = os.path.join(cwd, os.path.expanduser(args[idx + 1]))
            idx += 2
        else:
            idx += 1
    if idx >= len(args):
        return None, [], cwd
    return args[idx], args[idx + 1 :], cwd


def switched_branch(sub: str, rest: list[str]) -> str | None:
    """Branch a `git checkout`/`git switch` would leave HEAD on, if it switches branches."""
    if sub not in {"checkout", "switch"} or "--" in rest:
        return None
    for flag in ("-b", "-B", "-c", "-C", "--create", "--force-create"):
        if flag in rest and rest.index(flag) + 1 < len(rest):
            return ref_name(rest[rest.index(flag) + 1])
    names = positionals(rest, frozenset())
    return ref_name(names[0]) if len(names) == 1 else None


def check_push(args: list[str], branch: str | None) -> str | None:
    flags = short_flags(args)
    if "--force" in args or "f" in flags:
        return "force push is not allowed; use --force-with-lease on your own branch"
    if "--no-verify" in args:
        return "--no-verify bypasses repo hooks"
    if "--all" in args or "--mirror" in args:
        return "--all/--mirror would push protected branches"
    refspecs = positionals(args, PUSH_OPTS_WITH_VALUE)[1:]
    if any(ref.startswith("+") for ref in refspecs):
        return "force push via +refspec is not allowed"
    for ref in refspecs:
        dest = ref_name(ref.split(":")[-1])
        if dest == "HEAD":
            dest = branch or "HEAD"
        if dest in PROTECTED:
            return f"push targets protected branch '{dest}'"
    if not refspecs and (branch is None or branch in PROTECTED):
        return f"implicit push from protected or unknown branch '{branch}'"
    return None


def check_history_write(sub: str, rest: list[str], branch: str | None) -> str | None:
    if sub == "commit" and ("--no-verify" in rest or "n" in short_flags(rest)):
        return "--no-verify bypasses repo hooks"
    if sub == "commit" or not ESCAPE_FLAGS & set(rest):
        if branch is None:
            return f"`git {sub}` with unknown current branch; check `git status` first"
        if branch in PROTECTED:
            return f"`git {sub}` on protected branch '{branch}'; work on a feature branch + PR"
    return None


def check_git(args: list[str], cwd: str, branch_of: BranchLookup) -> str | None:
    sub, rest, repo = split_git(args, cwd)
    if sub in MAIN_HISTORY_WRITERS:
        return check_history_write(sub, rest, branch_of(repo))
    if sub == "push":
        return check_push(rest, branch_of(repo))
    names = {ref_name(arg) for arg in rest}
    if sub == "branch" and BRANCH_MUTATING_FLAGS & set(rest) and PROTECTED & names:
        return "deleting, renaming or force-moving a protected branch"
    if sub == "update-ref" and PROTECTED & names:
        return "update-ref on a protected branch"
    return None


def gh_api_method(args: list[str]) -> str:
    method = "GET"
    for idx, arg in enumerate(args):
        if arg in {"-X", "--method"} and idx + 1 < len(args):
            method = args[idx + 1]
        elif arg.startswith("--method="):
            method = arg.split("=", 1)[1]
        elif re.fullmatch(r"-X[A-Za-z]+", arg):
            method = arg[2:]
    return method.upper()


def check_gh(args: list[str]) -> str | None:
    if args[:2] == ["pr", "merge"]:
        return "agents never merge PRs; a human reviewer merges"
    if args[:2] == ["pr", "review"] and ("--approve" in args or "a" in short_flags(args)):
        return "agents never approve PRs"
    if args[:1] != ["api"]:
        return None
    if args[1:2] == ["graphql"] and GH_GRAPHQL_GUARDED.search(" ".join(args)):
        return "graphql merge/approve/branch-protection mutation"
    writes = gh_api_method(args) != "GET" or bool(GH_API_WRITE_FLAGS & set(args))
    if writes and any(GH_API_GUARDED_PATH.search(arg) for arg in args[1:]):
        return "mutating branch protection / rulesets / merges / protected refs via gh api"
    return None


def nested_script(prog: str, args: list[str]) -> str | None:
    """Script text run by `bash -c`, `bash -lc`, `eval ...`, else None."""
    if prog == "eval":
        return " ".join(args)
    if prog in SHELLS:
        for idx, arg in enumerate(args):
            if re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", arg) and idx + 1 < len(args):
                return args[idx + 1]
    return None


def evaluate(command: str, cwd: str, lookup: BranchLookup = current_branch) -> str | None:
    """Return a block reason for `command`, or None when it is allowed."""
    try:
        segments = tokenize(command)
    except ValueError:
        if MENTIONS_GIT.search(command):
            return "unparseable command touching git/gh (unbalanced quotes?); rewrite it"
        return None

    switched: dict[str, str] = {}

    def branch_of(repo: str) -> str | None:
        return switched[repo] if repo in switched else lookup(repo)

    for seg in segments:
        argv = strip_prefix(seg)
        if not argv:
            continue
        prog, args = os.path.basename(argv[0]), argv[1:]
        script = nested_script(prog, args)
        if script is not None:
            reason = evaluate(script, cwd, branch_of)
        elif prog == "cd" and args:
            cwd, reason = os.path.join(cwd, os.path.expanduser(args[0])), None
        elif prog == "git":
            reason = check_git(args, cwd, branch_of)
            sub, rest, repo = split_git(args, cwd)
            target = switched_branch(sub or "", rest)
            if target:
                switched[repo] = target
        else:
            reason = check_gh(args) if prog == "gh" else None
        if reason:
            return reason
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"guard_main: could not parse hook input ({exc}); allowing", file=sys.stderr)
        return 0
    if payload.get("tool_name") != "Bash":
        return 0
    command = (payload.get("tool_input") or {}).get("command", "")
    cwd = payload.get("cwd") or os.getcwd()
    reason = evaluate(command, cwd)
    if reason:
        print(f"Blocked by repo guard (RULES.md): {reason}.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
