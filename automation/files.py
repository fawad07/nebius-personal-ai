"""
automation/files.py

File operations with an asymmetric sandbox — deliberately:

  - READ (list_files, tree, read_file) is allowed anywhere under the
    project root, so the assistant can browse and read the codebase.
  - WRITE and DELETE (write_file, delete_file) stay confined to the
    workspace (data/workspace/), so the LLM — or a prompt injection — can
    never overwrite or delete source, config, or the security code itself.

Anything outside the relevant root is refused with OutsideWorkspaceError
unless the human approves it once via the consent flow (see core/agent.py).
The point is that reading is low-risk and useful, while destruction is
kept in a box.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from core.paths import DATA_DIR, PROJECT_ROOT

logger = logging.getLogger("automation.files")

# Write/delete are confined here. Read is allowed across READ_ROOT.
WORKSPACE_ROOT = (DATA_DIR / "workspace").resolve()
READ_ROOT = PROJECT_ROOT.resolve()

# Directories skipped in list/tree output: pure noise for a human browsing
# the code, and (for the encrypted data dirs) not meaningful to read.
_SKIP_NAMES = {
    ".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv",
    ".DS_Store", ".mypy_cache", ".claude",
}

# Cap on how much of a file read_file returns. The result is shown to the
# user AND replayed into later turns as context, so an unbounded read of a
# large file would overflow the LLM's context window. ~40k chars is well
# within a 4k-token window once history is also trimmed (see core/agent.py).
MAX_READ_CHARS = 40_000


class UnsafePathError(Exception):
    """Raised when a requested path would escape its allowed root."""


class OutsideWorkspaceError(UnsafePathError):
    """
    A requested path resolves outside its allowed root and no grant covers it.

    Carries the fully resolved path so the caller can show the human
    exactly what is being asked for before they approve or refuse. Kept a
    subclass of UnsafePathError so anything treating an escape as a hard
    error keeps working; the difference is only that this one is
    answerable by a person.
    """

    def __init__(self, path: Path, requested: str):
        super().__init__(
            f"Path '{requested}' resolves outside the allowed area: {path}"
        )
        self.path = path
        self.requested = requested


def _write_root() -> Path:
    """
    The workspace root (write/delete), always fully resolved.

    Candidates get .resolve()d, so comparing them to an unresolved root
    breaks whenever the path crosses a symlink — on macOS /var is a symlink
    to /private/var, so a workspace under a temp dir would both reject its
    own legitimate files and raise ValueError from relative_to().
    """
    return WORKSPACE_ROOT.resolve()


def _read_root() -> Path:
    return READ_ROOT.resolve()


def _resolve(relative_path: str, root: Path, grant: Path | None = None) -> Path:
    """
    Resolve an LLM-supplied path against `root` and confirm it stays inside
    it. Outside is refused with OutsideWorkspaceError unless `grant` names
    exactly that path (or a directory containing it) — a single-use
    permission the human approved for this one operation.

    `grant` is passed in per call and never stored: no previous approval
    can widen a later request, and the model cannot grant itself anything.
    """
    root = root.resolve()
    # expanduser first so "~/Desktop/x" is understood as the user's home
    # rather than a literal "~" directory inside the root.
    requested = Path(relative_path).expanduser()
    candidate = (requested if requested.is_absolute() else (root / requested)).resolve()

    # Component-wise containment check, NOT a string prefix check: a prefix
    # check lets sibling directories through ("data/workspace_evil"
    # startswith "data/workspace").
    if candidate.is_relative_to(root):
        return candidate

    if grant is not None:
        granted = Path(grant).resolve()
        if candidate == granted or candidate.is_relative_to(granted):
            logger.info("One-time approved access outside root: %s", candidate)
            return candidate

    raise OutsideWorkspaceError(candidate, relative_path)


def _resolve_read(relative_path: str, grant: Path | None = None) -> Path:
    """
    Resolve a path for a read operation.

    Files the assistant itself wrote live in the workspace, so a bare name
    ("notes.txt") should still find them there. Otherwise the path is
    resolved against the project root (so "core/agent.py" works), which is
    also where the consent flow kicks in for anything further out.
    """
    requested = Path(relative_path).expanduser()
    if not requested.is_absolute():
        ws = (_write_root() / requested).resolve()
        if ws.is_relative_to(_write_root()) and ws.exists():
            return ws
    return _resolve(relative_path, _read_root(), grant)


def _visible(entries):
    """Filter out noise names (VCS/cache dirs) from a directory listing."""
    return [p for p in entries if p.name not in _SKIP_NAMES]


def _tree_into(directory: Path, lines: list, prefix: str, depth: int = 0) -> None:
    """Append an indented `tree`-style listing of `directory` to `lines`."""
    if depth > 20:      # guard against a pathological/symlinked hierarchy
        return
    entries = _visible(sorted(directory.iterdir(),
                              key=lambda p: (p.is_file(), p.name.lower())))
    for i, entry in enumerate(entries):
        last = i == len(entries) - 1
        connector = "└── " if last else "├── "
        name = entry.name + ("/" if entry.is_dir() else "")
        lines.append(prefix + connector + name)
        if entry.is_dir():
            _tree_into(entry, lines, prefix + ("    " if last else "│   "), depth + 1)


def execute(action: str, args: dict, grant: Path | None = None):
    """
    grant: a single path the human approved for this one call (see
    _resolve). None means the default roots apply — reads across the
    project, writes/deletes only in the workspace.
    """
    _write_root().mkdir(parents=True, exist_ok=True)

    # The small model sometimes emits a tool call missing its required
    # argument (e.g. read_file with no "path"). Guard here so that returns a
    # friendly message instead of a raw KeyError crash.
    if action in ("read_file", "write_file", "delete_file"):
        if not str(args.get("path", "")).strip():
            return f"I need a file path for {action}, but none was given."

    if action == "list_files":
        subdir = args.get("path", ".")
        target = _resolve(subdir, _read_root(), grant)
        if not target.exists():
            return (f"No such directory in the project: {subdir}.")
        if not target.is_dir():
            return f"Not a directory: {subdir}"
        return [p.name + ("/" if p.is_dir() else "")
                for p in _visible(sorted(target.iterdir(), key=lambda p: p.name.lower()))]

    if action == "tree":
        subdir = args.get("path", ".")
        root = _resolve(subdir, _read_root(), grant)
        if not root.exists():
            return f"No such directory in the project: {subdir}."
        if not root.is_dir():
            return f"Not a directory: {subdir}"
        label = "." if root == _read_root() else root.name + "/"
        lines = [label]
        _tree_into(root, lines, prefix="")
        return "\n".join(lines) if len(lines) > 1 else label + "\n(empty)"

    if action == "read_file":
        target = _resolve_read(args["path"], grant)
        if not target.exists():
            return f"No such file in the project: {args['path']}."
        if target.is_dir():
            return f"That's a directory, not a file: {args['path']}"
        try:
            raw = target.read_bytes()
        except OSError as e:
            logger.error("Failed to read %s: %s", target, e)
            return f"Could not read file: {e}"
        # Refuse binary files: NUL bytes in the head mean it's not text, and
        # decoding it would produce garbage (and, for a large model file,
        # flood the LLM's context window).
        if b"\x00" in raw[:8192]:
            return (f"{args['path']} looks like a binary file "
                    f"({len(raw)} bytes) — I can only read text files.")
        text = raw.decode("utf-8", errors="replace")
        # Cap the size so reading a large file can't overflow the model's
        # context window (the whole result is later fed back as history).
        if len(text) > MAX_READ_CHARS:
            text = (text[:MAX_READ_CHARS]
                    + f"\n\n[... truncated: file is {len(text)} characters; "
                      f"showing the first {MAX_READ_CHARS} ...]")
        return text

    if action == "write_file":
        # Confined to the workspace: the assistant may create files in its
        # own area, never overwrite the project's source or config.
        target = _resolve(args["path"], _write_root(), grant)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_text(args.get("content", ""))
            try:
                shown = target.relative_to(_write_root())
            except ValueError:
                shown = target      # a granted path lies outside the workspace
            return f"Wrote {shown}"
        except OSError as e:
            logger.error("Failed to write %s: %s", target, e)
            return f"Could not write file: {e}"

    if action == "delete_file":
        # NOTE: core/safety.py marks delete_file as sensitive — this
        # should only be reached after admin auth succeeds (see agent.py).
        # Confined to the workspace: never deletes project files.
        target = _resolve(args["path"], _write_root(), grant)
        if not target.exists():
            return f"No such file in the workspace: {args['path']}."
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            return f"Deleted {args['path']}"
        except OSError as e:
            logger.error("Failed to delete %s: %s", target, e)
            return f"Could not delete file: {e}"

    raise NotImplementedError(f"automation.files: unknown action '{action}'")
