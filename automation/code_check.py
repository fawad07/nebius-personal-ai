"""
automation/code_check.py

Ground-truth Python syntax checking — so the assistant answers "is this
code correct?" from Python's OWN parser, not the language model's guess.

The small model happily says a SyntaxError "looks correct." This tool
removes the guessing: it compiles the code (compile(), which parses but
does NOT run it — no exec, no eval, nothing executes) and reports exactly
what Python says. It only judges SYNTAX, not logic or runtime behaviour,
and only Python — which it states plainly rather than overreaching.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("automation.code_check")

MAX_CODE_CHARS = 20_000   # a snippet to check, not a whole codebase


def _check_python(code: str) -> str:
    try:
        # mode="exec" parses a full snippet. compile() never runs the code;
        # it only turns valid source into a code object, raising SyntaxError
        # on invalid syntax. Nothing here executes what the user pasted.
        compile(code, "<checked-code>", "exec")
    except SyntaxError as e:
        line = e.lineno or 0
        offending = ""
        lines = code.splitlines()
        if 1 <= line <= len(lines):
            src = lines[line - 1]
            caret = " " * max(0, (e.offset or 1) - 1) + "^"
            offending = f"\n    {src}\n    {caret}"
        return (f"Not valid Python — SyntaxError on line {line}: {e.msg}."
                f"{offending}")
    except ValueError as e:
        # e.g. source containing a null byte.
        return f"Not valid Python: {e}"
    return "That's valid Python syntax. (Syntax only — I didn't check the logic.)"


def execute(action: str, args: dict, grant=None):
    if action == "check_code":
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            return "Give me the code to check (as the 'code' argument)."
        if len(code) > MAX_CODE_CHARS:
            return f"That's too long to check (limit {MAX_CODE_CHARS} characters)."
        return _check_python(code)

    raise NotImplementedError(f"automation.code_check: '{action}' not yet implemented")
