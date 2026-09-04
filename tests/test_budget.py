"""Size budgets for `src/stylebot`, enforced with a shrinking allowlist (D58).

Budgets: a module <= 800 lines, a function <= 80, a docstring <= 15, a module
docstring <= 30, and package-wide (docstring + comment) / code <= 0.40.

Everything over budget on 2026-09-04 is recorded in `ALLOWLIST` at its exact
current size. A recorded number is a ceiling that may only ever be lowered: the
test fails if it grows, and fails again once the module comes under budget, so
the entry has to be deleted. Nothing may be added to the allowlist without a
`decisions.md` entry saying why (P8).

Measured the way the 2026-09 seam QA measured: `ast.get_docstring` over module,
class and function nodes; `#`-prefixed lines as comments; a docstring's whole
source span counts as docstring (blank lines inside it included); blank lines
outside a docstring count as neither prose nor code.

Rules and rationale: `CLAUDE.md` "Budgets and layers" (P1-P8) and
`_plans/decisions.md` D58.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "stylebot"

MODULE_LINES = 800
FUNCTION_LINES = 80
DOCSTRING_LINES = 15
MODULE_DOCSTRING_LINES = 30
RATIO_BUDGET = 0.40

# (docstring + comment) lines in the package after F8's diet, when the ratio was
# 0.4108. This, not the ratio, is the ratchet: deleting mechanism raises the
# ratio while lowering the prose. Only ever lower it; set it to None once the
# package meets RATIO_BUDGET.
PROSE_CEILING: int | None = 4711

# module -> the budgets it is over today, at their exact current size.
# "functions" and "docstrings" list every over-budget span, largest first.
# A failure names the exact number to lower an entry to, or the entry to delete.
ALLOWLIST: dict[str, dict[str, object]] = {
    "bin/ai_style.py": {"lines": 1669, "functions": (152, 105, 88), "docstrings": (16,)},
    "bin/ai_style_data.py": {"functions": (113,)},
    "capture.py": {"functions": (99, 93)},
    "classify_train.py": {
        "lines": 965,
        "functions": (82,),
        "docstrings": (22, 19),
        "module_docstring": 33,
    },
    "generators.py": {"functions": (158, 102, 99)},
    "infer.py": {"lines": 865, "functions": (197, 96), "docstrings": (28, 18)},
    "report.py": {"lines": 885, "functions": (130,)},
    "session.py": {"lines": 845, "functions": (110,)},
    "splits.py": {"functions": (112,), "docstrings": (19,), "module_docstring": 32},
    "status.py": {"functions": (88,)},
    "synth.py": {"lines": 1264, "functions": (128, 128), "docstrings": (26,)},
    "synth_cli.py": {
        "lines": 1117,
        "functions": (259, 123),
        "docstrings": (16,),
    },
    "targets.py": {"functions": (95,)},
    "train.py": {
        "lines": 997,
        "functions": (97, 95, 90, 89),
        "docstrings": (26, 23, 20, 20, 18, 17),
        "module_docstring": 35,
    },
}

_DEFINITION = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)


def measure(path: Path) -> dict[str, object]:
    """Line counts for one module: sizes, and the prose/code split."""
    source = path.read_text()
    lines = source.splitlines()
    tree = ast.parse(source)

    docstring_lines: set[int] = set()
    docstrings: list[int] = []
    module_docstring = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, *_DEFINITION)):
            continue
        if ast.get_docstring(node, clean=False) is None:
            continue
        span = node.body[0]
        size = span.end_lineno - span.lineno + 1
        docstring_lines |= set(range(span.lineno, span.end_lineno + 1))
        if isinstance(node, ast.Module):
            module_docstring = size
        else:
            docstrings.append(size)

    functions = [
        node.end_lineno - node.lineno + 1
        for node in ast.walk(tree)
        if isinstance(node, _FUNCTION)
    ]

    comment = code = 0
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if number in docstring_lines or not stripped:
            continue
        if stripped.startswith("#"):
            comment += 1
        else:
            code += 1

    return {
        "lines": len(lines),
        "functions": sorted((n for n in functions if n > FUNCTION_LINES), reverse=True),
        "docstrings": sorted(
            (n for n in docstrings if n > DOCSTRING_LINES), reverse=True
        ),
        "module_docstring": module_docstring,
        "prose": len(docstring_lines) + comment,
        "code": code,
    }


def _modules() -> list[tuple[str, dict[str, object]]]:
    found = [
        (str(path.relative_to(PACKAGE)), measure(path))
        for path in sorted(PACKAGE.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]
    assert found, f"no modules under {PACKAGE}"
    return found


def _check_scalar(name: str, key: str, budget: int, actual: int, allowed: dict) -> list[str]:
    ceiling = allowed.get(key)
    if actual > budget:
        if ceiling is None:
            return [f"{name}: {key} is {actual} (budget {budget}) and is not allowlisted"]
        if actual > ceiling:
            return [f"{name}: {key} grew {ceiling} -> {actual}; the allowlist only shrinks"]
    elif ceiling is not None:
        return [f"{name}: {key} is {actual}, under budget — delete its allowlist entry"]
    return []


def _check_spans(name: str, key: str, budget: int, actual: list[int], allowed: dict) -> list[str]:
    ceilings = list(allowed.get(key, ()))
    if not actual:
        if ceilings:
            return [f"{name}: no {key} over {budget} lines — delete its allowlist entry"]
        return []
    if not ceilings:
        return [f"{name}: {key} over {budget} lines {actual} are not allowlisted"]
    if len(actual) > len(ceilings):
        return [f"{name}: {len(actual)} {key} over {budget} lines, allowlist has {len(ceilings)}"]
    grew = [(a, c) for a, c in zip(actual, ceilings, strict=False) if a > c]
    if grew:
        return [f"{name}: {key} grew {grew} (actual, ceiling); the allowlist only shrinks"]
    return []


def test_no_module_is_over_budget_or_over_its_allowance() -> None:
    problems: list[str] = []
    for name, m in _modules():
        allowed = ALLOWLIST.get(name, {})
        problems += _check_scalar(name, "lines", MODULE_LINES, m["lines"], allowed)
        problems += _check_scalar(
            name,
            "module_docstring",
            MODULE_DOCSTRING_LINES,
            m["module_docstring"],
            allowed,
        )
        problems += _check_spans(name, "functions", FUNCTION_LINES, m["functions"], allowed)
        problems += _check_spans(name, "docstrings", DOCSTRING_LINES, m["docstrings"], allowed)
    assert not problems, "size budget (D58):\n  " + "\n  ".join(problems)


def test_allowlist_names_real_modules() -> None:
    names = {name for name, _ in _modules()}
    stale = sorted(set(ALLOWLIST) - names)
    assert not stale, f"allowlist names modules that no longer exist: {stale}"


def test_prose_does_not_outgrow_the_code_it_explains() -> None:
    modules = _modules()
    prose = sum(m["prose"] for _, m in modules)
    code = sum(m["code"] for _, m in modules)
    ratio = prose / code
    if PROSE_CEILING is None:
        assert ratio <= RATIO_BUDGET, f"(docstring+comment)/code is {ratio:.4f}"
        return
    assert prose <= PROSE_CEILING, (
        f"docstring+comment lines grew {PROSE_CEILING} -> {prose} "
        f"((docstring+comment)/code {ratio:.4f}, budget {RATIO_BUDGET}). "
        "Say it once, in the layer that owns it (D58/P1)."
    )
    assert ratio > RATIO_BUDGET, (
        f"the package now meets the {RATIO_BUDGET} ratio ({ratio:.4f}) — "
        "set PROSE_CEILING = None"
    )
