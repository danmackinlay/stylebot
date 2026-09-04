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
import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "stylebot"

MODULE_LINES = 800
FUNCTION_LINES = 80
DOCSTRING_LINES = 15
MODULE_DOCSTRING_LINES = 30
RATIO_BUDGET = 0.40

# Two high-water marks, and satisfying EITHER satisfies the rule. Neither works
# alone: the count alone makes one documented helper cost a docstring deleted
# elsewhere, and the ratio alone fails an honest code diet, which is why the
# count was picked first. Together they fail only prose grown with no code to
# explain. Only ever lower them; set both to None once RATIO_BUDGET is met.
PROSE_CEILING: int | None = None
RATIO_PIN: float | None = None

# module -> the budgets it is over today, at their exact current size.
# "functions" maps each over-budget function's qualified name (`Class.method`,
# `outer.inner`) to its size, so a swap for a smaller function under a new name
# is flagged (M1a); a name that recurs in one module maps to a tuple of sizes,
# largest first. "docstrings" lists every over-budget span, largest first.
# A failure names the exact number to lower an entry to, or the entry to delete.
ALLOWLIST: dict[str, dict[str, object]] = {
    "bin/ai_style.py": {"lines": 995, "docstrings": (16,)},
    "classify_train.py": {
        "lines": 964,
        "functions": {"train": 82},
        "docstrings": (22, 17),
        "module_docstring": 31,
    },
    "generators.py": {
        "functions": {
            "openai_generator": 158,
            "openai_generator._make_generate": 102,
            "openai_generator._make_generate.generate": 99,
        },
    },
    "infer.py": {
        "lines": 854,
        "functions": {"rewrite_text": 174, "rewrite_candidates": 94},
        "docstrings": (17, 16),
    },
    "splits.py": {
        "functions": {"apply_eval_extension": 112},
        "docstrings": (19,),
        "module_docstring": 32,
    },
    "synth.py": {
        "lines": 1198,
        "functions": {"synthesize_pairs": 128, "_run_session": 128},
        "docstrings": (26,),
    },
    "synth_cli.py": {
        "lines": 1059,
        "functions": {"run_synth": 259, "_report_result": 123},
        "docstrings": (16,),
    },
    "targets.py": {"functions": {"_post_targets": 95}},
    "train.py": {
        "lines": 997,
        "functions": {
            "run_training": 97,
            "_tinker_runner": 95,
            "assemble_training_corpus": 90,
            "build_manifest": 89,
        },
        "docstrings": (26, 23, 20, 18, 17, 17),
        "module_docstring": 35,
    },
}

_DEFINITION = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)


def _functions(node: ast.AST, prefix: str = "") -> list[tuple[str, int]]:
    """`(qualified name, lines)` per function: `Class.method`, `outer.inner`."""
    found: list[tuple[str, int]] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _DEFINITION):
            qualified = prefix + child.name
            if isinstance(child, _FUNCTION):
                found.append((qualified, child.end_lineno - child.lineno + 1))
            found += _functions(child, qualified + ".")
        else:
            found += _functions(child, prefix)
    return found


def measure(path: Path) -> dict[str, object]:
    """Line counts for one module: sizes, and the prose/code split.

    `functions` is every function's sizes by qualified name, largest first;
    the other spans are the over-budget ones only.
    """
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

    functions: dict[str, list[int]] = {}
    for qualified, size in _functions(tree):
        functions.setdefault(qualified, []).append(size)

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
        "functions": {q: sorted(s, reverse=True) for q, s in functions.items()},
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


def _check_functions(name: str, actual: dict[str, list[int]], allowed: dict) -> list[str]:
    """Per qualified name: unlisted, grown, under budget or gone each fail."""
    budget = FUNCTION_LINES
    over = {q: [n for n in s if n > budget] for q, s in actual.items() if s[0] > budget}
    ceilings = {
        q: (c,) if isinstance(c, int) else tuple(c)
        for q, c in dict(allowed.get("functions", {})).items()
    }
    problems: list[str] = []
    for q, sizes in sorted(over.items()):
        c = ceilings.get(q)
        if c is None:
            problems.append(
                f"{name}: {q} is {sizes} lines (budget {budget}) and is not allowlisted"
            )
        elif len(sizes) > len(c):
            problems.append(
                f"{name}: {len(sizes)} defs of {q} over {budget} lines, allowlist has {len(c)}"
            )
        else:
            grew = [(a, k) for a, k in zip(sizes, c, strict=False) if a > k]
            if grew:
                problems.append(
                    f"{name}: {q} grew {grew} (actual, ceiling); the allowlist only shrinks"
                )
    for q in sorted(set(ceilings) - set(over)):
        if q in actual:
            problems.append(
                f"{name}: {q} is {actual[q]} lines, under budget — delete its allowlist entry"
            )
        else:
            problems.append(f"{name}: {q} no longer exists — delete its allowlist entry")
    return problems


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
        problems += _check_functions(name, m["functions"], allowed)
        problems += _check_spans(name, "docstrings", DOCSTRING_LINES, m["docstrings"], allowed)
    assert not problems, "size budget (D58):\n  " + "\n  ".join(problems)


def _module_with(tmp_path: Path, function: str) -> Path:
    body = "\n".join(f"    x{i} = {i}" for i in range(FUNCTION_LINES + 10))
    path = tmp_path / "mod.py"
    path.write_text(f"def {function}():\n{body}\n    return x0\n")
    return path


def test_a_renamed_over_budget_function_is_flagged(tmp_path: Path) -> None:
    allowed = {"functions": {"a": FUNCTION_LINES + 12}}
    as_a = measure(_module_with(tmp_path, "a"))["functions"]
    assert _check_functions("mod.py", as_a, allowed) == []
    as_b = measure(_module_with(tmp_path, "b"))["functions"]
    problems = _check_functions("mod.py", as_b, allowed)
    assert any(p.startswith("mod.py: b is") and "not allowlisted" in p for p in problems), problems
    assert any(p.startswith("mod.py: a no longer exists") for p in problems), problems


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
    assert prose <= PROSE_CEILING or ratio <= RATIO_PIN, (
        f"docstring+comment lines grew {PROSE_CEILING} -> {prose} AND the ratio "
        f"{RATIO_PIN} -> {ratio:.4f} (budget {RATIO_BUDGET}). New code pays for "
        "its own prose at the pinned ratio; past that, say it once, in the layer "
        "that owns it (D58/P1)."
    )
    assert ratio > RATIO_BUDGET, (
        f"the package now meets the {RATIO_BUDGET} ratio ({ratio:.4f}) — "
        "set PROSE_CEILING and RATIO_PIN = None"
    )


# ---------------------------------------------------------------------------
# P3: decisions.md stays indexed, and an entry stays one screen
# ---------------------------------------------------------------------------

#: P3's numbers. The live file is capped and its entries are capped; the
#: archive is history written before the rule, so only its index rows are
#: checked. Until F16c this was an awk one-liner pasted into agent briefs.
DECISIONS_LINES = 1300
DECISION_ENTRY_LINES = 31

#: Index rows reserved for an entry that was never written.
RESERVED_DECISIONS = frozenset({"D70"})


def _decision_entries(path: Path) -> dict[str, int]:
    """`Dn -> line count` per `## Dn:` section, up to the next `## ` heading."""
    entries: dict[str, int] = {}
    current, n = None, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if current:
                entries[current] = n
            match = re.match(r"## (D\d+):", line)
            current, n = (match.group(1) if match else None), 0
        n += 1
    if current:
        entries[current] = n
    return entries


def test_decisions_stay_indexed_and_short() -> None:
    plans = Path(__file__).resolve().parents[1] / "_plans"
    live_path = plans / "decisions.md"
    live_text = live_path.read_text(encoding="utf-8")
    live = _decision_entries(live_path)
    archived = _decision_entries(plans / "decisions-archive.md")

    n = len(live_text.splitlines())
    assert n <= DECISIONS_LINES, (
        f"decisions.md is {n} lines, over P3's {DECISIONS_LINES}. The ratchet is "
        "the archive, not trimming: move the oldest entries to "
        "decisions-archive.md and leave their index rows behind."
    )
    over = {d: size for d, size in live.items() if size > DECISION_ENTRY_LINES}
    assert not over, (
        f"decisions.md entries over {DECISION_ENTRY_LINES} lines: {over}. A "
        "number is a citation into findings.md, never a copy (P3, P4)."
    )
    both = sorted(set(live) & set(archived))
    assert not both, f"{both}: a full entry in both decisions.md and the archive"

    rows = set(re.findall(r"^\| (D\d+)", live_text, re.M))
    missing = sorted(set(archived) - rows)
    assert not missing, (
        f"archived {missing} has no index row in decisions.md — a moved entry "
        "keeps its one-line summary and status in the index (P3)."
    )
    orphans = sorted(rows - set(live) - set(archived) - RESERVED_DECISIONS)
    assert not orphans, f"index rows with no entry in either file: {orphans}"


# ---------------------------------------------------------------------------
# P4: a findings.md entry is one table plus <= 12 lines of reading
# ---------------------------------------------------------------------------

#: P4's number: prose lines after an entry's first table row (all its prose
#: when it has no table). The "one table" half is measured, not enforced (M1c).
FINDINGS_PROSE_LINES = 12

# heading -> prose lines today, for the entries over budget on 2026-09-04. Same
# three rules as ALLOWLIST: unlisted-over fails, grown fails, under-budget or
# gone fails until the row is deleted. A new row needs a D-entry (D58, P8).
FINDINGS_ALLOWLIST: dict[str, int] = {
    "Generation experiments": 15,
    "Data hygiene / QA": 19,
    "eval_v2 — applied 2026-09-03 (W10 item 4 / D42)": 13,
}


def _findings_entries(path: Path) -> dict[str, int]:
    """`heading -> prose lines` per section, up to the next heading of any level.

    Prose is a non-blank line that is not a `|` table row; only the prose after
    the first table row counts, so a group heading with intro prose and no
    table is measured on all of it.
    """
    entries: dict[str, int] = {}
    heading, prose, seen_table = None, 0, False
    for line in path.read_text(encoding="utf-8").splitlines():
        if re.match(r"^#{1,6} ", line):
            if heading is not None:
                entries[heading] = prose
            heading, prose, seen_table = line.lstrip("#").strip(), 0, False
            assert heading not in entries, f"duplicate heading {heading!r}"
        elif heading is None or not line.strip():
            continue
        elif line.lstrip().startswith("|"):
            if not seen_table:
                seen_table, prose = True, 0
        else:
            prose += 1
    if heading is not None:
        entries[heading] = prose
    return entries


def _check_findings(entries: dict[str, int], allowed: dict[str, int]) -> list[str]:
    problems: list[str] = []
    for heading, n in entries.items():
        ceiling = {"prose": allowed[heading]} if heading in allowed else {}
        problems += _check_scalar(repr(heading), "prose", FINDINGS_PROSE_LINES, n, ceiling)
    for heading in sorted(set(allowed) - set(entries)):
        problems.append(f"{heading!r} no longer exists — delete its allowlist entry")
    return problems


def test_findings_entries_stay_one_table_and_short() -> None:
    path = Path(__file__).resolve().parents[1] / "_plans" / "findings.md"
    problems = _check_findings(_findings_entries(path), FINDINGS_ALLOWLIST)
    assert not problems, "findings.md prose after the table (P4, D58):\n  " + "\n  ".join(problems)


def _findings_with(tmp_path: Path, prose_lines: int) -> Path:
    prose = "\n".join(f"reading {i}." for i in range(prose_lines))
    path = tmp_path / "findings.md"
    path.write_text(f"## Entry\n\nintro\n\n| a |\n|---|\n| 1 |\n\n{prose}\n")
    return path


def test_a_findings_entry_is_measured_after_its_table(tmp_path: Path) -> None:
    fits = _findings_entries(_findings_with(tmp_path, FINDINGS_PROSE_LINES))
    assert fits == {"Entry": FINDINGS_PROSE_LINES}
    assert _check_findings(fits, {}) == []
    over = _findings_entries(_findings_with(tmp_path, FINDINGS_PROSE_LINES + 1))
    problems = _check_findings(over, {})
    assert problems == [
        f"'Entry': prose is {FINDINGS_PROSE_LINES + 1} (budget "
        f"{FINDINGS_PROSE_LINES}) and is not allowlisted"
    ]
    assert _check_findings(over, {"Entry": FINDINGS_PROSE_LINES + 1}) == []


# ---------------------------------------------------------------------------
# P6: CLAUDE.md is an operating manual
# ---------------------------------------------------------------------------

#: P6's two numbers. The file reached 244 lines before F7a cut it to 160, and
#: P6 was the only budget in "Budgets and layers" with no test behind it —
#: which is how it grew the first time. F12e.
CLAUDE_MD_LINES = 160
CLAUDE_MD_BULLET_LINES = 6


def _bullet_runs(path: Path) -> list[tuple[int, int]]:
    """`(line number, length)` per top-level `- ` bullet, continuations included.

    A bullet runs until the next bullet at any depth, a blank line, or a
    heading. A nested bullet is a bullet: P8 is one line plus five sub-rules,
    not one ten-line rule.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    runs: list[tuple[int, int]] = []
    start = None
    for i, line in enumerate(lines, 1):
        if re.match(r"^\s*[-*] ", line):
            if start is not None:
                runs.append((start, i - start))
            start = i
        elif start is not None and (line.startswith("#") or (not line.strip())):
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(lines) + 1 - start))
    return runs


def test_claude_md_stays_an_operating_manual() -> None:
    path = Path(__file__).resolve().parents[1] / "CLAUDE.md"
    n = len(path.read_text(encoding="utf-8").splitlines())
    assert n <= CLAUDE_MD_LINES, (
        f"CLAUDE.md is {n} lines, over P6's {CLAUDE_MD_LINES}. Every bullet is a "
        "rule plus a pointer; the story of how the rule was learned belongs in "
        "_plans/decisions.md (P1)."
    )
    over = [(ln, size) for ln, size in _bullet_runs(path) if size > CLAUDE_MD_BULLET_LINES]
    assert not over, (
        f"CLAUDE.md bullets over {CLAUDE_MD_BULLET_LINES} lines at "
        f"{[ln for ln, _ in over]} — cut the anecdote to a Dn pointer (P1, P6)."
    )
