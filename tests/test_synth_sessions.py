"""Phase 2 run engine: parallelism, multi-turn sessions, budgets, reflow.

Carved verbatim out of ``test_synth.py``. Session composition is a runtime
outcome, so the invariants here are that keys never fold it and that a session
which ends early sheds nothing.
"""

from __future__ import annotations

from synth_helpers import _fake, _targets

from stylebot import synth

# ---------------------------------------------------------------------------
# Parallelism + live sessions (async engine, window-position covariates)
# ---------------------------------------------------------------------------


def test_parallel_stateless_run_writes_each_pair_once(tmp_path):
    import json

    targets = _targets(10)
    gens = [_fake("a"), _fake("b")]
    first = synth.synthesize_pairs(targets, tmp_path / "c", gens, max_workers=4)
    assert first.written == 10
    assert set(first.per_generator) == {"a", "b"}  # hash split, not exactly even
    assert sum(first.per_generator.values()) == 10

    second = synth.synthesize_pairs(targets, tmp_path / "c", gens, max_workers=4)
    assert second.written == 0 and second.skipped_existing == 10

    keys = [
        json.loads(ln)["meta"]["synth_key"]
        for ln in (tmp_path / "c" / "pairs.jsonl").read_text().splitlines()
    ]
    assert len(keys) == 10 == len(set(keys))


def test_async_injected_generator_works(tmp_path):
    async def gen(text, history=None):
        return "[async-slop] " + text

    result = synth.synthesize_pairs(
        _targets(2), tmp_path / "c", [synth.Generator("ag", generate=gen)]
    )
    assert result.written == 2


def test_keys_are_content_only_in_all_modes():
    # Keys never fold session membership: the same cell keys identically
    # stateless, in a session, or after a reflow — corpus resume must survive
    # the blog growing AND sessions re-chunking (2026-07-21 post-mortem).
    t1, t2 = _targets(2)
    g = synth.Generator(name="g")
    k1, k2 = synth._synth_key("g", t1.text), synth._synth_key("g", t2.text)
    keyed = [(t1, g, "", k1), (t2, g, "", k2)]
    [stateless1, stateless2] = synth._plan_sessions(keyed, session_turns=1)
    [sessioned] = synth._plan_sessions(keyed, session_turns=2)
    assert stateless1.session_id == "" and sessioned.session_id != ""
    assert (
        [t.key for t in sessioned.turns]
        == [stateless1.turns[0].key, stateless2.turns[0].key]
        == [k1, k2]
    )
    # Reflow keeps keys, renumbers indices, and re-derives the session label
    # from the actual composition.
    respun = synth._respin_session(g, sessioned.turns[1:])
    assert [t.key for t in respun.turns] == [k2]
    assert [t.index for t in respun.turns] == [1]
    assert respun.session_id != sessioned.session_id


def test_identical_cells_collapse_within_a_run(tmp_path):
    # Two chunks with identical (config, context, text) are ONE cell: a single
    # pair is generated, not two. (Formerly the session fold gave duplicated
    # text distinct per-turn keys; content-only keys deliberately collapse it.)
    t = _targets(1)[0]
    dup = synth.Target(text=t.text, source="elsewhere.qmd", chunk_index=9, chunk_total=9)
    result = synth.synthesize_pairs(
        [t, dup],
        tmp_path / "c",
        [synth.Generator("g", generate=lambda x, history=None: "[slop] " + x)],
        session_turns=2,
    )
    assert result.written == 1
    assert result.skipped_existing == 1


def test_session_history_accumulates_and_covariates_recorded(tmp_path):
    import json

    targets = _targets(3)
    seen_hist: list[list[dict]] = []

    def gen(text, history=None):
        seen_hist.append(list(history or []))
        return synth.GenOutput("[slop] " + text, {"prompt_tokens": 100 * (len(seen_hist))})

    result = synth.synthesize_pairs(
        targets,
        tmp_path / "c",
        [synth.Generator("g", generate=gen)],
        session_turns=3,
        context_windows={"g": 10_000},
    )
    assert result.written == 3
    assert [len(h) for h in seen_hist] == [0, 2, 4]  # turn N sees N-1 exchanges
    assert seen_hist[1][0] == {"role": "user", "content": targets[0].text}
    assert seen_hist[1][1]["content"] == "[slop] " + targets[0].text

    recs = [json.loads(ln) for ln in (tmp_path / "c" / "pairs.jsonl").read_text().splitlines()]
    assert [r["meta"]["gen"]["session_turn"] for r in recs] == [1, 2, 3]
    assert len({r["meta"]["gen"]["session_id"] for r in recs}) == 1
    assert recs[1]["meta"]["gen"]["context_window"] == 10_000
    assert recs[1]["meta"]["gen"]["window_fill"] == round(200 / 10_000, 4)


def test_session_resume_packs_only_missing_cells(tmp_path):
    # Resume is a set difference over content keys: the retried cell runs in a
    # FRESH session (no replay of recorded turns — window position restarts and
    # is recorded, not keyed).
    targets = _targets(3)
    calls = {"n": 0}

    def flaky(text, history=None):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return "[slop] " + text

    first = synth.synthesize_pairs(
        targets, tmp_path / "c", [synth.Generator("g", generate=flaky)], session_turns=3
    )
    assert first.written == 2 and len(first.errors) == 1

    hists: list[list[dict]] = []

    def good(text, history=None):
        hists.append(list(history or []))
        return "[slop] " + text

    second = synth.synthesize_pairs(
        targets, tmp_path / "c", [synth.Generator("g", generate=good)], session_turns=3
    )
    assert second.written == 1 and second.skipped_existing == 2
    # Exactly one generation, starting from clean history.
    assert hists == [[]]


def test_turn_error_reflows_remainder(tmp_path):
    # Needs a 2/2 split so `bad` fills exactly one 2-turn session. Rendezvous
    # assignment is multinomial, so pick the four targets that land that way
    # rather than assuming the first four do (bad: 0,2 / good: 1,5).
    targets = [t for i, t in enumerate(_targets(6)) if i in (0, 1, 2, 5)]

    def bad(text, history=None):
        raise RuntimeError("api down")

    result = synth.synthesize_pairs(
        targets,
        tmp_path / "c",
        [
            synth.Generator("bad", generate=bad),
            synth.Generator("good", generate=lambda t, history=None: "[slop] " + t),
        ],
        session_turns=2,
    )
    assert result.per_generator.get("good") == 2
    assert "bad" not in result.per_generator
    # The failed turn is shed, the REST of its session reflows and is attempted
    # (and fails too, here) — errors name every attempted-and-failed turn, so
    # nothing is silently dropped.
    assert len(result.errors) == 2
    assert result.reflow_sessions == 1 and result.reflowed_turns == 1
    _key, msg = result.errors[0]
    # The error is attributable: generator, strategy, effort, and session turn.
    assert msg.startswith("[bad strategy=polish effort=off turn=1] RuntimeError")


def test_estimate_next_prompt():
    # Pure token-estimate arithmetic (hoisted out of the session loop so it is
    # directly testable): measured path = last prompt + last reply's visible
    # tokens + next passage at ~4 chars/token; unmeasured path = pure chars/4.
    history = [{"role": "user", "content": "x" * 400}, {"role": "assistant", "content": "y" * 400}]
    assert synth._estimate_next_prompt(history, 1000, 50, "z" * 400) == 1000 + 50 + 100
    assert synth._estimate_next_prompt(history, 0, 0, "z" * 400) == (800 + 400) // 4


def test_session_token_cap():
    g = synth.Generator("x")
    assert synth._session_token_cap(g, None, 32_000) == 32_000
    assert synth._session_token_cap(g, 10_000, 32_000) == 8_000  # 0.8 x window wins
    assert synth._session_token_cap(g, None, None) is None
    g_budget = synth.Generator("x", session_budget=5_000)
    assert synth._session_token_cap(g_budget, 10_000, 32_000) == 5_000  # per-generator beats both


def test_budget_bind_reflows_instead_of_dropping(tmp_path):
    # The 2026-07-21 post-mortem: a binding token budget must END sessions, not
    # DISCARD their remaining turns. Every planned turn gets written, across as
    # many fresh sessions as the budget requires.
    import json

    targets = _targets(3)

    def gen(text, history=None):
        return synth.GenOutput("[slop] " + text, {"prompt_tokens": 10_000, "completion_tokens": 50})

    result = synth.synthesize_pairs(
        targets,
        tmp_path / "c",
        [synth.Generator("g", generate=gen)],
        session_turns=3,
        session_max_tokens=100,
    )
    assert result.written == 3  # budget binds after every turn — but nothing is lost
    assert not result.errors  # a budget stop is not an error
    assert result.budget_bound_sessions == 2
    assert result.reflow_sessions == 2 and result.reflowed_turns == 3  # (2 leftover) + (1 leftover)
    # Each pair ran turn 1 of its own session: depth restarted, honestly recorded.
    recs = [json.loads(ln) for ln in (tmp_path / "c" / "pairs.jsonl").read_text().splitlines()]
    assert [r["meta"]["gen"]["session_turn"] for r in recs] == [1, 1, 1]
    assert len({r["meta"]["gen"]["session_id"] for r in recs}) == 3


def test_generator_session_budget_beats_global(tmp_path):
    targets = _targets(3)

    def gen(text, history=None):
        return synth.GenOutput("[slop] " + text, {"prompt_tokens": 60})

    g = synth.Generator("g", generate=gen, session_budget=50)
    result = synth.synthesize_pairs(
        targets, tmp_path / "c", [g], session_turns=3, session_max_tokens=1_000_000
    )
    # Only the per-generator budget (50 < the huge global) can have bound here.
    assert result.written == 3
    assert result.budget_bound_sessions == 2


def test_window_cap_stops_session(tmp_path):
    targets = _targets(3)

    def gen(text, history=None):
        return synth.GenOutput("[slop] " + text, {"prompt_tokens": 790, "completion_tokens": 50})

    result = synth.synthesize_pairs(
        targets,
        tmp_path / "c",
        [synth.Generator("g", generate=gen)],
        session_turns=3,
        session_max_tokens=None,
        context_windows={"g": 1000},
    )
    # cap = 0.8 * 1000 = 800; next-turn estimate from prompt_tokens=790 exceeds
    # it, ending each session after one turn — reflow still writes all three.
    assert result.written == 3
    assert result.budget_bound_sessions == 2


def test_session_loop_uses_begin_session_per_session(tmp_path):
    made = []

    def begin():
        def g(text, history=None):
            return "[slop] " + text

        made.append(g)
        return g

    gen = synth.Generator("g", generate=lambda t, history=None: "[slop] " + t, begin_session=begin)
    result = synth.synthesize_pairs(_targets(4), tmp_path / "c", [gen], session_turns=2)
    assert result.written == 4
    assert len(made) == 2  # one fresh closure per session

    made.clear()
    stateless = synth.synthesize_pairs(_targets(2), tmp_path / "c2", [gen])
    assert stateless.written == 2 and made == []  # stateless never begins a session
