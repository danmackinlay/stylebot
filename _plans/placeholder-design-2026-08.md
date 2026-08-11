# How models treat protected spans: findings for parser/masker design

Evidence for `protected_spans` in `src/stylebot/markdown.py`, and for any future
masking scheme here.

Measured 2026-08-11 in the **livingthing** repo, during a model-roster bench for
`ai-preen`. The experiments ran against livingthing's `pandoc_mask`, but the
problem is identical to stylebot's: hand a model text containing regions it must
not touch, and get those regions back intact. The findings transfer; the code
does not, since the two segmenters are a deliberate double implementation (see
the `segment_for_edit` conformance note in livingthing's CLAUDE.md).

Source data and reproduction scripts are in the livingthing checkout
(`_tmp/model-roster/`, gitignored — copy anything wanted before it is swept).
Companion write-ups there: `_plans/masking-url-integrity-2026-08.md` and
`_plans/openrouter-operational-notes-2026-08.md`.

---

## The headline: positional plausibility, not character exoticism

The intuition we started from was that `〈URL_0000〉` fails because U+2329/U+232A
and an underscore-numbered token look like nothing in training data. Testing
three schemes on identical chunks says that intuition is **wrong in its
mechanism**, though right that the current scheme is weak.

12 link-bearing chunks, ~122 placeholders per cell, retention = placeholder
returned verbatim:

| scheme | gemini-3.1-flash-lite | **qwen3.6-35b-a3b** | glm-4.7-flash |
|---|---|---|---|
| `〈URL_0000〉` — Unicode brackets | 100% | **45.9%** | 0% |
| `https://example.com/u/<sha1>` — URL-shaped | 100% | **68.9%** | 6.0% |
| `[[URL_0000]]` — plain ASCII | 100% | **37.3%** | 12.9% |

If exotic characters were the problem, the ASCII variant would have won. It
came **last** — worse than the Unicode it was meant to fix.

What predicts survival is whether the token **looks like the kind of thing that
belongs in that slot**. A URL-shaped string sitting between `](` and `)` reads
as a URL and gets copied through. `〈URL_0000〉` and `[[URL_0000]]` both read as
"some identifier the editor has been handed", and an editing model treats
identifiers as fair game.

**Design rule: a placeholder should resemble what it replaces, not announce
itself.** For a parser with several protected classes, that means *different
shapes per class*, not one uniform sigil:

| protected span | plausible stand-in |
|---|---|
| link URL | `https://example.invalid/u/<hash>-<n>` |
| inline code | a short plausible identifier in backticks |
| inline math | a short plausible expression in `$…$` |
| display math | a plausible short equation block |

This is the opposite of the instinct to make placeholders maximally distinctive
so they are easy to find again. Distinctiveness helps *your* regex and hurts
*the model's* copying behaviour. Resolve it by making the token plausible in
shape but reserved in content (`example.invalid` is RFC-reserved and can never
collide with a real URL).

## Retention is a model property, and the spread is enormous

```
gemini-3.1-flash-lite   100%    on every scheme
qwen3.6-35b-a3b       37–69%    scheme-sensitive
glm-4.7-flash          0–13%    failing the task outright
```

Two consequences for a parser:

- **Never assume placeholders survive.** Verify on return, always.
- **Scheme choice only matters in the middle band.** For a model at 100% it is
  wasted effort; for a model at 6% no sigil design will save you. Optimising
  the placeholder is worthwhile only for models that are nearly good enough.

## Failure modes observed, and what each implies

### Digit insertion inside the token

```
emitted by masker:  〈URL_0020〉        (format is f"…{i:04d}…", always 4 digits)
returned by model:  〈URL_00020〉       (five digits — a zero inserted)
```

A restoration map keyed on the exact string misses, and the placeholder leaks
into the output where the content belonged. **Implication:** the *detector*
regex must be looser than the *emitter* format (ours is `〈([A-Z_]+)_(\d+)〉`,
digit-count-agnostic, which is why this was caught rather than shipped). Match
loosely, then reconcile by position.

### Truncation, misattributed to the mask

The log line `Some placeholders not restored: [('URL','0012'),('URL','0013'),
('URL','0014')]` names the highest-numbered placeholders in the chunk. They
were not mangled — the model stopped emitting before reaching them. Perfect
correlation over 311 runs:

```
chunks with output <90% of input length   12 of 311
  of 23 reverted chunks    12 were short   (52%)
  of 288 clean chunks       0 were short   (0%)
```

**Implication:** check for truncation *before* attributing damage to the
model's editing, and report it as its own failure. Otherwise every downstream
analysis of "which spans does this model damage" is contaminated by "this model
ran out of budget".

A second, nastier variant: reasoning models that consume the whole token budget
and emit **zero characters**. `out_tok=15001, text=""`. That arrives looking
like "the model deleted everything".

### Whole-span deletion

```
deepseek-v4-flash   10 links → 0
kimi-k2.6            7 links → 0
qwen3.6-35b         18 links → 17     ← the only genuine single-span drop
```

Five of six were the empty-output case above. Real single-span deletion is
rare, but it happens and nothing about the placeholder shape prevents it.

### Duplicate collapse — a bug in the *mask*, not the model

Content-addressed placeholders (hash of the URL) mean two occurrences of the
same URL produce the same token. Duplicates collapse, positional identity is
lost, and reordering becomes undetectable. Observed as 6 spurious "duplicates"
that were the masker's doing.

**Implication:** any content-addressed scheme needs a position component —
`<hash>-<n>` — so the map stays a bijection. And use a stable digest, never
Python's `hash()`, which is salted per process via `PYTHONHASHSEED` and would
mask the same input differently on every run, breaking retries and resumes.

## Verification design: one label per distinguishable failure

Our guard compares ordered lists and reports a single reason:

```python
if orig_urls != edit_urls:
    return False, "link_url_changed", …
```

That one label covers deletion, addition, reordering and genuine rewriting.
It cost us a wrong conclusion: reverts labelled `link_url_changed` were read as
"the models rewrite URLs", when in fact **no model rewrote a URL to a different
real URL in 63 runs**. The apparent rewrites were a positional `zip()` diff over
a list with one deletion, which reports every subsequent element as changed.

Two rules:

1. **Distinguish the failures you will act on differently.** `count_changed`,
   `order_changed` and `content_changed` want different fixes; collapsing them
   guarantees somebody chases the wrong one.
2. **Align before diffing.** Use LCS/`difflib`, not `zip()`. A single deletion
   otherwise manufactures N false positives.

## Coverage: the syntaxes a masker forgets

Static audit over 8,881 editable chunks in this corpus:

```
bare url                 191   ← unmasked AND unchecked by the guard
reference link [t][r]     11
reference def [r]: url     6
autolink <https://…>       3
```

`mask_inline_elements` handles `[text](url)` because it walks the Pandoc AST for
`Link` nodes — but the *guard* uses a regex that matches only the same
inline form. So a bare URL is invisible to both: a model can mangle one and
nothing notices.

**Implication for stylebot:** enumerate the syntaxes a construct can take
before deciding coverage is complete. For links alone that is inline,
reference, reference-definition, autolink and bare — five forms, of which the
obvious one is a minority of the *risk* (the 191 bare URLs dwarf everything
else here).

Related: masking-integrity failure currently **fails open** — when
`validate_masking_integrity` fails, the edit proceeds with raw content exposed.
It fires on 1 chunk in 8,881, so failing closed costs almost nothing and
removes a silent-corruption path.

## Ordering of protection vs. transformation

One more, from an earlier bug in this repo and worth carrying into any parser:
smart-quote curling was applied **after** unmasking, so everything the mask had
just protected got rewritten anyway (`$P'$` → `$P’$`, `` `P.get("x")` `` →
`` `P.get(“x”)` ``). The fix was a curler that does its own masking
(`smart_quotes.smarten`).

**Implication:** any post-processing step that runs after unmasking needs its
own protection pass. Protection is not a phase you exit; it is a property each
transformation must independently respect.

---

## Acting on this here

The relevant surface is `protected_spans` in `src/stylebot/markdown.py`. In
priority order:

1. Per-class plausible placeholders rather than one uniform sigil shape.
2. Position component in every placeholder; stable digest if content-addressed.
3. Loose detector regex, strict emitter format.
4. Truncation checked and reported before span-damage attribution.
5. Distinct verification labels; align sequences before diffing.
6. Enumerate all syntactic forms per construct; fail closed on integrity errors.

Reproduce the scheme comparison with livingthing's
`_tmp/model-roster/bench_placeholder2.py` (`PH_MODELS` env var). **Re-measure
before choosing a scheme:** retention is strongly model-dependent (100% vs 46%
vs 6% across the three models tested), and stylebot's target model is not the
one these numbers came from. A styler fine-tune in particular may behave
differently again, since it has been trained on this exact corpus shape.
