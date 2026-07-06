"""Cross-language segmentation parity for the VS Code marker.

The marker extension (vscode-voice-marker/src/segment.ts) re-implements
stylebot's segmentation so paragraph scores stay on-distribution:
`segment_for_edit` protected blocks + `split_paragraphs` blank-line splitting,
plus the two marker-side drops (YAML frontmatter, heading lines) that the
training pipeline handles upstream.

This test pins the *Python* composition over a shared fixture and asserts it
matches the checked-in `expected_segments.json`, which the TS unit test
(vscode-voice-marker/test/segment.test.ts) asserts against too — same fixture,
same expectation, both languages. If stylebot's segmentation changes, run with
STYLEBOT_REGEN_SEGMENTS=1 to regenerate the JSON, then re-run the TS test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from stylebot.lib import segment_for_edit, split_paragraphs

FIXTURE = Path(__file__).parent / "fixtures" / "segmentation.qmd"
EXPECTED = Path(__file__).parent / "fixtures" / "expected_segments.json"


def _strip_frontmatter(content: str) -> str:
    """Blank out leading YAML frontmatter (mirror of segment.ts frontmatterSpan).

    Replaced with an equal-length spacer rather than removed, so downstream
    prose keeps its character offsets in the original document.
    """
    if not content.startswith("---\n"):
        return content
    for close in ("\n---\n", "\n...\n"):
        idx = content.find(close, 3)
        if idx >= 0:
            end = idx + len(close)
            return "\n" * end + content[end:]
    return content


def _is_heading(paragraph: str) -> bool:
    stripped = paragraph.lstrip(" \t")
    return stripped.startswith("#") and stripped.lstrip("#").startswith((" ", "\t"))


def marker_paragraphs(content: str) -> list[str]:
    """The scoreable prose paragraphs, per the marker's segmentation contract."""
    content = _strip_frontmatter(content)
    paragraphs: list[str] = []
    for segment, editable in segment_for_edit(content):
        if not editable:
            continue
        for para in split_paragraphs(segment):
            if para and not _is_heading(para):
                paragraphs.append(para)
    return paragraphs


def test_fixture_matches_expected_segments():
    paragraphs = marker_paragraphs(FIXTURE.read_text(encoding="utf-8"))

    if os.environ.get("STYLEBOT_REGEN_SEGMENTS"):
        EXPECTED.write_text(
            json.dumps({"paragraphs": paragraphs}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))["paragraphs"]
    assert paragraphs == expected


def test_fixture_exercises_every_protected_kind():
    # The fixture only guards parity if it actually contains the cases.
    content = FIXTURE.read_text(encoding="utf-8")
    protected = "".join(seg for seg, editable in segment_for_edit(content) if not editable)
    assert "```python" in protected
    assert "```sh" in protected  # fence adjacent to prose, no blank line
    assert "\\int_0^1" in protected  # $$math$$ with {#eq-...} attributes
    assert "> A blockquote" in protected
    assert "callout-tip" in protected  # nested ::: div

    paragraphs = marker_paragraphs(content)
    text = "\n\n".join(paragraphs)
    assert "must not be scored" not in text  # heading dropped
    assert "title: Segmentation parity fixture" not in text  # frontmatter dropped
    assert "Short." in paragraphs  # minChars is the caller's policy, not segmentation's
    assert any(p.startswith("- a list item") for p in paragraphs)
    assert any("adjacent fence" not in p for p in paragraphs)
