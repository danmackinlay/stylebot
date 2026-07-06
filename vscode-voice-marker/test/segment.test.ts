/**
 * Cross-language segmentation parity — the TS half of the contract pinned by
 * stylebot/tests/test_marker_segmentation.py. Both tests assert the same
 * checked-in expected_segments.json over the same fixture .qmd.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import { proseParagraphs } from "../src/segment";

// out/test/segment.test.js → out/test → out → vscode-voice-marker → stylebot
const FIXTURES = join(__dirname, "..", "..", "..", "tests", "fixtures");
const fixture = readFileSync(join(FIXTURES, "segmentation.qmd"), "utf8");
const expected: { paragraphs: string[] } = JSON.parse(
  readFileSync(join(FIXTURES, "expected_segments.json"), "utf8"),
);

test("fixture paragraphs match the Python-pinned expectation", () => {
  const paragraphs = proseParagraphs(fixture, 0);
  assert.deepEqual(
    paragraphs.map((p) => p.text),
    expected.paragraphs,
  );
});

test("offsets slice back to the paragraph text", () => {
  for (const p of proseParagraphs(fixture, 0)) {
    assert.equal(fixture.slice(p.start, p.end), p.text);
  }
});

test("minChars drops short paragraphs only", () => {
  const texts = proseParagraphs(fixture, 80).map((p) => p.text);
  assert.ok(!texts.includes("Short."));
  assert.ok(texts.length < expected.paragraphs.length);
  for (const t of texts) {
    assert.ok(expected.paragraphs.includes(t));
  }
});

test("frontmatter-less documents still segment", () => {
  const doc = "Just one paragraph.\n\nAnd a second one here.\n";
  const texts = proseParagraphs(doc, 0).map((p) => p.text);
  assert.deepEqual(texts, ["Just one paragraph.", "And a second one here."]);
});
