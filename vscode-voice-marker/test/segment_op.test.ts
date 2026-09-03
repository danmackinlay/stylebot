/**
 * The `segment` op over a real child process.
 *
 * The extension no longer segments: `stylebot.serve.prose_paragraphs` does, and
 * this package only has to ask and index the answer. So what is left to test
 * here is the client half of that seam — the request shape (`min_chars`, not
 * `minChars`), and that the returned offsets index the string that was SENT,
 * which is what lets `scoreEditor` slice its own snapshot instead of re-reading
 * a document the user may have typed into meanwhile.
 *
 * The child is a stub NDJSON server rather than `ai-style serve`: the real one
 * loads a 240 MB embedding model, and the Python side of this contract is
 * pinned by stylebot's `tests/test_serve.py`.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import { Sidecar } from "../src/sidecar";

/** A stub sidecar: answers `info`, and `segment` with one span per blank-line
 * paragraph — enough to exercise the wire, not a segmenter. */
const STUB = `
let buf = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  buf += chunk;
  let nl;
  while ((nl = buf.indexOf("\\n")) >= 0) {
    const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
    if (!line.trim()) continue;
    const req = JSON.parse(line);
    if (req.op === "info") { process.stdout.write(JSON.stringify({ id: req.id, meta: { name: "stub" } }) + "\\n"); continue; }
    if (req.op === "segment") {
      const segments = []; let pos = 0;
      for (const part of req.text.split("\\n\\n")) {
        if (part.trim() && part.length >= (req.min_chars || 0)) {
          segments.push({ start: pos, end: pos + part.length, kind: "prose", context: null });
        }
        pos += part.length + 2;
      }
      process.stdout.write(JSON.stringify({ id: req.id, segments }) + "\\n");
      continue;
    }
    process.stdout.write(JSON.stringify({ id: req.id, error: "unknown op" }) + "\\n");
  }
});
`;

function stubSidecar(): Sidecar {
  return new Sidecar(["node", "-e", STUB], undefined, { onState: () => {}, onStderr: () => {} });
}

test("segment offsets index the text that was sent", async () => {
  const sidecar = stubSidecar();
  sidecar.start();
  try {
    const text = "First paragraph here.\n\nSecond paragraph here.";
    const segments = await sidecar.segment(text, 0);
    assert.deepEqual(
      segments.map((s) => text.slice(s.start, s.end)),
      ["First paragraph here.", "Second paragraph here."],
    );
  } finally {
    sidecar.dispose();
  }
});

test("minChars travels as the protocol's min_chars", async () => {
  const sidecar = stubSidecar();
  sidecar.start();
  try {
    const text = "Tiny.\n\nA paragraph comfortably past the minimum.";
    const segments = await sidecar.segment(text, 20);
    assert.deepEqual(
      segments.map((s) => text.slice(s.start, s.end)),
      ["A paragraph comfortably past the minimum."],
    );
  } finally {
    sidecar.dispose();
  }
});

test("a sidecar too old to know the op rejects rather than hanging", async () => {
  const sidecar = new Sidecar(
    ["node", "-e", STUB.replace('req.op === "segment"', "false")],
    undefined,
    { onState: () => {}, onStderr: () => {} },
  );
  sidecar.start();
  try {
    await assert.rejects(() => sidecar.segment("some prose", 0), /unknown op/);
  } finally {
    sidecar.dispose();
  }
});
