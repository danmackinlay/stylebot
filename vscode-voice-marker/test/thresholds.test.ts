/**
 * Detector selection and threshold lookup (stylebot D106) — pure functions,
 * no VS Code, no sidecar.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import { activeDetector, armedDetectors, thresholdsFor } from "../src/thresholds";

test("a pre-D106 sidecar arms exactly its one named head", () => {
  assert.deepEqual(armedDetectors({ name: "voice-clf" }), ["voice-clf"]);
  assert.deepEqual(armedDetectors(null), []);
});

test("the configured detector wins only if the sidecar armed it", () => {
  const meta = { name: "voice-clf", detector: "voice-clf", detectors: ["voice-clf", "sapling-x"] };
  assert.equal(activeDetector("sapling-x", meta), "sapling-x");
  assert.equal(activeDetector("gptzero", meta), "voice-clf"); // not armed: the default
  assert.equal(activeDetector("", meta), "voice-clf");
  assert.equal(activeDetector("sapling-x", { name: "voice-clf" }), "voice-clf");
  assert.equal(activeDetector("sapling-x", null), null);
});

test("an array setting is every detector's edges; an object is keyed by name", () => {
  assert.deepEqual(thresholdsFor([0.6, 0.5], "anything"), [0.5, 0.6]);
  const keyed = { "voice-clf": [0.27, 0.32], "sapling-x": [0.9, 0.95, 0.99] };
  assert.deepEqual(thresholdsFor(keyed, "sapling-x"), [0.9, 0.95, 0.99]);
  assert.deepEqual(thresholdsFor(keyed, "voice-clf"), [0.27, 0.32]);
});

test("a detector without edges is uncalibrated, never silently bucketed", () => {
  assert.equal(thresholdsFor({ "voice-clf": [0.5] }, "sapling-x"), null);
  assert.equal(thresholdsFor({ "sapling-x": [] }, "sapling-x"), null);
  assert.equal(thresholdsFor({ "sapling-x": ["0.5" as unknown as number] }, "sapling-x"), null);
  assert.equal(thresholdsFor(undefined, "voice-clf"), null);
});
