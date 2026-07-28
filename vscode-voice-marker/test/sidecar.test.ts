/**
 * Sidecar request-timeout behaviour.
 *
 * Regression for 2026-07-28: a sidecar wedged on a dead socket (tinker
 * backend, stale network) settled no promise and never exited, so the editor
 * hung forever. `request()` must now fail on a budget, and a repeatedly
 * unresponsive child must be restarted rather than nursed.
 *
 * Drives a real child process — `sleep`, which models the wedged case
 * exactly: alive, stdin writable, never a byte on stdout. (`cat` is NOT a
 * substitute: it echoes our own request line back, and since that line
 * carries the request `id` the client reads it as a valid response and
 * settles the promise.)
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import { Sidecar, SidecarState } from "../src/sidecar";

/** A sidecar whose child never answers: alive, accepts stdin, silent stdout. */
function hangingSidecar(timeouts: Record<string, number>, states: [SidecarState, string?][]) {
  return new Sidecar(
    ["sleep", "120"],
    undefined,
    {
      onState: (state, detail) => states.push([state, detail]),
      onStderr: () => {},
    },
    timeouts,
  );
}

test("a request that outruns its budget rejects instead of hanging", async () => {
  const states: [SidecarState, string?][] = [];
  const sidecar = hangingSidecar({ info: 60_000, score: 50 }, states);
  sidecar.start();
  try {
    await assert.rejects(
      () => sidecar.score(["some prose"]),
      (err: Error) => {
        assert.match(err.message, /'score' timed out after 0s|'score' timed out/);
        assert.match(err.message, /not responding/);
        return true;
      },
    );
  } finally {
    sidecar.dispose();
  }
});

test("two consecutive timeouts restart the presumed-wedged child", async () => {
  const states: [SidecarState, string?][] = [];
  const sidecar = hangingSidecar({ info: 60_000, score: 50 }, states);
  sidecar.start();
  try {
    await sidecar.score(["a"]).catch(() => {});
    await sidecar.score(["b"]).catch(() => {});
    const restart = states.find(([, detail]) => detail?.includes("unresponsive"));
    assert.ok(restart, `expected an unresponsive-restart state, got ${JSON.stringify(states)}`);
    assert.equal(restart![0], "starting");
  } finally {
    sidecar.dispose();
  }
});

test("dispose settles in-flight requests rather than orphaning them", async () => {
  const states: [SidecarState, string?][] = [];
  const sidecar = hangingSidecar({ info: 60_000, score: 60_000 }, states);
  sidecar.start();
  const inflight = sidecar.score(["never answered"]);
  sidecar.dispose();
  await assert.rejects(() => inflight, /disposed/);
});
