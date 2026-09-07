import assert from "node:assert/strict";
import { test } from "node:test";
import { paragraphAround, SEM_CLOSE, SEM_OPEN, toggleSemWrap } from "../src/sem";

test("toggleSemWrap wraps a bare span and unwraps a wrapped one, keeping edge whitespace", () => {
  assert.equal(toggleSemWrap("An added claim."), `${SEM_OPEN}An added claim.${SEM_CLOSE}`);
  assert.equal(toggleSemWrap(" An added claim. "), ` ${SEM_OPEN}An added claim.${SEM_CLOSE} `);
  assert.equal(toggleSemWrap(`${SEM_OPEN}An added claim.${SEM_CLOSE}`), "An added claim.");
  assert.equal(toggleSemWrap(` ${SEM_OPEN} spaced ${SEM_CLOSE}\n`), " spaced\n");
  assert.equal(toggleSemWrap("   "), "   ");
});

test("paragraphAround finds the blank-line paragraph holding the offset", () => {
  const text = "first para\nstill first\n\nsecond para\n\nthird";
  assert.deepEqual(paragraphAround(text, 3), [0, 22]);
  assert.deepEqual(paragraphAround(text, 26), [24, 35]);
  assert.deepEqual(paragraphAround(text, text.length), [37, text.length]);
});
