/**
 * Graded P(slop) decorations — one fixed-style TextEditorDecorationType per
 * intensity bucket (per-range colour variation isn't possible in the VS Code
 * API; hover text is). Bucket 0 (below the first threshold) draws nothing.
 */

import * as vscode from "vscode";

// Bucket colours above the first threshold: amber → red ramp, background kept
// faint (the tint is ambient suspicion, not an error squiggle).
const BUCKET_STYLES = [
  { background: "rgba(255, 200, 0, 0.06)", ruler: "rgba(255, 200, 0, 0.5)" },
  { background: "rgba(255, 160, 0, 0.10)", ruler: "rgba(255, 160, 0, 0.7)" },
  { background: "rgba(255, 110, 0, 0.14)", ruler: "rgba(255, 110, 0, 0.85)" },
  { background: "rgba(255, 60, 40, 0.18)", ruler: "rgba(255, 60, 40, 1.0)" },
];

export interface ScoredRange {
  range: vscode.Range;
  score: number;
}

export class Decorations {
  private types: vscode.TextEditorDecorationType[];
  private thresholds: number[];

  constructor(thresholds: number[]) {
    this.thresholds = [...thresholds].sort((a, b) => a - b);
    this.types = this.thresholds.map((_, i) => {
      const style = BUCKET_STYLES[Math.min(i, BUCKET_STYLES.length - 1)];
      return vscode.window.createTextEditorDecorationType({
        isWholeLine: true,
        backgroundColor: style.background,
        overviewRulerColor: style.ruler,
        overviewRulerLane: vscode.OverviewRulerLane.Right,
      });
    });
  }

  /** Bucket index for a score: 0 = unmarked, 1..N = increasing suspicion. */
  bucket(score: number): number {
    let b = 0;
    for (const t of this.thresholds) {
      if (score >= t) {
        b += 1;
      }
    }
    return b;
  }

  apply(editor: vscode.TextEditor, scored: ScoredRange[], modelName: string): void {
    const perBucket: vscode.DecorationOptions[][] = this.types.map(() => []);
    for (const { range, score } of scored) {
      const b = this.bucket(score);
      if (b === 0) {
        continue;
      }
      perBucket[b - 1].push({
        range,
        hoverMessage: new vscode.MarkdownString(
          `**P(slop) ${score.toFixed(2)}** — ${modelName} (graded suspicion, not a verdict)`,
        ),
      });
    }
    this.types.forEach((type, i) => editor.setDecorations(type, perBucket[i]));
  }

  clear(editor: vscode.TextEditor): void {
    for (const type of this.types) {
      editor.setDecorations(type, []);
    }
  }

  dispose(): void {
    for (const type of this.types) {
      type.dispose();
    }
  }
}
