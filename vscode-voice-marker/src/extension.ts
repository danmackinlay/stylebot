/**
 * Voice Marker — mark paragraphs dan-or-bot with P(slop) from `ai-style serve`.
 *
 * Flow: activate on markdown/quarto → spawn the sidecar from
 * `voiceMarker.command` (dormant if unset) → on debounced edits, ASK THE
 * SIDECAR which spans are prose, score them, and paint graded decorations.
 * See stylebot/_plans/vscode-marker.md.
 *
 * The sidecar may arm several detectors (stylebot D106): `voiceMarker.detector`
 * (or *Switch Detector*) picks which one marks, its edges come from
 * `voiceMarker.thresholds` keyed by name, and `voiceMarker.trigger` decides
 * whether a keystroke, a save or only a command sends text — a hosted
 * detector spends per character, so the default idle rescoring is a choice.
 *
 * There is no segmenter in this package. There was one — a TypeScript port of
 * stylebot's splitter, pinned by a fixture — and it drifted in the three ways a
 * port always drifts: it had no `div_is_prose` hook (so callout prose was never
 * scored and never rewritable), no unclosed-fence handling, and its own heading
 * spelling. Its boundaries won, because `rewrite_candidates` re-segments only to
 * veto. Segmentation is now one `segment` NDJSON hop on the already-debounced
 * path; see stylebot `_plans/decisions.md` D44 for the measured cost.
 */

import * as vscode from "vscode";
import { Decorations } from "./decorations";
import { paragraphAround, toggleSemWrap } from "./sem";
import { ProseSegment, RewriteCandidate, Sidecar, SidecarState } from "./sidecar";
import { activeDetector, armedDetectors, thresholdsFor, ThresholdsSetting } from "./thresholds";

const LANGS = new Set(["markdown", "quarto"]);
type Trigger = "idle" | "save" | "manual";

/** Virtual documents backing the live rewrite-preview diff. */
class RewritePreview implements vscode.TextDocumentContentProvider {
  private contents = new Map<string, string>();
  private emitter = new vscode.EventEmitter<vscode.Uri>();
  onDidChange = this.emitter.event;

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.contents.get(uri.path) ?? "";
  }

  set(path: string, text: string): void {
    this.contents.set(path, text);
    this.emitter.fire(vscode.Uri.from({ scheme: "voice-marker", path }));
  }
}

const preview = new RewritePreview();

let sidecar: Sidecar | null = null;
let decorations: Decorations | null = null;
let statusBar: vscode.StatusBarItem;
let output: vscode.OutputChannel;
let debounceTimer: NodeJS.Timeout | undefined;
let scoreGeneration = 0;
/** The instrument currently marking (null until the sidecar's handshake). */
let activeName: string | null = null;
/** The last `score` reply's spend counters, for the status-bar tooltip. */
let lastStats: Record<string, number> | null = null;

function config() {
  const cfg = vscode.workspace.getConfiguration("voiceMarker");
  return {
    command: cfg.get<string[]>("command", []),
    cwd: cfg.get<string>("cwd", "") || vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
    thresholds: cfg.get<ThresholdsSetting>("thresholds", [0.5, 0.55, 0.6, 0.65]),
    detector: cfg.get<string>("detector", ""),
    trigger: cfg.get<Trigger>("trigger", "idle"),
    minChars: cfg.get<number>("minChars", 80),
    debounceMs: cfg.get<number>("debounceMs", 500),
    enabled: cfg.get<boolean>("enabled", true),
    bestOf: cfg.get<number>("bestOf", 4),
    timeouts: cfg.get<Record<string, number>>("timeouts", {}),
  };
}

function setStatus(state: SidecarState, detail?: string): void {
  if (detail) {
    output.appendLine(`[sidecar] ${state}: ${detail}`);
  }
  switch (state) {
    case "starting":
      statusBar.text = "$(loading~spin) voice";
      statusBar.tooltip = "Voice Marker: loading the classifier…";
      break;
    case "ready": {
      lastStats = null;
      applyDetector();
      // First paint once the model is up — only when keystrokes may score
      // anyway; under `save`/`manual` nothing goes over the wire unasked.
      if (config().trigger === "idle") {
        scheduleScore(vscode.window.activeTextEditor, 0);
      }
      break;
    }
    case "failed":
      statusBar.text = "$(error) voice";
      statusBar.tooltip = `Voice Marker: sidecar failed — ${detail ?? "see output"}`;
      break;
    case "stopped":
      statusBar.text = "$(circle-slash) voice";
      statusBar.tooltip = "Voice Marker: sidecar stopped";
      break;
  }
  statusBar.show();
}

/** Resolve which armed detector marks, rebuild its decorations from its own
 * edges (an uncalibrated one gets none and marks nothing), and say so. */
function applyDetector(): void {
  const cfg = config();
  const armed = armedDetectors(sidecar?.meta);
  activeName = activeDetector(cfg.detector, sidecar?.meta);
  decorations?.dispose();
  decorations = null;
  const edges = activeName ? thresholdsFor(cfg.thresholds, activeName) : null;
  if (edges) {
    decorations = new Decorations(edges);
  }
  statusBar.text = `$(pulse) voice: ${activeName ?? "?"}`;
  refreshTooltip(edges === null ? "uncalibrated — no voiceMarker.thresholds entry" : null, armed);
}

function refreshTooltip(note: string | null, armed?: string[]): void {
  const lines = [`Voice Marker: ${activeName ?? "?"} ready`];
  const others = (armed ?? armedDetectors(sidecar?.meta)).filter((n) => n !== activeName);
  if (others.length > 0) {
    lines.push(`also armed: ${others.join(", ")} (Voice Marker: Switch Detector)`);
  }
  if (note) {
    lines.push(note);
  }
  if (lastStats) {
    const s = lastStats;
    lines.push(
      `${s.texts_scored ?? 0} texts / ${s.chars_sent ?? 0} chars sent this session` +
        (s.cache_hits !== undefined ? `, ${s.cache_hits} cache hits` : ""),
    );
  }
  lines.push(`trigger: ${config().trigger}`);
  statusBar.tooltip = lines.join("\n");
}

function scheduleScore(editor: vscode.TextEditor | undefined, delayMs?: number): void {
  if (!editor || !LANGS.has(editor.document.languageId)) {
    return;
  }
  if (debounceTimer) {
    clearTimeout(debounceTimer);
  }
  debounceTimer = setTimeout(() => void scoreEditor(editor), delayMs ?? config().debounceMs);
}

async function scoreEditor(editor: vscode.TextEditor): Promise<void> {
  const cfg = config();
  if (!cfg.enabled || !sidecar || !decorations) {
    return;
  }
  const document = editor.document;
  const generation = ++scoreGeneration;
  const version = document.version;
  // One snapshot, two hops: the segment offsets index THIS string, so slicing
  // it (rather than re-reading the document) stays exact even if the user has
  // typed on since. The version check below is what decides to paint.
  const text = document.getText();
  let segments: ProseSegment[];
  try {
    segments = await sidecar.segment(text, cfg.minChars);
  } catch (err) {
    output.appendLine(`[segment] ${err}`);
    return;
  }
  if (segments.length === 0) {
    if (generation === scoreGeneration && document.version === version) {
      decorations.clear(editor);
    }
    return;
  }
  let scores: (number | null)[];
  try {
    const resp = await sidecar.score(
      segments.map((s) => text.slice(s.start, s.end)),
      activeName ?? undefined,
    );
    scores = resp.scores;
    if (resp.stats) {
      lastStats = resp.stats;
      refreshTooltip(null);
    }
  } catch (err) {
    output.appendLine(`[score] ${err}`);
    return;
  }
  // Drop stale results: another score started, or the buffer moved on.
  if (generation !== scoreGeneration || document.version !== version) {
    return;
  }
  const scored = segments.map((s, i) => ({
    range: new vscode.Range(document.positionAt(s.start), document.positionAt(s.end)),
    score: scores[i] ?? null,
  }));
  decorations.apply(editor, scored, activeName ?? "voice-clf");
}

/** *Voice Marker: Switch Detector* — pick among what the sidecar armed; the
 * choice lands in workspace settings and repaints at once (an explicit ask,
 * so it scores regardless of `trigger`). */
async function switchDetectorCommand(): Promise<void> {
  const armed = armedDetectors(sidecar?.meta);
  if (armed.length === 0) {
    void vscode.window.showWarningMessage("Voice Marker: the sidecar is not ready yet.");
    return;
  }
  const cfg = config();
  const pick = await vscode.window.showQuickPick(
    armed.map((name) => ({
      label: name,
      description: name === activeName ? "marking now" : undefined,
      detail: thresholdsFor(cfg.thresholds, name)
        ? undefined
        : "uncalibrated: add a voiceMarker.thresholds entry (dan-style calibrate --detector …)",
    })),
    { placeHolder: "Which detector marks paragraphs?" },
  );
  if (!pick) {
    return;
  }
  await vscode.workspace
    .getConfiguration("voiceMarker")
    .update("detector", pick.label, vscode.ConfigurationTarget.Workspace);
  applyDetector();
  scheduleScore(vscode.window.activeTextEditor, 0);
}

/** The rewrite target and its heading context, from the sidecar's segments:
 * the selection if non-empty, else the prose paragraph at the cursor. Null
 * range when the cursor is in no prose paragraph (a fence, a figure div, …). */
function rewriteTarget(
  editor: vscode.TextEditor,
  segments: ProseSegment[],
): { range: vscode.Range; context: string | null } | null {
  const document = editor.document;
  const offset = document.offsetAt(
    editor.selection.isEmpty ? editor.selection.active : editor.selection.start,
  );
  // The context of the last segment starting at or before the anchor — which
  // for a paragraph target is its own, and for a multi-paragraph selection is
  // the heading the selection opens under.
  let context: string | null = null;
  for (const s of segments) {
    if (s.start <= offset) {
      context = s.context;
    }
  }
  if (!editor.selection.isEmpty) {
    return { range: new vscode.Range(editor.selection.start, editor.selection.end), context };
  }
  for (const s of segments) {
    if (s.start <= offset && offset <= s.end) {
      return {
        range: new vscode.Range(document.positionAt(s.start), document.positionAt(s.end)),
        context: s.context,
      };
    }
  }
  return null;
}

const firstLine = (text: string, max = 90): string => {
  const line = text.split("\n")[0];
  return line.length > max ? line.slice(0, max - 1) + "…" : line;
};

interface CandidateItem extends vscode.QuickPickItem {
  candidate: RewriteCandidate | null; // null = keep original
  presentedIndex: number | null;
}

async function rewriteCommand(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor || !LANGS.has(editor.document.languageId)) {
    return;
  }
  if (!sidecar) {
    void vscode.window.showWarningMessage("Voice Marker: sidecar is not configured (voiceMarker.command).");
    return;
  }
  const document = editor.document;
  const text = document.getText();
  let segments: ProseSegment[];
  try {
    segments = await sidecar.segment(text, 0);
  } catch (err) {
    void vscode.window.showErrorMessage(`Voice Marker: segmentation failed — ${(err as Error).message}`);
    return;
  }
  const target = rewriteTarget(editor, segments);
  if (!target) {
    void vscode.window.showInformationMessage("Voice Marker: put the cursor in a prose paragraph (or select prose).");
    return;
  }
  const { range, context } = target;
  const original = document.getText(range);
  const bestOf = config().bestOf;

  let resp;
  try {
    resp = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `Voice Marker: sampling ${bestOf} rewrite(s)…`,
      },
      () => sidecar!.rewrite(original, context, bestOf),
    );
  } catch (err) {
    void vscode.window.showErrorMessage(`Voice Marker: rewrite failed — ${(err as Error).message}`);
    return;
  }
  const fmt = (s: number | null) => (s === null ? "?" : s.toFixed(2));
  // Identity candidates (the styler declined to change the text) are the
  // "Keep original" row by another name — don't present them as rewrites.
  const distinct = resp.candidates.filter((c) => !c.identity);
  if (distinct.length === 0) {
    void vscode.window.showInformationMessage(
      `Voice Marker: the styler proposes no change ` +
        `(input P(slop) ${fmt(resp.input_score)}, ${resp.n_sampled ?? bestOf} sample(s)).`,
    );
    return;
  }

  // Present ascending by P(slop) (unscored last); the presented order is what
  // gets logged — presentation bias is recorded, not hidden.
  const presented = [...distinct].sort(
    (a, b) => (a.score ?? Infinity) - (b.score ?? Infinity),
  );
  const items: CandidateItem[] = presented.map((c, i) => ({
    label:
      `$(sparkle) P(slop) ${fmt(c.score)}` +
      (c.anchors_lost.length ? `  $(warning) loses ${c.anchors_lost.join(", ")}` : ""),
    description: firstLine(c.text),
    candidate: c,
    presentedIndex: i,
  }));
  items.push({
    label: `$(discard) Keep original (P(slop) ${fmt(resp.input_score)})`,
    description: firstLine(original),
    candidate: null,
    presentedIndex: null,
  });

  // A live diff (original ↔ highlighted candidate) opens beside the picker —
  // the QuickPick rows are too cramped to judge a whole-paragraph rewrite.
  const origUri = vscode.Uri.from({ scheme: "voice-marker", path: "/original.md" });
  const candUri = vscode.Uri.from({ scheme: "voice-marker", path: "/candidate.md" });
  preview.set(origUri.path, original);
  let diffOpen = false;
  const showPreview = (item: CandidateItem | undefined): void => {
    if (!item) {
      return;
    }
    preview.set(candUri.path, item.candidate ? item.candidate.text : original);
    if (!diffOpen) {
      diffOpen = true;
      void vscode.commands.executeCommand(
        "vscode.diff", origUri, candUri, "Voice Marker: rewrite preview",
        { preview: true, preserveFocus: true, viewColumn: vscode.ViewColumn.Beside },
      );
    }
  };

  const qp = vscode.window.createQuickPick<CandidateItem>();
  qp.title = `Rewrite in Dan's voice — input P(slop) ${fmt(resp.input_score)}`;
  qp.placeholder = "Choose a candidate (Esc aborts without logging a preference)";
  qp.ignoreFocusOut = true;
  qp.items = items;
  qp.onDidChangeActive((active) => showPreview(active[0]));
  const pick = await new Promise<CandidateItem | undefined>((resolve) => {
    qp.onDidAccept(() => resolve(qp.selectedItems[0]));
    qp.onDidHide(() => resolve(undefined));
    qp.show();
  });
  qp.hide();
  qp.dispose();
  // Close the preview diff tab regardless of outcome.
  for (const group of vscode.window.tabGroups.all) {
    for (const tab of group.tabs) {
      if (
        tab.input instanceof vscode.TabInputTextDiff &&
        tab.input.modified.scheme === "voice-marker"
      ) {
        void vscode.window.tabGroups.close(tab);
      }
    }
  }
  if (!pick) {
    return; // Esc: an abort is not a judgement — log nothing.
  }

  // Discourse context for the record, read BEFORE the edit mutates offsets:
  // the prose paragraphs flanking the target. Future longer-scale /
  // coherence-aware training (v2) needs this, and it cannot be backfilled —
  // the working copy is uncommitted.
  const startOffset = document.offsetAt(range.start);
  const endOffset = document.offsetAt(range.end);
  const before = [...segments].reverse().find((s) => s.end <= startOffset);
  const after = segments.find((s) => s.start >= endOffset);

  if (pick.candidate) {
    const ok = await editor.edit((edit) => edit.replace(range, pick.candidate!.text));
    if (!ok) {
      void vscode.window.showErrorMessage("Voice Marker: buffer edit failed; nothing logged.");
      return;
    }
  }
  // Fire-and-forget: preference capture must never block the editor.
  sidecar
    .prefer({
      chunk: original,
      context,
      candidates: presented.map((c) => c.text),
      scores: presented.map((c) => c.score),
      chosen: pick.presentedIndex,
      kept_input: pick.candidate === null,
      doc: vscode.workspace.asRelativePath(document.uri),
      offsets: [startOffset, endOffset],
      para_before: before ? text.slice(before.start, before.end) : null,
      para_after: after ? text.slice(after.start, after.end) : null,
    })
    .catch((err) => output.appendLine(`[prefer] not recorded: ${err}`));
  scheduleScore(editor, 0);
}

/** Toggle the `sem` markers around each selection; an empty selection takes
 * its paragraph (stylebot D93). Pure text work in ./sem.ts. */
/** Preen the paragraph under the cursor (or the selection) in place: the
 * sidecar's `preen` op is the blog's copy-edit pass at chunk scale, so a typo
 * is fixed in the file BEFORE `log save` freezes it into a pair (stylebot D107).
 * Undo reverts it like any edit. */
async function preenCommand(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor || !LANGS.has(editor.document.languageId)) {
    return;
  }
  if (!sidecar) {
    void vscode.window.showWarningMessage("Voice Marker: sidecar is not configured (voiceMarker.command).");
    return;
  }
  const document = editor.document;
  let segments: ProseSegment[];
  try {
    segments = await sidecar.segment(document.getText(), 0);
  } catch (err) {
    void vscode.window.showErrorMessage(`Voice Marker: segmentation failed — ${(err as Error).message}`);
    return;
  }
  const target = rewriteTarget(editor, segments);
  if (!target) {
    void vscode.window.showInformationMessage("Voice Marker: put the cursor in a prose paragraph (or select prose).");
    return;
  }
  const original = document.getText(target.range);
  let finished: string;
  try {
    finished = await vscode.window.withProgress(
      { location: vscode.ProgressLocation.Notification, title: "Voice Marker: preening the paragraph…" },
      () => sidecar!.preen(original),
    );
  } catch (err) {
    void vscode.window.showErrorMessage(`Voice Marker: preen failed — ${(err as Error).message}`);
    return;
  }
  if (finished === original) {
    vscode.window.setStatusBarMessage("Voice Marker: preen found nothing to fix", 4000);
    return;
  }
  const ok = await editor.edit((edit) => edit.replace(target.range, finished));
  if (!ok) {
    void vscode.window.showErrorMessage("Voice Marker: buffer edit failed; the paragraph is unchanged.");
    return;
  }
  vscode.window.setStatusBarMessage("Voice Marker: paragraph preened — undo reverts it", 4000);
}

async function markSemanticCommand(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return;
  }
  const doc = editor.document;
  await editor.edit((edit) => {
    for (const sel of editor.selections) {
      let range: vscode.Range = sel;
      if (sel.isEmpty) {
        const [start, end] = paragraphAround(doc.getText(), doc.offsetAt(sel.active));
        range = new vscode.Range(doc.positionAt(start), doc.positionAt(end));
      }
      edit.replace(range, toggleSemWrap(doc.getText(range)));
    }
  });
}

export function activate(context: vscode.ExtensionContext): void {
  output = vscode.window.createOutputChannel("Voice Marker");
  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 90);
  context.subscriptions.push(output, statusBar);

  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider("voice-marker", preview),
  );

  const cfg = config();
  if (cfg.command.length === 0) {
    // Unconfigured workspace: stay dormant (no process, no marks, no status).
    output.appendLine("voiceMarker.command is unset; staying dormant");
    return;
  }

  const spawnSidecar = (): void => {
    const c = config();
    sidecar?.dispose();
    sidecar = new Sidecar(c.command, c.cwd, {
      onState: setStatus,
      onStderr: (chunk) => output.append(chunk),
    }, c.timeouts);
    sidecar.start();
  };
  spawnSidecar();
  context.subscriptions.push({
    dispose: () => {
      sidecar?.dispose();
      decorations?.dispose();
    },
  });

  context.subscriptions.push(
    vscode.workspace.onDidChangeTextDocument((e) => {
      const editor = vscode.window.activeTextEditor;
      if (editor && e.document === editor.document && config().trigger === "idle") {
        scheduleScore(editor);
      }
    }),
    vscode.workspace.onDidSaveTextDocument((doc) => {
      const editor = vscode.window.activeTextEditor;
      if (editor && doc === editor.document && config().trigger === "save") {
        scheduleScore(editor, 0);
      }
    }),
    vscode.window.onDidChangeActiveTextEditor((editor) => {
      if (config().trigger === "idle") {
        scheduleScore(editor, 0);
      }
    }),
    vscode.commands.registerCommand("voiceMarker.rescore", () =>
      scheduleScore(vscode.window.activeTextEditor, 0),
    ),
    vscode.commands.registerCommand("voiceMarker.switchDetector", () => void switchDetectorCommand()),
    vscode.commands.registerCommand("voiceMarker.toggle", async () => {
      const now = !config().enabled;
      await vscode.workspace
        .getConfiguration("voiceMarker")
        .update("enabled", now, vscode.ConfigurationTarget.Workspace);
      const editor = vscode.window.activeTextEditor;
      if (!now && editor && decorations) {
        decorations.clear(editor);
      } else {
        scheduleScore(editor, 0);
      }
    }),
    vscode.commands.registerCommand("voiceMarker.restartSidecar", () => sidecar?.restart()),
    vscode.commands.registerCommand("voiceMarker.rewrite", () => void rewriteCommand()),
    vscode.commands.registerCommand("voiceMarker.markSemantic", () => void markSemanticCommand()),
    vscode.commands.registerCommand("voiceMarker.preen", () => void preenCommand()),
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("voiceMarker.command") || e.affectsConfiguration("voiceMarker.cwd")) {
        output.appendLine("voiceMarker.command changed; respawning the sidecar");
        spawnSidecar();
        return;
      }
      if (
        e.affectsConfiguration("voiceMarker.thresholds") ||
        e.affectsConfiguration("voiceMarker.detector")
      ) {
        // Edges or instrument changed: repaint. Unchanged paragraphs answer
        // from the sidecar's cache, so this spends nothing on a hosted one.
        applyDetector();
        scheduleScore(vscode.window.activeTextEditor, 0);
      } else if (e.affectsConfiguration("voiceMarker.trigger")) {
        refreshTooltip(null);
      }
    }),
  );
}

export function deactivate(): void {
  sidecar?.dispose();
}
