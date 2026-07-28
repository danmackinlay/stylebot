/**
 * Voice Marker — mark paragraphs dan-or-bot with P(slop) from `ai-style serve`.
 *
 * Flow: activate on markdown/quarto → spawn the sidecar from
 * `voiceMarker.command` (dormant if unset) → on debounced edits, segment the
 * document (parity with stylebot.lib), score the prose paragraphs, and paint
 * graded decorations. See stylebot/_plans/vscode-marker.md.
 */

import * as vscode from "vscode";
import { Decorations } from "./decorations";
import { proseParagraphs } from "./segment";
import { RewriteCandidate, Sidecar, SidecarState } from "./sidecar";

const LANGS = new Set(["markdown", "quarto"]);

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

function config() {
  const cfg = vscode.workspace.getConfiguration("voiceMarker");
  return {
    command: cfg.get<string[]>("command", []),
    cwd: cfg.get<string>("cwd", "") || vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
    thresholds: cfg.get<number[]>("thresholds", [0.5, 0.55, 0.6, 0.65]),
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
      const name = sidecar?.meta?.name ?? "voice-clf";
      statusBar.text = "$(pulse) voice";
      statusBar.tooltip = `Voice Marker: ${name} ready (${sidecar?.meta?.embed_model ?? "?"})`;
      // First paint once the model is up.
      scheduleScore(vscode.window.activeTextEditor, 0);
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
  const paragraphs = proseParagraphs(document.getText(), cfg.minChars);
  if (paragraphs.length === 0) {
    decorations.clear(editor);
    return;
  }
  let scores: number[];
  try {
    scores = await sidecar.score(paragraphs.map((p) => p.text));
  } catch (err) {
    output.appendLine(`[score] ${err}`);
    return;
  }
  // Drop stale results: another score started, or the buffer moved on.
  if (generation !== scoreGeneration || document.version !== version) {
    return;
  }
  const scored = paragraphs.map((p, i) => ({
    range: new vscode.Range(document.positionAt(p.start), document.positionAt(p.end)),
    score: scores[i],
  }));
  decorations.apply(editor, scored, sidecar.meta?.name ?? "voice-clf");
}

/** The rewrite target: the selection if non-empty, else the prose paragraph
 * at the cursor (proseParagraphs offsets). Null when neither applies. */
function rewriteTarget(editor: vscode.TextEditor): vscode.Range | null {
  if (!editor.selection.isEmpty) {
    return new vscode.Range(editor.selection.start, editor.selection.end);
  }
  const document = editor.document;
  const offset = document.offsetAt(editor.selection.active);
  for (const p of proseParagraphs(document.getText(), 0)) {
    if (p.start <= offset && offset <= p.end) {
      return new vscode.Range(document.positionAt(p.start), document.positionAt(p.end));
    }
  }
  return null;
}

/** Nearest markdown heading line above `line` — the training-time context. */
function headingAbove(document: vscode.TextDocument, line: number): string | null {
  for (let i = line - 1; i >= 0; i--) {
    const text = document.lineAt(i).text;
    if (/^[ \t]{0,3}#{1,6}[ \t]/.test(text)) {
      return text.trim();
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
  const range = rewriteTarget(editor);
  if (!range) {
    void vscode.window.showInformationMessage("Voice Marker: put the cursor in a prose paragraph (or select prose).");
    return;
  }
  const document = editor.document;
  const original = document.getText(range);
  const context = headingAbove(document, range.start.line);
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
  const paras = proseParagraphs(document.getText(), 0);
  const before = [...paras].reverse().find((p) => p.end <= startOffset);
  const after = paras.find((p) => p.start >= endOffset);

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
      para_before: before ? before.text : null,
      para_after: after ? after.text : null,
    })
    .catch((err) => output.appendLine(`[prefer] not recorded: ${err}`));
  scheduleScore(editor, 0);
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

  decorations = new Decorations(cfg.thresholds);
  sidecar = new Sidecar(cfg.command, cfg.cwd, {
    onState: setStatus,
    onStderr: (chunk) => output.append(chunk),
  }, cfg.timeouts);
  context.subscriptions.push({
    dispose: () => {
      sidecar?.dispose();
      decorations?.dispose();
    },
  });
  sidecar.start();

  context.subscriptions.push(
    vscode.workspace.onDidChangeTextDocument((e) => {
      const editor = vscode.window.activeTextEditor;
      if (editor && e.document === editor.document) {
        scheduleScore(editor);
      }
    }),
    vscode.window.onDidChangeActiveTextEditor((editor) => scheduleScore(editor, 0)),
    vscode.commands.registerCommand("voiceMarker.rescore", () =>
      scheduleScore(vscode.window.activeTextEditor, 0),
    ),
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
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("voiceMarker.thresholds")) {
        decorations?.dispose();
        decorations = new Decorations(config().thresholds);
        scheduleScore(vscode.window.activeTextEditor, 0);
      }
    }),
  );
}

export function deactivate(): void {
  sidecar?.dispose();
}
