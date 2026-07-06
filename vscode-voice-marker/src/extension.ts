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
import { Sidecar, SidecarState } from "./sidecar";

const LANGS = new Set(["markdown", "quarto"]);

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

export function activate(context: vscode.ExtensionContext): void {
  output = vscode.window.createOutputChannel("Voice Marker");
  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 90);
  context.subscriptions.push(output, statusBar);

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
  });
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
