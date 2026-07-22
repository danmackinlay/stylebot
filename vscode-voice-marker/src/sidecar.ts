/**
 * Client for the `ai-style serve` NDJSON sidecar (stylebot.serve).
 *
 * Owns the child-process lifecycle: spawn, id-matched request/response over
 * stdin/stdout, restart with capped backoff on crash, kill on dispose. No
 * `vscode` import — the extension wires status/stderr through callbacks.
 */

import { ChildProcess, spawn } from "child_process";

export type SidecarState = "stopped" | "starting" | "ready" | "failed";

/** One styler candidate from the `rewrite` op (stylebot.infer.rewrite_candidates). */
export interface RewriteCandidate {
  text: string;
  /** Detector P(slop); lower = more Dan. Null when unscored. */
  score: number | null;
  /** Links/URLs/citations the input had that this candidate lost (flag, not veto). */
  anchors_lost: string[];
  /** True when the candidate is byte-equal to the input — the styler
   * declining to change already-Dan prose (a do-no-harm answer). */
  identity?: boolean;
}

export interface RewriteResponse {
  input_score: number | null;
  /** Draws taken; the server dedupes identical samples, so
   * candidates.length <= n_sampled. */
  n_sampled?: number;
  candidates: RewriteCandidate[];
}

/** The `prefer` op payload — mirrors livingthing's preferences.jsonl schema. */
export interface PreferenceDecision {
  chunk: string;
  context: string | null;
  /** Candidate texts IN THE ORDER PRESENTED (presentation bias is logged, not hidden). */
  candidates: string[];
  scores: (number | null)[];
  /** Index into `candidates`; null when the input was kept. */
  chosen: number | null;
  kept_input: boolean;
  doc: string;
}

export interface SidecarEvents {
  onState: (state: SidecarState, detail?: string) => void;
  onStderr: (chunk: string) => void;
}

interface Pending {
  resolve: (value: any) => void;
  reject: (err: Error) => void;
}

const MAX_RESTARTS = 3;

export class Sidecar {
  private child: ChildProcess | null = null;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private buffer = "";
  private restarts = 0;
  private disposed = false;
  /** The artifact meta.json from the ready handshake (model name, metrics). */
  meta: Record<string, any> | null = null;

  constructor(
    private command: string[],
    private cwd: string | undefined,
    private events: SidecarEvents,
  ) {}

  start(): void {
    if (this.disposed || this.child) {
      return;
    }
    const [exe, ...args] = this.command;
    this.events.onState("starting");
    const child = spawn(exe, args, {
      cwd: this.cwd,
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });
    this.child = child;

    child.stdout!.setEncoding("utf8");
    child.stdout!.on("data", (chunk: string) => this.onData(chunk));
    child.stderr!.setEncoding("utf8");
    child.stderr!.on("data", (chunk: string) => this.events.onStderr(chunk));
    child.on("error", (err) => this.onExit(`spawn failed: ${err.message}`));
    child.on("exit", (code, signal) =>
      this.onExit(`exited (code=${code}, signal=${signal})`),
    );

    // The info handshake: resolves only once the model has loaded (~5 s),
    // which is exactly what "ready" should mean.
    this.request<{ meta: Record<string, any> }>({ op: "info" })
      .then((resp) => {
        this.meta = resp.meta;
        this.restarts = 0; // a full handshake resets the crash budget
        this.events.onState("ready");
      })
      .catch(() => {
        /* exit/error path already reported state */
      });
  }

  /** Score texts; resolves to one P(slop) per text, in order. */
  async score(texts: string[]): Promise<number[]> {
    const resp = await this.request<{ scores: number[] }>({ op: "score", texts });
    return resp.scores;
  }

  /** Best-of candidates for one prose chunk (server needs a styler armed). */
  async rewrite(text: string, context: string | null, bestOf: number): Promise<RewriteResponse> {
    return this.request<RewriteResponse>({ op: "rewrite", text, context, best_of: bestOf });
  }

  /** Record a human choice among candidates (server needs a preferences sink). */
  async prefer(decision: PreferenceDecision): Promise<void> {
    await this.request<{ ok: boolean }>({ op: "prefer", decision });
  }

  private request<T>(body: Record<string, any>): Promise<T> {
    const child = this.child;
    if (!child || !child.stdin || !child.stdin.writable) {
      return Promise.reject(new Error("sidecar not running"));
    }
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      child.stdin!.write(JSON.stringify({ id, ...body }) + "\n");
    });
  }

  private onData(chunk: string): void {
    this.buffer += chunk;
    let nl: number;
    while ((nl = this.buffer.indexOf("\n")) >= 0) {
      const line = this.buffer.slice(0, nl).trim();
      this.buffer = this.buffer.slice(nl + 1);
      if (!line) {
        continue;
      }
      let resp: any;
      try {
        resp = JSON.parse(line);
      } catch {
        this.events.onStderr(`unparseable sidecar line: ${line}\n`);
        continue;
      }
      const entry = this.pending.get(resp.id);
      if (!entry) {
        continue;
      }
      this.pending.delete(resp.id);
      if (resp.error) {
        entry.reject(new Error(resp.error));
      } else {
        entry.resolve(resp);
      }
    }
  }

  private onExit(detail: string): void {
    this.child = null;
    this.buffer = "";
    for (const entry of this.pending.values()) {
      entry.reject(new Error(`sidecar ${detail}`));
    }
    this.pending.clear();
    if (this.disposed) {
      return;
    }
    if (this.restarts >= MAX_RESTARTS) {
      this.events.onState("failed", `${detail}; gave up after ${MAX_RESTARTS} restarts`);
      return;
    }
    this.restarts += 1;
    const delay = 1000 * 2 ** (this.restarts - 1); // 1s, 2s, 4s
    this.events.onState("starting", `${detail}; restart ${this.restarts}/${MAX_RESTARTS} in ${delay}ms`);
    setTimeout(() => this.start(), delay);
  }

  /** Manual restart (command changed, artifact retrained, …). */
  restart(): void {
    this.restarts = 0;
    if (this.child) {
      const child = this.child;
      this.child = null; // suppress the auto-restart path's stale handle
      child.removeAllListeners();
      child.kill();
    }
    this.start();
  }

  dispose(): void {
    this.disposed = true;
    if (this.child) {
      this.child.kill();
      this.child = null;
    }
    this.events.onState("stopped");
  }
}
