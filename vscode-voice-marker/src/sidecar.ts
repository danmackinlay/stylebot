/**
 * Client for the `ai-style serve` NDJSON sidecar (stylebot.serve).
 *
 * Owns the child-process lifecycle: spawn, id-matched request/response over
 * stdin/stdout, restart with capped backoff on crash, kill on dispose. No
 * `vscode` import — the extension wires status/stderr through callbacks.
 */

import { ChildProcess, spawn } from "child_process";

export type SidecarState = "stopped" | "starting" | "ready" | "failed";

/** One scoreable prose paragraph from the `segment` op (stylebot.serve.prose_paragraphs).
 *
 * The extension owns no segmenter: which spans of a document are prose, and
 * what heading each sits under, is a question only the sidecar can answer,
 * because the answer depends on the `div_is_prose` policy it was started with
 * (a `.callout-note` paragraph is prose; a `#fig-` div's caption is not). */
export interface ProseSegment {
  /** UTF-16 code-unit offsets into the document that was sent — the same index
   * space as `TextDocument.positionAt`, so no conversion is needed here. */
  start: number;
  end: number;
  kind: string;
  /** Nearest heading line above, or null. Provenance for the `prefer` record;
   * whether the styler actually SEES it is the serving diet's call, made
   * server-side in `stylebot.infer.rewrite_candidates`. */
  context: string | null;
}

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

/** The `score` op reply (stylebot.serve._score_op). */
export interface ScoreResponse {
  scores: (number | null)[];
  /** Which armed detector answered; absent from a pre-D106 sidecar. */
  detector?: string;
  /** Session spend counters (requests, texts_scored, chars_sent, cache_hits) —
   * present only for a detector that keeps them (a hosted one). */
  stats?: Record<string, number>;
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
  /** Character offsets of `chunk` in the document at decision time. */
  offsets?: [number, number];
  /** Adjacent prose paragraphs — discourse context for future longer-scale /
   * coherence-aware training (v2); unrecoverable later since the working copy
   * is uncommitted. Null at a document edge. */
  para_before?: string | null;
  para_after?: string | null;
}

export interface SidecarEvents {
  onState: (state: SidecarState, detail?: string) => void;
  onStderr: (chunk: string) => void;
}

interface Pending {
  resolve: (value: any) => void;
  reject: (err: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

const MAX_RESTARTS = 3;

/**
 * Per-op request budgets (ms). Without these a wedged sidecar hangs the UI
 * forever: `request()` settles only on a matching response line or on process
 * exit, and a Python process blocked on a dead socket does neither. Observed
 * 2026-07-28 — a days-old sidecar on the tinker backend, parked on a
 * CLOSE_WAIT socket to CloudFront left over from a previous network, 0% CPU,
 * no timeout anywhere in the chain.
 *
 * `rewrite` is generous because a legitimate best-of-4 sample over the
 * network (or a cold local MLX load) genuinely takes tens of seconds; `score`
 * is local detector inference and should be near-instant; `info` covers the
 * handshake's model load.
 */
const TIMEOUT_MS: Record<string, number> = {
  info: 60_000,
  score: 30_000,
  segment: 30_000,
  rewrite: 180_000,
  prefer: 15_000,
};
const DEFAULT_TIMEOUT_MS = 60_000;

/** Consecutive timeouts before the process is presumed wedged and restarted. */
const WEDGED_AFTER_TIMEOUTS = 2;

export class Sidecar {
  private child: ChildProcess | null = null;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private buffer = "";
  private restarts = 0;
  private disposed = false;
  /** Timeouts in a row with no intervening response — a wedged-process signal. */
  private consecutiveTimeouts = 0;
  /** The artifact meta.json from the ready handshake (model name, metrics). */
  meta: Record<string, any> | null = null;

  constructor(
    private command: string[],
    private cwd: string | undefined,
    private events: SidecarEvents,
    /** Per-op timeout overrides (ms), e.g. from voiceMarker.timeouts. */
    private timeouts?: Record<string, number>,
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

  /** The document's scoreable prose spans, under the sidecar's own policy. */
  async segment(text: string, minChars: number): Promise<ProseSegment[]> {
    const resp = await this.request<{ segments: ProseSegment[] }>({
      op: "segment",
      text,
      min_chars: minChars,
    });
    return resp.segments;
  }

  /** Score texts with the named detector (the sidecar's default when omitted;
   * stylebot D106): one P(slop) per text, in order, null where it abstained;
   * `stats` is the detector's running spend when it keeps one. */
  async score(texts: string[], detector?: string): Promise<ScoreResponse> {
    const body: Record<string, any> = { op: "score", texts };
    if (detector) {
      body.detector = detector;
    }
    return this.request<ScoreResponse>(body);
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
    const op = String(body.op ?? "");
    const budget = this.timeoutFor(op);
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => this.onTimeout(id, op, budget), budget);
      // Never hold the extension host open on our behalf.
      (timer as any).unref?.();
      this.pending.set(id, { resolve, reject, timer });
      child.stdin!.write(JSON.stringify({ id, ...body }) + "\n");
    });
  }

  private timeoutFor(op: string): number {
    const override = this.timeouts?.[op];
    return override ?? TIMEOUT_MS[op] ?? DEFAULT_TIMEOUT_MS;
  }

  /** A request that outran its budget: fail it loudly, and restart the child
   * if it looks wedged rather than merely slow. */
  private onTimeout(id: number, op: string, budget: number): void {
    const entry = this.pending.get(id);
    if (!entry) {
      return;
    }
    this.pending.delete(id);
    this.consecutiveTimeouts += 1;
    const secs = Math.round(budget / 1000);
    const wedged = this.consecutiveTimeouts >= WEDGED_AFTER_TIMEOUTS;
    entry.reject(
      new Error(
        `'${op}' timed out after ${secs}s — the sidecar is not responding. ` +
          `A styler backend that dials out (tinker) can park on a dead socket; ` +
          `check the Voice Marker output channel, or switch to the local ` +
          `backend (DAN_STYLE_BACKEND=local with the MLX server up).` +
          (wedged ? " Restarting the sidecar." : ""),
      ),
    );
    if (wedged) {
      this.events.onState("starting", `unresponsive after ${this.consecutiveTimeouts} timeouts; restarting`);
      this.consecutiveTimeouts = 0;
      this.restart();
    }
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
      clearTimeout(entry.timer);
      this.consecutiveTimeouts = 0; // the child is demonstrably alive
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
    this.failPending(`sidecar ${detail}`);
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

  /** Manual restart (command changed, artifact retrained, wedged child, …). */
  restart(): void {
    this.restarts = 0;
    this.consecutiveTimeouts = 0;
    if (this.child) {
      const child = this.child;
      this.child = null; // suppress the auto-restart path's stale handle
      child.removeAllListeners();
      child.kill();
    }
    // removeAllListeners() suppressed the exit path that would have settled
    // these, so reject here or they hang for the life of the window.
    this.failPending("sidecar restarted");
    this.buffer = "";
    this.start();
  }

  dispose(): void {
    this.disposed = true;
    if (this.child) {
      this.child.kill();
      this.child = null;
    }
    this.failPending("sidecar disposed");
    this.events.onState("stopped");
  }

  private failPending(reason: string): void {
    for (const entry of this.pending.values()) {
      clearTimeout(entry.timer);
      entry.reject(new Error(reason));
    }
    this.pending.clear();
  }
}
