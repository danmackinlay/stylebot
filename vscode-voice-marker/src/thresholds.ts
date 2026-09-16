/**
 * Which instrument marks, and with which bucket edges — pure, no `vscode`.
 *
 * The sidecar may arm several detectors (stylebot D106): `info` meta lists
 * them in `detectors`, with the default in `detector`. `voiceMarker.thresholds`
 * is either one array (legacy: the same edges for every detector) or an object
 * keyed by detector name; a detector with no entry is *uncalibrated* and marks
 * nothing, because edges cut for one score distribution are meaningless on
 * another (`dan-style calibrate --detector …` cuts them).
 */

export type ThresholdsSetting = number[] | Record<string, number[]>;

export interface SidecarMeta {
  name?: string;
  detector?: string;
  detectors?: string[];
}

/** The detectors the sidecar armed, in its order; a pre-D106 sidecar has one. */
export function armedDetectors(meta: SidecarMeta | null | undefined): string[] {
  if (meta?.detectors && meta.detectors.length > 0) {
    return [...meta.detectors];
  }
  return meta?.name ? [meta.name] : [];
}

/** The instrument to mark with: the configured one if the sidecar armed it,
 * else the sidecar's default. Null when nothing is armed yet. */
export function activeDetector(
  configured: string | undefined,
  meta: SidecarMeta | null | undefined,
): string | null {
  const armed = armedDetectors(meta);
  if (armed.length === 0) {
    return null;
  }
  if (configured && armed.includes(configured)) {
    return configured;
  }
  return meta?.detector && armed.includes(meta.detector) ? meta.detector : armed[0];
}

/** The ascending edges for `detector`, or null when it has none (uncalibrated). */
export function thresholdsFor(
  setting: ThresholdsSetting | null | undefined,
  detector: string,
): number[] | null {
  if (!setting) {
    return null;
  }
  const edges = Array.isArray(setting) ? setting : setting[detector];
  if (!Array.isArray(edges) || edges.length === 0 || !edges.every((e) => typeof e === "number")) {
    return null;
  }
  return [...edges].sort((a, b) => a - b);
}
