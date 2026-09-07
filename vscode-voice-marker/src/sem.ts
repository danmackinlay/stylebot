/**
 * Dan's content annotation (stylebot D93): wrap a span of an edit that changes
 * what the text says in `<!-- sem -->…<!-- /sem -->`, so the view cuts it from
 * the pair instead of teaching it as style. Pure text logic, testable without
 * VS Code; the command in extension.ts applies it to the selection.
 */

export const SEM_OPEN = "<!-- sem -->";
export const SEM_CLOSE = "<!-- /sem -->";

/** Wrap `text` in the markers, or unwrap it when it is already wrapped (a toggle). */
export function toggleSemWrap(text: string): string {
  const trimmed = text.trim();
  if (trimmed.startsWith(SEM_OPEN) && trimmed.endsWith(SEM_CLOSE)) {
    const inner = trimmed.slice(SEM_OPEN.length, trimmed.length - SEM_CLOSE.length);
    const lead = text.slice(0, text.length - text.trimStart().length);
    const tail = text.slice(text.trimEnd().length);
    return lead + inner.trim() + tail;
  }
  if (!trimmed) {
    return text;
  }
  const lead = text.slice(0, text.length - text.trimStart().length);
  const tail = text.slice(text.trimEnd().length);
  return `${lead}${SEM_OPEN}${trimmed}${SEM_CLOSE}${tail}`;
}

/** The [start, end) of the blank-line paragraph around `offset` in `text`. */
export function paragraphAround(text: string, offset: number): [number, number] {
  const before = text.lastIndexOf("\n\n", Math.max(0, offset - 1));
  const start = before === -1 ? 0 : before + 2;
  const after = text.indexOf("\n\n", offset);
  const end = after === -1 ? text.length : after;
  return [start, end];
}
