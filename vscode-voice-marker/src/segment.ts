/**
 * Markdown prose segmentation — a parity port of stylebot's splitter.
 *
 * Mirrors `stylebot.lib.segment_for_edit` (protected blocks: fenced code,
 * $$math$$, nesting-aware ::: divs, blockquotes) and
 * `stylebot.lib.split_paragraphs` (blank-line paragraph boundaries), plus two
 * marker-specific drops the training pipeline handles upstream: YAML
 * frontmatter and heading lines. The classifier trained on bare prose bodies,
 * so scoring anything else would be off-distribution.
 *
 * Keep behaviour in sync with stylebot/src/stylebot/markdown.py — the shared
 * fixture (`stylebot/tests/fixtures/segmentation.qmd` +
 * `expected_segments.json`) is asserted by BOTH pytest and this package's
 * tests. No `vscode` import here, so tests run under plain `node --test`.
 */

export interface Paragraph {
  /** Paragraph text, exactly as sliced from the document. */
  text: string;
  /** Character offset of the first character in the original document. */
  start: number;
  /** Character offset one past the last character. */
  end: number;
}

// Protected-span regexes — transliterations of stylebot.markdown's patterns.
// Python re.MULTILINE|re.DOTALL becomes /gm with [\s\S] standing in for
// DOTALL-dot; Python's (?:\n|\Z) trailing-newline consumption becomes \n?.
// Opener and closer alternate independently — deliberately NOT a backreference
// (see stylebot.markdown.CODE_FENCE_RE for the measurement).
const CODE_FENCE_RE = /^[ \t]*(?:```|~~~).*?$[\s\S]*?^[ \t]*(?:```|~~~)[ \t]*$\n?/gm;
const MATH_BLOCK_RE = /\$\$[\s\S]*?\$\$[ \t]*(?:\{[^}\n]*\})?/g;
const BLOCKQUOTE_RE = /^(?:[ \t]{0,3}>[^\n]*\n?)+/gm;

type Span = [number, number];

/** Nesting-aware ::: div/callout spans — port of stylebot.markdown.find_div_blocks. */
function findDivBlocks(content: string): Span[] {
  const spans: Span[] = [];
  const lines = content.split(/(?<=\n)/); // keepends=True equivalent
  let i = 0;
  let currentPos = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.trim().startsWith(":::")) {
      const startPos = currentPos;
      let depth = 1;
      let j = i + 1;
      while (j < lines.length && depth > 0) {
        const inner = lines[j].trim();
        if (inner.startsWith(":::")) {
          if (inner === ":::") {
            depth -= 1;
          } else {
            depth += 1;
          }
        }
        j += 1;
      }
      if (depth === 0) {
        let endPos = startPos;
        for (let k = i; k < j; k++) {
          endPos += lines[k].length;
        }
        spans.push([startPos, endPos]);
        currentPos = endPos;
        i = j;
        continue;
      }
    }
    currentPos += line.length;
    i += 1;
  }
  return spans;
}

/** YAML frontmatter span at the very start of the document (qmd convention). */
function frontmatterSpan(content: string): Span | null {
  if (!content.startsWith("---\n") && content !== "---") {
    return null;
  }
  const close = /^(?:---|\.\.\.)[ \t]*$\n?/gm;
  close.lastIndex = 4; // past the opening "---\n"
  const m = close.exec(content);
  return m ? [0, m.index + m[0].length] : null;
}

function regexSpans(content: string, re: RegExp): Span[] {
  const spans: Span[] = [];
  re.lastIndex = 0;
  for (const m of content.matchAll(re)) {
    spans.push([m.index!, m.index! + m[0].length]);
  }
  return spans;
}

/**
 * Port of stylebot.markdown._is_block_anchored: does the span occupy whole
 * lines (at most 3 spaces in, nothing but whitespace out)?
 *
 * Display math is protected only when it is block-anchored. Excising a
 * mid-sentence `$$k$$` joins the remnants — `high-$$d$$ behaviour` becomes
 * `high- behaviour` — which corrupted real training targets and would split a
 * scoreable paragraph in two here.
 */
function isBlockAnchored(content: string, start: number, end: number): boolean {
  const lineStart = content.lastIndexOf("\n", start - 1) + 1;
  const indent = content.slice(lineStart, start);
  if (indent.length > 3 || /[^ ]/.test(indent)) {
    return false;
  }
  const nl = content.indexOf("\n", end);
  const tail = nl === -1 ? content.slice(end) : content.slice(end, nl);
  return tail.trim() === "";
}

/** Merged protected spans (code, math, divs, blockquotes, frontmatter). */
export function protectedSpans(content: string): Span[] {
  const spans: Span[] = [
    ...regexSpans(content, CODE_FENCE_RE),
    ...regexSpans(content, MATH_BLOCK_RE).filter(([s, e]) => isBlockAnchored(content, s, e)),
    ...regexSpans(content, BLOCKQUOTE_RE),
    ...findDivBlocks(content),
  ];
  const fm = frontmatterSpan(content);
  if (fm) {
    spans.push(fm);
  }
  spans.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const merged: Span[] = [];
  for (const [start, end] of spans) {
    const last = merged[merged.length - 1];
    if (last && start <= last[1]) {
      last[1] = Math.max(last[1], end);
    } else {
      merged.push([start, end]);
    }
  }
  return merged;
}

// Port of stylebot.markdown.ATX_HEADING_RE, tested against a paragraph's FIRST
// line: up to 3 *spaces* of indent (a tab indents by 4, so it cannot open a
// heading), 1-6 hashes, then end-of-line or a space/tab and the title.
const HEADING_RE = /^ {0,3}#{1,6}(?:[ \t].*?)?[ \t]*$/;

/**
 * The scoreable prose paragraphs of a markdown document, with offsets.
 *
 * Within each editable (non-protected) span, paragraphs are blank-line runs
 * (split_paragraphs semantics); heading-only paragraphs and paragraphs
 * shorter than `minChars` are dropped.
 */
export function proseParagraphs(content: string, minChars = 0): Paragraph[] {
  const paragraphs: Paragraph[] = [];
  let cursor = 0;
  const editable: Span[] = [];
  for (const [s, e] of protectedSpans(content)) {
    if (cursor < s) {
      editable.push([cursor, s]);
    }
    cursor = e;
  }
  if (cursor < content.length) {
    editable.push([cursor, content.length]);
  }

  for (const [spanStart, spanEnd] of editable) {
    let paraStart = -1; // -1 = not inside a paragraph
    let lineStart = spanStart;
    let lastNonBlankEnd = spanStart;
    const flush = (endExclusive: number) => {
      if (paraStart >= 0) {
        pushParagraph(paragraphs, content, paraStart, endExclusive, minChars);
        paraStart = -1;
      }
    };
    while (lineStart < spanEnd) {
      let lineEnd = content.indexOf("\n", lineStart);
      if (lineEnd === -1 || lineEnd >= spanEnd) {
        lineEnd = spanEnd;
      } else {
        lineEnd += 1; // include the newline
      }
      const line = content.slice(lineStart, lineEnd);
      if (line.trim() === "") {
        flush(lastNonBlankEnd);
      } else {
        if (paraStart < 0) {
          paraStart = lineStart;
        }
        lastNonBlankEnd = lineEnd;
      }
      lineStart = lineEnd;
    }
    flush(lastNonBlankEnd);
  }
  return paragraphs;
}

function pushParagraph(
  out: Paragraph[],
  content: string,
  start: number,
  end: number,
  minChars: number,
): void {
  // rstrip("\n") parity with split_paragraphs
  while (end > start && content[end - 1] === "\n") {
    end -= 1;
  }
  const text = content.slice(start, end);
  if (text.length === 0 || HEADING_RE.test(text.split("\n", 1)[0])) {
    return;
  }
  if (text.length < minChars) {
    return;
  }
  out.push({ text, start, end });
}
