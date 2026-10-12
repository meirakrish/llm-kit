/**
 * Text cut off by a length limit ends mid-sentence; trim it back to the last complete sentence.
 * Text that is clearly shorter than the limit, or already ends cleanly, is returned trimmed but otherwise unchanged.
 */
export function trimToSentence(text: string, limit: number): string {
  const t = text.trim();
  if (t.length < limit - 5 || /[.!?)`"']$/.test(t)) return t;
  const end = Math.max(t.lastIndexOf(". "), t.lastIndexOf("! "), t.lastIndexOf("? "));
  return end > t.length / 3 ? t.slice(0, end + 1) : `${t}…`;
}

/**
 * Shortens text to at most `maxChars`, keeping the start and the end and cutting the
 * middle. For logs and describe output the end usually holds the error, so the tail
 * gets the larger share. Cuts happen at line breaks when possible.
 */
export function truncateMiddle(text: string, maxChars: number, headShare = 0.3): string {
  if (text.length <= maxChars) return text;
  const marker = (omitted: number) => `\n... [${omitted} characters truncated] ...\n`;
  const budget = Math.max(0, maxChars - marker(text.length).length);
  let head = text.slice(0, Math.floor(budget * headShare));
  let tail = text.slice(text.length - (budget - head.length));

  // Prefer whole lines: trim the head back to its last newline, the tail forward to its first.
  const headCut = head.lastIndexOf("\n");
  if (headCut > head.length / 2) head = head.slice(0, headCut);
  const tailCut = tail.indexOf("\n");
  if (tailCut !== -1 && tailCut < tail.length / 2) tail = tail.slice(tailCut + 1);

  return `${head}${marker(text.length - head.length - tail.length)}${tail}`;
}
