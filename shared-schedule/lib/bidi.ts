// next/og (Satori) has NO bidi algorithm: it draws characters in logical
// (storage) order, left to right, which renders Hebrew mirror-reversed.
// Browsers fix this automatically for HTML; for the PNG renderer we must
// reorder text to VISUAL order ourselves.
//
// `he()` converts a logical string to visual order for an RTL context:
// the character sequence is reversed, except runs of LTR characters
// (digits, Latin letters, and clock-time punctuation between them) which
// keep their internal order — so "יולי 2026" renders with "2026" intact
// to the left of "יולי", and "19:00" is never mangled.

const LTR_CHAR = /[A-Za-z0-9]/;
const LTR_JOINER = /[:./-]/; // keeps "19:00", "12.5", "3-4" together inside an LTR run

export function he(input: string): string {
  const chars = [...input];
  const runs: { ltr: boolean; text: string }[] = [];

  for (let i = 0; i < chars.length; i++) {
    const ch = chars[i];
    const prev = runs[runs.length - 1];
    const isLtr =
      LTR_CHAR.test(ch) ||
      // joiner counts as LTR only when sandwiched between LTR chars
      (LTR_JOINER.test(ch) &&
        prev?.ltr === true &&
        i + 1 < chars.length &&
        LTR_CHAR.test(chars[i + 1]));

    if (prev && prev.ltr === isLtr) prev.text += ch;
    else runs.push({ ltr: isLtr, text: ch });
  }

  return runs
    .reverse()
    .map((r) => (r.ltr ? r.text : [...r.text].reverse().join("")))
    .join("");
}
