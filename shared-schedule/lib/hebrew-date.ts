import { HDate } from "@hebcal/core";

/** Full Hebrew-calendar date in Hebrew gematria, e.g. "כ״ה תמוז תשפ״ו".
 * Niqqud (vowel points) suppressed — everyday Hebrew usage doesn't show them. */
export function hebrewDateGematriya(ymd: string): string {
  const [y, m, d] = ymd.split("-").map(Number);
  const hd = new HDate(new Date(y, m - 1, d));
  return hd.renderGematriya(true);
}

/** Day + month only (no year), for compact display in dense grids. */
export function hebrewDateShort(ymd: string): string {
  const [y, m, d] = ymd.split("-").map(Number);
  const hd = new HDate(new Date(y, m - 1, d));
  return hd.renderGematriya(true).split(" ").slice(0, 2).join(" ");
}
