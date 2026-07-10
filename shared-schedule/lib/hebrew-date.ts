import { HDate } from "@hebcal/core";

/** Full Hebrew-calendar date in Hebrew gematria, e.g. "כ״ה תַּמּוּז תשפ״ו". */
export function hebrewDateGematriya(ymd: string): string {
  const [y, m, d] = ymd.split("-").map(Number);
  const hd = new HDate(new Date(y, m - 1, d));
  return hd.renderGematriya();
}

/** Day + month only (no year), for compact display in dense grids. */
export function hebrewDateShort(ymd: string): string {
  const [y, m, d] = ymd.split("-").map(Number);
  const hd = new HDate(new Date(y, m - 1, d));
  return hd.renderGematriya().split(" ").slice(0, 2).join(" ");
}
