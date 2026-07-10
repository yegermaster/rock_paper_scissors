// All dates are treated as plain calendar dates (YYYY-MM-DD), arithmetic done
// against UTC-midnight Date objects purely as a calendar, never as a real
// instant-in-time — matches the "single fixed timezone" decision (no
// timezone conversion happens anywhere in this app).

export function toDate(ymd: string): Date {
  const [y, m, d] = ymd.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d));
}

export function toYMD(date: Date): string {
  return date.toISOString().slice(0, 10);
}

export function addDays(ymd: string, days: number): string {
  const d = toDate(ymd);
  d.setUTCDate(d.getUTCDate() + days);
  return toYMD(d);
}

export function diffDays(fromYMD: string, toYMDStr: string): number {
  const a = toDate(fromYMD).getTime();
  const b = toDate(toYMDStr).getTime();
  return Math.round((b - a) / 86400000);
}

export function dayOfWeek(ymd: string): number {
  return toDate(ymd).getUTCDay(); // 0=Sunday..6=Saturday
}

export function startOfWeek(ymd: string): string {
  return addDays(ymd, -dayOfWeek(ymd));
}

export function startOfMonth(ymd: string): string {
  const [y, m] = ymd.split("-").map(Number);
  return `${y}-${String(m).padStart(2, "0")}-01`;
}

export function startOfYear(ymd: string): string {
  const [y] = ymd.split("-").map(Number);
  return `${y}-01-01`;
}

export function addMonths(ymd: string, months: number): string {
  const [y, m, d] = ymd.split("-").map(Number);
  const date = new Date(Date.UTC(y, m - 1 + months, 1));
  return toYMD(date);
}

export function addYears(ymd: string, years: number): string {
  const [y, m, d] = ymd.split("-").map(Number);
  return `${y + years}-${String(m).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
}

export function daysInMonth(year: number, month1to12: number): number {
  return new Date(Date.UTC(year, month1to12, 0)).getUTCDate();
}

// Gregorian weekday/month names, in Hebrew (the Hebrew *calendar* dates are
// separate — see lib/hebrew-date.ts).
export const WEEKDAY_LABELS = ["א׳", "ב׳", "ג׳", "ד׳", "ה׳", "ו׳", "ש׳"];
export const MONTH_LABELS = [
  "ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני",
  "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר",
];
