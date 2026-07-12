import { dayOfWeek, MONTH_LABELS } from "./dates";

const WEEKDAY_FULL = ["יום ראשון", "יום שני", "יום שלישי", "יום רביעי", "יום חמישי", "יום שישי", "שבת"];
const DOW_NAMES = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"];

export function formatHebrewDateLong(ymd: string): string {
  const [y, m, d] = ymd.split("-").map(Number);
  return `${WEEKDAY_FULL[dayOfWeek(ymd)]}, ${d} ב${MONTH_LABELS[m - 1]} ${y}`;
}

export function formatTimeHebrew(hhmmss: string | null | undefined): string {
  if (!hhmmss) return "ללא שעה קבועה";
  return hhmmss.slice(0, 5);
}

export function formatRecurrenceHebrew(
  frequency: "daily" | "weekly" | "monthly" | null | undefined,
  daysOfWeek: number[] | null | undefined,
  interval: number | null | undefined,
  endDate: string | null | undefined
): string {
  if (!frequency) return "חד פעמי";
  const n = interval ?? 1;
  const unit =
    frequency === "daily" ? (n > 1 ? "ימים" : "יום")
    : frequency === "weekly" ? (n > 1 ? "שבועות" : "שבוע")
    : (n > 1 ? "חודשים" : "חודש");
  const prefix = n > 1 ? `כל ${n} ${unit}` : `כל ${unit}`;
  const days =
    frequency === "weekly" && daysOfWeek && daysOfWeek.length > 0
      ? ` (${daysOfWeek.map((d) => DOW_NAMES[d]).join(", ")})`
      : "";
  const until = endDate ? ` עד ${formatHebrewDateLong(endDate)}` : "";
  return `חוזר ${prefix}${days}${until}`;
}
