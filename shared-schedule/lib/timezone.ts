// Single fixed timezone for the whole app — "today" is computed in this
// zone, never per-viewer. No conversion logic exists anywhere else.
export function todayInAppTimezone(): string {
  const tz = process.env.APP_TIMEZONE || "UTC";
  const formatter = new Intl.DateTimeFormat("en-CA", {
    timeZone: tz,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  return formatter.format(new Date()); // en-CA gives YYYY-MM-DD
}
