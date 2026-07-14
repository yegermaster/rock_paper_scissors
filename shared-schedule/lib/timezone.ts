// Single fixed timezone for the whole app — "today" is computed in this
// zone, never per-viewer. No conversion logic exists anywhere else.
//
// Hardcoded rather than read from an env var: this app is permanently a
// calendar for a couple in Tel Aviv, and a misconfigured/missing
// APP_TIMEZONE env var (it was set to Asia/Seoul — likely confused with
// the unrelated Vercel deployment region, icn1/Seoul, chosen only for
// latency to the Supabase project) silently shifted "today" by hours,
// making the app show the wrong date around midnight Israel time.
const APP_TIMEZONE = "Asia/Jerusalem";

export function todayInAppTimezone(): string {
  const formatter = new Intl.DateTimeFormat("en-CA", {
    timeZone: APP_TIMEZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  return formatter.format(new Date()); // en-CA gives YYYY-MM-DD
}
