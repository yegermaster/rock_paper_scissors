import { NextRequest } from "next/server";
import { ImageResponse } from "next/og";
import { getSupabaseServerClient } from "@/lib/supabase";
import { expandOccurrences } from "@/lib/recurrence";
import { addDays, daysInMonth, startOfMonth, startOfWeek, startOfYear } from "@/lib/dates";
import { todayInAppTimezone } from "@/lib/timezone";
import { renderWeekView } from "@/lib/views/week";
import { renderMonthView } from "@/lib/views/month";
import { renderYearView } from "@/lib/views/year";
import type { EventRecord, ViewKind } from "@/lib/types";

export const runtime = "edge";

function rangeFor(view: ViewKind, anchor: string): { start: string; end: string } {
  if (view === "week") {
    const start = startOfWeek(anchor);
    return { start, end: addDays(start, 6) };
  }
  if (view === "month") {
    const start = startOfMonth(anchor);
    const [y, m] = start.split("-").map(Number);
    return { start, end: `${y}-${String(m).padStart(2, "0")}-${daysInMonth(y, m)}` };
  }
  const start = startOfYear(anchor);
  const y = Number(start.slice(0, 4));
  return { start, end: `${y}-12-31` };
}

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const view = (searchParams.get("view") as ViewKind) || "week";
  const anchor = searchParams.get("date") || todayInAppTimezone();
  const today = todayInAppTimezone();

  const { start, end } = rangeFor(view, anchor);

  const supabase = getSupabaseServerClient();
  const { data, error } = await supabase.from("events").select("*");
  if (error) {
    return new Response(`Failed to load events: ${error.message}`, { status: 500 });
  }
  const events = (data ?? []) as EventRecord[];
  const occurrences = expandOccurrences(events, start, end);

  const width = 1400;
  let node;
  if (view === "week") node = renderWeekView(start, occurrences, today);
  else if (view === "month") node = renderMonthView(start, occurrences, today);
  else node = renderYearView(start, occurrences, today);

  return new ImageResponse(node, { width });
}
