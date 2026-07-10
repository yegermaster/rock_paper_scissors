import { NextRequest } from "next/server";
import { ImageResponse } from "next/og";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { getSupabaseServerClient } from "@/lib/supabase";
import { expandOccurrences } from "@/lib/recurrence";
import { addDays, daysInMonth, startOfMonth, startOfWeek, startOfYear } from "@/lib/dates";
import { todayInAppTimezone } from "@/lib/timezone";
import { renderWeekView } from "@/lib/views/week";
import { renderMonthView } from "@/lib/views/month";
import { renderYearView } from "@/lib/views/year";
import type { EventRecord, ViewKind } from "@/lib/types";

// nodejs (not edge): @hebcal/core's Hebrew-calendar math isn't guaranteed
// edge-runtime safe, and next/og's ImageResponse works fine under either.
export const runtime = "nodejs";

const FONTS_DIR = path.join(process.cwd(), "app", "api", "calendar-image", "fonts");

async function loadHebrewFonts() {
  const [regular, bold] = await Promise.all([
    readFile(path.join(FONTS_DIR, "NotoSansHebrew-Regular.ttf")),
    readFile(path.join(FONTS_DIR, "NotoSansHebrew-Bold.ttf")),
  ]);
  return [
    { name: "Noto Sans Hebrew", data: regular, weight: 400 as const, style: "normal" as const },
    { name: "Noto Sans Hebrew", data: bold, weight: 700 as const, style: "normal" as const },
  ];
}

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

  const rendered =
    view === "week"
      ? renderWeekView(start, occurrences, today)
      : view === "month"
        ? renderMonthView(start, occurrences, today)
        : renderYearView(start, occurrences, today);

  const fonts = await loadHebrewFonts();
  return new ImageResponse(rendered.node, { width: rendered.width, height: rendered.height, fonts });
}
