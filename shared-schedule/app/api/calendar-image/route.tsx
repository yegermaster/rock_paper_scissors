import { NextRequest } from "next/server";
import { ImageResponse } from "next/og";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { getSupabaseServerClient } from "@/lib/supabase";
import { expandOccurrences } from "@/lib/recurrence";
import { rangeFor } from "@/lib/view-range";
import { todayInAppTimezone } from "@/lib/timezone";
import { renderWeekView } from "@/lib/views/week";
import { renderMonthView } from "@/lib/views/month";
import { renderYearView } from "@/lib/views/year";
import type { EventRecord, ViewKind } from "@/lib/types";

// nodejs (not edge): @hebcal/core's Hebrew-calendar math isn't guaranteed
// edge-runtime safe, and next/og's ImageResponse works fine under either.
export const runtime = "nodejs";
// Run near the Supabase project's region (Seoul) — every request here does
// a DB round-trip, and a cross-Pacific hop would otherwise add hundreds of
// ms per image render on top of everything else.
export const preferredRegion = "icn1";

const FONTS_DIR = path.join(process.cwd(), "app", "api", "calendar-image", "fonts");

// Cached at module scope: a warm serverless instance reuses the same font
// buffers across requests instead of hitting disk on every image render.
let fontsPromise: ReturnType<typeof loadHebrewFontsUncached> | null = null;

async function loadHebrewFontsUncached() {
  const [regular, bold] = await Promise.all([
    readFile(path.join(FONTS_DIR, "NotoSansHebrew-Regular.ttf")),
    readFile(path.join(FONTS_DIR, "NotoSansHebrew-Bold.ttf")),
  ]);
  return [
    { name: "Noto Sans Hebrew", data: regular, weight: 400 as const, style: "normal" as const },
    { name: "Noto Sans Hebrew", data: bold, weight: 700 as const, style: "normal" as const },
  ];
}

function loadHebrewFonts() {
  if (!fontsPromise) fontsPromise = loadHebrewFontsUncached();
  return fontsPromise;
}

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const view = (searchParams.get("view") as ViewKind) || "week";
  const anchor = searchParams.get("date") || todayInAppTimezone();
  const today = todayInAppTimezone();

  const { start, end } = rangeFor(view, anchor);

  const supabase = getSupabaseServerClient();
  // Fonts (cached after the first request) and the Supabase fetch don't
  // depend on each other — run them concurrently instead of back-to-back.
  const [{ data, error }, fonts] = await Promise.all([
    supabase.from("events").select("*"),
    loadHebrewFonts(),
  ]);
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

  return new ImageResponse(rendered.node, { width: rendered.width, height: rendered.height, fonts });
}
