import { NextRequest, NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabase";
import { expandOccurrences } from "@/lib/recurrence";
import { rangeFor } from "@/lib/view-range";
import { todayInAppTimezone } from "@/lib/timezone";
import type { EventRecord, ViewKind } from "@/lib/types";

export const runtime = "nodejs";
export const preferredRegion = "icn1";

function toTimeColumn(hhmm: string | null | undefined): string | null {
  if (!hhmm) return null;
  return hhmm.length === 5 ? `${hhmm}:00` : hhmm;
}

// Manual (non-AI) CRUD for the event list panel under the calendar image.

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const view = (searchParams.get("view") as ViewKind) || "week";
  const anchor = searchParams.get("date") || todayInAppTimezone();
  const { start, end } = rangeFor(view, anchor);

  const supabase = getSupabaseServerClient();
  const { data, error } = await supabase.from("events").select("*");
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  const events = (data ?? []) as EventRecord[];
  const occurrences = expandOccurrences(events, start, end);

  const items = occurrences
    .map((occ) => ({
      id: occ.event.id,
      occurrenceDate: occ.date,
      title: occ.event.title,
      category: occ.event.category,
      person: occ.event.person,
      startTime: occ.event.start_time ? occ.event.start_time.slice(0, 5) : null,
      durationMinutes: occ.event.duration_minutes,
      isRecurring: !!occ.event.recurrence_frequency,
      recurrenceFrequency: occ.event.recurrence_frequency,
      recurrenceInterval: occ.event.recurrence_interval,
      recurrenceDaysOfWeek: occ.event.recurrence_days_of_week,
      recurrenceEndDate: occ.event.recurrence_end_date,
      isMultiDaySpan: occ.isMultiDaySpan,
      spanStart: occ.spanStart,
      spanEnd: occ.spanEnd,
    }))
    .sort((a, b) => a.occurrenceDate.localeCompare(b.occurrenceDate) || (a.startTime ?? "99:99").localeCompare(b.startTime ?? "99:99"));

  return NextResponse.json({ items });
}

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => null);
  if (!body?.title || !body?.start_date) {
    return NextResponse.json({ error: "כותרת ותאריך הם שדות חובה." }, { status: 400 });
  }

  const supabase = getSupabaseServerClient();
  const { error } = await supabase.from("events").insert({
    title: body.title,
    category: body.category || "other",
    person: body.person || null,
    start_date: body.start_date,
    end_date: body.end_date || body.start_date,
    start_time: toTimeColumn(body.start_time || null),
    duration_minutes: body.duration_minutes || 60,
    recurrence_frequency: body.recurrence?.frequency || null,
    recurrence_interval: body.recurrence?.interval || (body.recurrence?.frequency ? 1 : null),
    recurrence_days_of_week: body.recurrence?.days_of_week || null,
    recurrence_end_date: body.recurrence?.end_date || null,
  });

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json({ ok: true });
}
