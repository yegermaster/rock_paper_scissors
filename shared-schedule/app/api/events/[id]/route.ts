import { NextRequest, NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabase";
import { normalizePerson } from "@/lib/people";
import type { EventRecord } from "@/lib/types";

export const runtime = "nodejs";
export const preferredRegion = "icn1";

function toTimeColumn(hhmm: string | null | undefined): string | null {
  if (!hhmm) return null;
  return hhmm.length === 5 ? `${hhmm}:00` : hhmm;
}

export async function PATCH(req: NextRequest, { params }: { params: { id: string } }) {
  const body = await req.json().catch(() => null);
  if (!body) return NextResponse.json({ error: "גוף בקשה לא תקין." }, { status: 400 });

  const update: Partial<EventRecord> = {};
  if (body.title !== undefined) update.title = body.title;
  if (body.category !== undefined) update.category = body.category;
  if (body.person !== undefined) update.person = normalizePerson(body.person);
  if (body.start_date !== undefined) update.start_date = body.start_date;
  if (body.end_date !== undefined) update.end_date = body.end_date || body.start_date;
  if (body.start_time !== undefined) update.start_time = toTimeColumn(body.start_time);
  if (body.duration_minutes !== undefined) update.duration_minutes = body.duration_minutes;
  if (body.recurrence !== undefined) {
    update.recurrence_frequency = body.recurrence?.frequency || null;
    update.recurrence_interval = body.recurrence?.interval || (body.recurrence?.frequency ? 1 : null);
    update.recurrence_days_of_week = body.recurrence?.days_of_week || null;
    update.recurrence_end_date = body.recurrence?.end_date || null;
  }

  const supabase = getSupabaseServerClient();
  const { error } = await supabase.from("events").update(update).eq("id", params.id);
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json({ ok: true });
}

export async function DELETE(_req: NextRequest, { params }: { params: { id: string } }) {
  const supabase = getSupabaseServerClient();
  const { error } = await supabase.from("events").delete().eq("id", params.id);
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json({ ok: true });
}
