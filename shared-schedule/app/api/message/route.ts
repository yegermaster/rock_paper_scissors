import { NextRequest, NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabase";
import { parseMessage } from "@/lib/gemini";
import { expandOccurrences } from "@/lib/recurrence";
import { addDays } from "@/lib/dates";
import { todayInAppTimezone } from "@/lib/timezone";
import type { EventRecord } from "@/lib/types";

export const runtime = "nodejs";

function toTimeColumn(hhmm: string | null | undefined): string | null {
  if (!hhmm) return null;
  return hhmm.length === 5 ? `${hhmm}:00` : hhmm;
}

async function findTarget(
  supabase: ReturnType<typeof getSupabaseServerClient>,
  target: { date_hint: string | null; title_hint: string | null } | null,
  currentDate: string
): Promise<{ event: EventRecord | null; candidates: { event: EventRecord; date: string }[] }> {
  const { data, error } = await supabase.from("events").select("*");
  if (error) throw error;
  const all = (data ?? []) as EventRecord[];

  const windowStart = target?.date_hint ?? currentDate;
  const windowEnd = target?.date_hint ?? addDays(currentDate, 90);

  const occurrences = expandOccurrences(all, windowStart, windowEnd);

  const titleHint = target?.title_hint?.trim().toLowerCase();
  const matches = occurrences.filter((occ) => {
    if (target?.date_hint && occ.date !== target.date_hint) return false;
    if (titleHint) {
      const hay = `${occ.event.title} ${occ.event.category}`.toLowerCase();
      if (!hay.includes(titleHint)) return false;
    }
    return true;
  });

  // Dedupe by event id (a recurring event may match multiple occurrences).
  const byId = new Map<string, { event: EventRecord; date: string }>();
  for (const m of matches) {
    if (!byId.has(m.event.id)) byId.set(m.event.id, { event: m.event, date: m.date });
  }
  const candidates = [...byId.values()];

  if (candidates.length === 1) return { event: candidates[0].event, candidates };
  return { event: null, candidates };
}

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => null);
  const text = body?.text;
  const context: string | null = typeof body?.context === "string" ? body.context : null;
  if (!text || typeof text !== "string") {
    return NextResponse.json({ ok: false, message: "לא התקבל טקסט." }, { status: 400 });
  }

  const currentDate = todayInAppTimezone();
  const supabase = getSupabaseServerClient();

  const clarify = (question: string) => {
    const nextContext = `${context ? context + "\n" : ""}User: ${text}\nAssistant: ${question}`;
    return NextResponse.json({ ok: false, needsClarification: true, message: question, context: nextContext });
  };

  let parsed;
  try {
    parsed = await parseMessage({ text, currentDate, context });
  } catch (err) {
    console.error("Gemini parse failed", err);
    return NextResponse.json(
      { ok: false, message: "לא הצלחתי להתחבר למנוע הניתוח — נסה שוב בעוד רגע." },
      { status: 502 }
    );
  }

  if (parsed.needs_clarification || parsed.intent === "unknown") {
    return clarify(parsed.clarification_question ?? "סליחה, לא הבנתי — אפשר לנסח את זה מחדש?");
  }

  if (parsed.intent === "add") {
    const e = parsed.event;
    if (!e || !e.title || !e.start_date) {
      return clarify("מה האירוע, ובאיזה תאריך?");
    }

    const { error } = await supabase.from("events").insert({
      title: e.title,
      category: e.category ?? "other",
      person: e.person ?? null,
      start_date: e.start_date,
      end_date: e.end_date ?? e.start_date,
      start_time: toTimeColumn(e.start_time),
      duration_minutes: e.duration_minutes ?? 60,
      recurrence_frequency: e.recurrence?.frequency ?? null,
      recurrence_interval: e.recurrence?.interval ?? (e.recurrence?.frequency ? 1 : null),
      recurrence_days_of_week: e.recurrence?.days_of_week ?? null,
      recurrence_end_date: e.recurrence?.end_date ?? null,
    });

    if (error) {
      console.error("Insert failed", error);
      return NextResponse.json({ ok: false, message: "לא הצלחתי לשמור — נסה שוב." }, { status: 500 });
    }

    return NextResponse.json({ ok: true, message: `נוסף "${e.title}" ✅` });
  }

  if (parsed.intent === "delete" || parsed.intent === "edit") {
    const { event, candidates } = await findTarget(supabase, parsed.target, currentDate);

    if (!event) {
      if (candidates.length === 0) {
        return clarify("לא מצאתי אירוע מתאים — לאיזה אירוע התכוונת?");
      }
      const options = candidates
        .slice(0, 5)
        .map((c) => `"${c.event.title}" בתאריך ${c.date}`)
        .join(", ");
      return clarify(`מצאתי כמה התאמות: ${options}. לאיזו התכוונת?`);
    }

    if (parsed.intent === "delete") {
      const { error } = await supabase.from("events").delete().eq("id", event.id);
      if (error) {
        return NextResponse.json({ ok: false, message: "לא הצלחתי למחוק — נסה שוב." }, { status: 500 });
      }
      const note = event.recurrence_frequency
        ? " (זה מבטל את כל סדרת האירועים החוזרים, לא רק מופע אחד)"
        : "";
      return NextResponse.json({ ok: true, message: `נמחק "${event.title}"${note} ✅` });
    }

    // edit
    const e = parsed.event;
    if (!e) {
      return clarify(`מה לשנות ב-"${event.title}"?`);
    }
    const update: Partial<EventRecord> = {};
    if (e.title) update.title = e.title;
    if (e.category) update.category = e.category;
    if (e.person !== undefined && e.person !== null) update.person = e.person;
    if (e.start_date) update.start_date = e.start_date;
    if (e.end_date) update.end_date = e.end_date;
    if (e.start_time !== undefined) update.start_time = toTimeColumn(e.start_time);
    if (e.duration_minutes) update.duration_minutes = e.duration_minutes;
    if (e.recurrence) {
      update.recurrence_frequency = e.recurrence.frequency ?? null;
      update.recurrence_interval = e.recurrence.interval ?? null;
      update.recurrence_days_of_week = e.recurrence.days_of_week ?? null;
      update.recurrence_end_date = e.recurrence.end_date ?? null;
    }

    const { error } = await supabase.from("events").update(update).eq("id", event.id);
    if (error) {
      return NextResponse.json({ ok: false, message: "לא הצלחתי לעדכן — נסה שוב." }, { status: 500 });
    }
    return NextResponse.json({ ok: true, message: `עודכן "${event.title}" ✅` });
  }

  return NextResponse.json({ ok: true, message: "בוצע." });
}
