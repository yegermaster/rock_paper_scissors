import { NextRequest, NextResponse } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabase";
import { parseMessage, reviseAction } from "@/lib/gemini";
import { expandOccurrences } from "@/lib/recurrence";
import { addDays } from "@/lib/dates";
import { rangeFor } from "@/lib/view-range";
import { todayInAppTimezone } from "@/lib/timezone";
import { formatHebrewDateLong, formatRecurrenceHebrew, formatTimeHebrew } from "@/lib/hebrew-format";
import type { EventRecord, PendingAction, ParsedEventFields, ViewKind } from "@/lib/types";

export const runtime = "nodejs";
// Run near the Supabase project's region (Seoul) — see calendar-image/route.tsx.
export const preferredRegion = "icn1";

function toTimeColumn(hhmm: string | null | undefined): string | null {
  if (!hhmm) return null;
  return hhmm.length === 5 ? `${hhmm}:00` : hhmm;
}

function describeView(view: unknown, anchor: unknown): string | null {
  if (typeof view !== "string" || typeof anchor !== "string") return null;
  if (view !== "week" && view !== "month" && view !== "year") return null;
  const { start, end } = rangeFor(view as ViewKind, anchor);
  return start === end ? start : `${start} to ${end}`;
}

const AFFIRMATIVE = new Set([
  "כן", "כן בבקשה", "אשר", "מאשר", "מאשרת", "בטח", "סבבה", "אוקיי", "אוקי", "אישור", "בסדר", "בסדר גמור",
  "yes", "y", "ok", "okay", "confirm", "sure", "yep", "yup",
]);
const NEGATIVE = new Set([
  "לא", "לא תודה", "בטל", "ביטול", "עזוב", "עזבי", "תשכח מזה", "no", "n", "cancel", "nope",
]);

function classifyYesNo(text: string): "yes" | "no" | "other" {
  const t = text.trim().toLowerCase().replace(/[!.?׳"]+$/, "");
  if (AFFIRMATIVE.has(t)) return "yes";
  if (NEGATIVE.has(t)) return "no";
  return "other";
}

function summarize(action: PendingAction): string {
  if (action.intent === "add" && action.event) {
    const e = action.event;
    const lines = [
      `להוסיף: "${e.title}"`,
      `📅 ${e.start_date ? formatHebrewDateLong(e.start_date) : "?"}`,
      `🕐 ${formatTimeHebrew(e.start_time)}`,
      `🔁 ${formatRecurrenceHebrew(e.recurrence?.frequency, e.recurrence?.days_of_week, e.recurrence?.interval, e.recurrence?.end_date)}`,
    ];
    if (e.person) lines.push(`👤 ${e.person}`);
    lines.push("לאשר? (כן / לא)");
    return lines.join("\n");
  }
  if (action.intent === "delete") {
    const note = action.targetIsRecurring
      ? "\n⚠️ זה ימחק את כל סדרת האירועים החוזרים, לא רק מופע אחד."
      : "";
    return `למחוק את "${action.targetTitle}"?${note}\nלאשר? (כן / לא)`;
  }
  // edit
  const e = action.event;
  const parts: string[] = [];
  if (e?.title) parts.push(`כותרת חדשה: "${e.title}"`);
  if (e?.start_date) parts.push(`תאריך חדש: ${formatHebrewDateLong(e.start_date)}`);
  if (e?.start_time !== undefined && e?.start_time !== null) parts.push(`שעה חדשה: ${formatTimeHebrew(e.start_time)}`);
  if (e?.category) parts.push(`קטגוריה חדשה: ${e.category}`);
  if (e?.recurrence) parts.push(`חזרתיות: ${formatRecurrenceHebrew(e.recurrence.frequency, e.recurrence.days_of_week, e.recurrence.interval, e.recurrence.end_date)}`);
  return `לעדכן את "${action.targetTitle}":\n${parts.join("\n")}\nלאשר? (כן / לא)`;
}

function needsConfirmationResponse(action: PendingAction) {
  return NextResponse.json({
    ok: false,
    needsConfirmation: true,
    message: summarize(action),
    pendingAction: action,
  });
}

async function commitAction(supabase: ReturnType<typeof getSupabaseServerClient>, action: PendingAction) {
  if (action.intent === "add") {
    const e = action.event as ParsedEventFields;
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
      return NextResponse.json({ ok: false, message: "לא הצלחתי לשמור — נסה שוב.", pendingAction: null }, { status: 500 });
    }
    return NextResponse.json({ ok: true, changed: true, message: `נוסף "${e.title}" ✅`, pendingAction: null });
  }

  if (action.intent === "delete") {
    const { error } = await supabase.from("events").delete().eq("id", action.targetEventId);
    if (error) {
      return NextResponse.json({ ok: false, message: "לא הצלחתי למחוק — נסה שוב.", pendingAction: null }, { status: 500 });
    }
    const note = action.targetIsRecurring ? " (כל הסדרה בוטלה)" : "";
    return NextResponse.json({ ok: true, changed: true, message: `נמחק "${action.targetTitle}"${note} ✅`, pendingAction: null });
  }

  // edit
  const e = action.event as ParsedEventFields;
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
  const { error } = await supabase.from("events").update(update).eq("id", action.targetEventId);
  if (error) {
    return NextResponse.json({ ok: false, message: "לא הצלחתי לעדכן — נסה שוב.", pendingAction: null }, { status: 500 });
  }
  return NextResponse.json({ ok: true, changed: true, message: `עודכן "${action.targetTitle}" ✅`, pendingAction: null });
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
  const pendingAction: PendingAction | null = body?.pendingAction ?? null;
  if (!text || typeof text !== "string") {
    return NextResponse.json({ ok: false, message: "לא התקבל טקסט." }, { status: 400 });
  }

  const currentDate = todayInAppTimezone();
  const supabase = getSupabaseServerClient();
  const viewContext = describeView(body?.view, body?.anchor);

  const clarify = (question: string) => {
    const nextContext = `${context ? context + "\n" : ""}User: ${text}\nAssistant: ${question}`;
    return NextResponse.json({ ok: false, needsClarification: true, message: question, context: nextContext, pendingAction: null });
  };

  // A confirmation is already pending — this reply is answering it.
  if (pendingAction) {
    const cls = classifyYesNo(text);
    if (cls === "yes") return await commitAction(supabase, pendingAction);
    if (cls === "no") {
      return NextResponse.json({ ok: true, changed: false, message: "בסדר, ביטלתי.", pendingAction: null });
    }

    let revised;
    try {
      revised = await reviseAction({ pendingEvent: pendingAction.event, text, currentDate });
    } catch (err) {
      console.error("Gemini revise failed", err);
      return NextResponse.json(
        { ok: false, message: "לא הצלחתי להבין את התשובה — אפשר לכתוב 'כן' לאישור או 'לא' לביטול?" },
        { status: 502 }
      );
    }
    if (revised.cancelled) {
      return NextResponse.json({ ok: true, changed: false, message: "בסדר, ביטלתי.", pendingAction: null });
    }
    if (revised.confirmed) return await commitAction(supabase, pendingAction);

    const updatedAction: PendingAction = {
      ...pendingAction,
      event: revised.updatedEvent ?? pendingAction.event,
    };
    return needsConfirmationResponse(updatedAction);
  }

  // Fresh message — parse it.
  let parsed;
  try {
    parsed = await parseMessage({ text, currentDate, context, viewContext });
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
    return needsConfirmationResponse({ intent: "add", event: e, targetEventId: null, targetTitle: null, targetIsRecurring: false });
  }

  if (parsed.intent === "delete" || parsed.intent === "edit") {
    const { event, candidates } = await findTarget(supabase, parsed.target, currentDate);

    if (!event) {
      if (candidates.length === 0) {
        return clarify("לא מצאתי אירוע מתאים — לאיזה אירוע התכוונת?");
      }
      const options = candidates.slice(0, 5).map((c) => `"${c.event.title}" בתאריך ${c.date}`).join(", ");
      return clarify(`מצאתי כמה התאמות: ${options}. לאיזו התכוונת?`);
    }

    if (parsed.intent === "delete") {
      return needsConfirmationResponse({
        intent: "delete",
        event: null,
        targetEventId: event.id,
        targetTitle: event.title,
        targetIsRecurring: !!event.recurrence_frequency,
      });
    }

    // edit
    const e = parsed.event;
    if (!e) return clarify(`מה לשנות ב-"${event.title}"?`);
    return needsConfirmationResponse({
      intent: "edit",
      event: e,
      targetEventId: event.id,
      targetTitle: event.title,
      targetIsRecurring: !!event.recurrence_frequency,
    });
  }

  return NextResponse.json({ ok: true, message: "בוצע." });
}
