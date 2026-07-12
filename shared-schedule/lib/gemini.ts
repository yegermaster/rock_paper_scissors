import { GoogleGenerativeAI, SchemaType } from "@google/generative-ai";
import type { ParsedIntent, ParsedEventFields, RevisedAction } from "./types";

const eventSchema = {
  type: SchemaType.OBJECT,
  nullable: true,
  properties: {
    title: { type: SchemaType.STRING, nullable: true },
    category: { type: SchemaType.STRING, nullable: true },
    person: { type: SchemaType.STRING, nullable: true, enum: ["itamar", "hadas", "both"] },
    start_date: { type: SchemaType.STRING, nullable: true },
    end_date: { type: SchemaType.STRING, nullable: true },
    start_time: { type: SchemaType.STRING, nullable: true },
    duration_minutes: { type: SchemaType.NUMBER, nullable: true },
    recurrence: {
      type: SchemaType.OBJECT,
      nullable: true,
      properties: {
        frequency: { type: SchemaType.STRING, nullable: true, enum: ["daily", "weekly", "monthly"] },
        interval: { type: SchemaType.NUMBER, nullable: true },
        days_of_week: { type: SchemaType.ARRAY, nullable: true, items: { type: SchemaType.NUMBER } },
        end_date: { type: SchemaType.STRING, nullable: true },
      },
    },
  },
};

const responseSchema = {
  type: SchemaType.OBJECT,
  properties: {
    intent: { type: SchemaType.STRING, enum: ["add", "edit", "delete", "view", "unknown"] },
    needs_clarification: { type: SchemaType.BOOLEAN },
    clarification_question: { type: SchemaType.STRING, nullable: true },
    event: eventSchema,
    target: {
      type: SchemaType.OBJECT,
      nullable: true,
      properties: {
        date_hint: { type: SchemaType.STRING, nullable: true },
        title_hint: { type: SchemaType.STRING, nullable: true },
      },
    },
  },
  required: ["intent", "needs_clarification"],
};

const SYSTEM_INSTRUCTIONS = `You turn one chat message into a structured calendar action for a shared
schedule app used by a couple: איתמר (Itamar, "itamar") and הדס (Hadas,
"hadas"). The app's UI is entirely in Hebrew and messages will usually be
written in Hebrew (occasionally English).

Respond ONLY with the JSON described by the schema. Rules:

- clarification_question, when set, MUST be written in Hebrew — the user
  only reads Hebrew in this app. title/category should stay in whatever
  language the user wrote them in (usually Hebrew) — don't translate those.

- intent: "add" for a new event, "edit" to change an existing one, "delete"
  to remove one, "view" if the message is just navigation/chit-chat with no
  event content, "unknown" if you genuinely cannot make sense of the message.
- If required info is missing or ambiguous for an add/edit/delete (e.g. no
  date can be inferred at all), set needs_clarification=true and write ONE
  short, specific follow-up question in clarification_question. Otherwise
  needs_clarification=false and clarification_question=null. (Don't worry
  about being extra-cautious here — every parsed action is shown back to
  the user for explicit confirmation before anything is saved, so only ask
  when you truly cannot proceed at all.)
- Dates are always "YYYY-MM-DD", using the CURRENT DATE given below to
  resolve relative terms ("tomorrow", "next Tuesday", "this Friday").
- You may also be given VIEWING — the date range currently shown on the
  user's screen (e.g. "month of August 2026"). If the message gives a bare
  weekday or day-of-month with no other anchor ("add dinner on the 5th",
  "add something for Friday") and that date exists within or very near
  VIEWING, prefer the occurrence inside VIEWING over the nearest one from
  CURRENT DATE — the user is almost certainly talking about what they're
  looking at. Explicit relative terms anchored to today ("tomorrow", "in
  two weeks") still resolve from CURRENT DATE regardless of VIEWING.
- start_time is 24h "HH:MM" or null if the message gives no specific time
  (e.g. a bare to-do like "buy anniversary gift" has no time).
- category is a short free-text label you infer from the message (e.g.
  "dinner", "work", "dance class", "todo") — do not force it into a fixed
  set, just pick something short and sensible.
- person: EVERY event must be tagged "itamar", "hadas", or "both" — never
  leave it null for an add/edit. Infer from the message:
  - Named explicitly ("איתמר", "הדס", "Itamar", "Hadas", or a possessive
    referring to one of them) → that person.
  - Something the couple clearly does together (dinner together, a joint
    outing, a shared errand), or no cue at all about who specifically →
    "both". This is the safe default — don't strain to guess a single
    person from a weak cue.
  - The user sees exactly who it's tagged to in the confirmation step and
    can correct it with a follow-up reply, so a reasonable guess is fine.
- recurrence: only set frequency/interval/days_of_week/end_date when the
  message clearly describes a repeating event ("every Monday", "every other
  week", "each weekday"). days_of_week uses 0=Sunday..6=Saturday. Leave the
  whole recurrence object null for one-off events — don't guess it's
  recurring from a vague hint.
- start_date is still REQUIRED for a recurring event even when the message
  gives no explicit first occurrence ("dance class every Monday at 6pm" has
  no date at all). In that case, infer start_date yourself as the next
  upcoming occurrence of that weekday from CURRENT DATE (or from VIEWING if
  it applies, per the rule above) — do NOT ask a clarifying question just
  because a recurring event lacks an explicit starting date.
- end_date on the event should equal start_date unless the message clearly
  describes a multi-day span ("Thu through Sun", "next week", "the whole
  weekend at a hotel") — spans can run many days, there's no upper limit.
- duration_minutes: infer from the message when possible (e.g. "from 9 to
  17" → 480, "for two hours" → 120, "a 12-hour shift" → 720). If the
  message gives an explicit end time earlier in the clock than start_time
  ("night shift 23:00 to 07:00"), that means it crosses midnight — compute
  duration_minutes as the full elapsed time (23:00→07:00 is 480 minutes),
  don't just subtract the raw hours. If nothing suggests a duration, leave
  it null (the app defaults untimed items and default-length events
  sensibly) rather than guessing a random length.
- For "edit"/"delete", fill 'target' with your best guess at which existing
  event the user means (date_hint + a short title_hint), so the server can
  look it up. Leave 'event' null unless the user also specified new values
  to change it to.
- Never invent a date. If you truly cannot infer one, ask via
  clarification_question instead of guessing.
- You may receive a "Conversation so far" transcript above the new reply —
  that means an earlier message in this same exchange needed clarification.
  Combine the transcript with the new reply to finalize a single event;
  don't lose details already given earlier in the transcript.`;

function getModel(systemInstruction: string, schema: object) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("Missing GEMINI_API_KEY env var.");
  const genAI = new GoogleGenerativeAI(apiKey);
  return genAI.getGenerativeModel({
    model: "gemini-3.1-flash-lite",
    systemInstruction,
    generationConfig: {
      responseMimeType: "application/json",
      responseSchema: schema as any,
    },
  });
}

export async function parseMessage(params: {
  text: string;
  currentDate: string; // YYYY-MM-DD, in the app's fixed timezone
  context?: string | null; // prior turns of an in-progress clarification exchange
  viewContext?: string | null; // human-readable description of what's on screen
}): Promise<ParsedIntent> {
  const model = getModel(SYSTEM_INSTRUCTIONS, responseSchema);

  const messageBody = params.context
    ? `Conversation so far:\n${params.context}\n\nNew reply: "${params.text}"`
    : `MESSAGE: "${params.text}"`;

  const viewingLine = params.viewContext ? `\nVIEWING: ${params.viewContext}` : "";

  const result = await model.generateContent(
    `CURRENT DATE: ${params.currentDate}${viewingLine}\n\n${messageBody}`
  );

  const raw = result.response.text();
  const parsed = JSON.parse(raw) as ParsedIntent;

  return {
    intent: parsed.intent ?? "unknown",
    needs_clarification: parsed.needs_clarification ?? false,
    clarification_question: parsed.clarification_question ?? null,
    event: parsed.event ?? null,
    target: parsed.target ?? null,
  };
}

const REVISE_SCHEMA = {
  type: SchemaType.OBJECT,
  properties: {
    confirmed: { type: SchemaType.BOOLEAN },
    cancelled: { type: SchemaType.BOOLEAN },
    updated_event: eventSchema,
  },
  required: ["confirmed", "cancelled"],
};

const REVISE_INSTRUCTIONS = `You are handling a reply to a pending calendar-event confirmation, for a
shared calendar app used by a couple: איתמר (Itamar, "itamar") and הדס
(Hadas, "hadas") — every event's person field is one of "itamar", "hadas",
"both". The app already showed the user a summary of an event (given below
as PENDING EVENT) and asked "confirm?". Now interpret their reply:

- If they're plainly agreeing ("כן", "מאשר", "בטח", "yes", "מעולה", "סבבה")
  → confirmed=true, cancelled=false, updated_event=null.
- If they're plainly declining the whole thing ("לא", "בטל", "עזוב",
  "no", "תשכח מזה") → confirmed=false, cancelled=true, updated_event=null.
- If they're correcting or adding a detail instead of a plain yes/no
  ("לא, תזיז לשמונה", "זה כל שבוע לא חד פעמי", "שיהיה ביום שלישי", "זה רק
  בשבילי", "זה של הדס לא שנינו") → confirmed=false, cancelled=false, and
  updated_event = a COMPLETE event
  object: copy every field from PENDING EVENT unchanged except the ones the
  reply corrects. Never drop fields the user didn't mention.
- Dates in updated_event follow the same "YYYY-MM-DD" / CURRENT DATE rules
  as normal parsing.

Respond ONLY with the JSON per schema.`;

export async function reviseAction(params: {
  pendingEvent: ParsedEventFields | null;
  text: string;
  currentDate: string;
}): Promise<RevisedAction> {
  const model = getModel(REVISE_INSTRUCTIONS, REVISE_SCHEMA);

  const result = await model.generateContent(
    `CURRENT DATE: ${params.currentDate}\n\nPENDING EVENT: ${JSON.stringify(params.pendingEvent)}\n\nUSER REPLY: "${params.text}"`
  );

  const raw = result.response.text();
  const parsed = JSON.parse(raw) as { confirmed?: boolean; cancelled?: boolean; updated_event?: ParsedEventFields | null };

  return {
    confirmed: parsed.confirmed ?? false,
    cancelled: parsed.cancelled ?? false,
    updatedEvent: parsed.updated_event ?? null,
  };
}
