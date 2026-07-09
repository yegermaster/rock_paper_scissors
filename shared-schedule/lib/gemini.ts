import { GoogleGenerativeAI, SchemaType } from "@google/generative-ai";
import type { ParsedIntent } from "./types";

const responseSchema = {
  type: SchemaType.OBJECT,
  properties: {
    intent: {
      type: SchemaType.STRING,
      enum: ["add", "edit", "delete", "view", "unknown"],
    },
    needs_clarification: { type: SchemaType.BOOLEAN },
    clarification_question: { type: SchemaType.STRING, nullable: true },
    event: {
      type: SchemaType.OBJECT,
      nullable: true,
      properties: {
        title: { type: SchemaType.STRING, nullable: true },
        category: { type: SchemaType.STRING, nullable: true },
        person: { type: SchemaType.STRING, nullable: true },
        start_date: { type: SchemaType.STRING, nullable: true },
        end_date: { type: SchemaType.STRING, nullable: true },
        start_time: { type: SchemaType.STRING, nullable: true },
        duration_minutes: { type: SchemaType.NUMBER, nullable: true },
        recurrence: {
          type: SchemaType.OBJECT,
          nullable: true,
          properties: {
            frequency: {
              type: SchemaType.STRING,
              nullable: true,
              enum: ["daily", "weekly", "monthly"],
            },
            interval: { type: SchemaType.NUMBER, nullable: true },
            days_of_week: {
              type: SchemaType.ARRAY,
              nullable: true,
              items: { type: SchemaType.NUMBER },
            },
            end_date: { type: SchemaType.STRING, nullable: true },
          },
        },
      },
    },
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
schedule app used by two people (a couple).

Respond ONLY with the JSON described by the schema. Rules:

- intent: "add" for a new event, "edit" to change an existing one, "delete"
  to remove one, "view" if the message is just navigation/chit-chat with no
  event content, "unknown" if you genuinely cannot make sense of the message.
- If required info is missing or ambiguous for an add/edit/delete (e.g. no
  date can be inferred at all), set needs_clarification=true and write ONE
  short, specific follow-up question in clarification_question. Otherwise
  needs_clarification=false and clarification_question=null.
- Dates are always "YYYY-MM-DD", using the CURRENT DATE given below to
  resolve relative terms ("tomorrow", "next Tuesday", "this Friday").
- start_time is 24h "HH:MM" or null if the message gives no specific time
  (e.g. a bare to-do like "buy anniversary gift" has no time).
- category is a short free-text label you infer from the message (e.g.
  "dinner", "work", "dance class", "todo") — do not force it into a fixed
  set, just pick something short and sensible.
- person: only set this if the message itself implies ownership (e.g. "my
  shift", "her appointment") — otherwise leave null (shared/unowned).
- recurrence: only set frequency/interval/days_of_week/end_date when the
  message clearly describes a repeating event ("every Monday", "every other
  week", "each weekday"). days_of_week uses 0=Sunday..6=Saturday. Leave the
  whole recurrence object null for one-off events.
- end_date on the event should equal start_date unless the message clearly
  describes a multi-day span ("Thu through Sun", "next week").
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

export async function parseMessage(params: {
  text: string;
  currentDate: string; // YYYY-MM-DD, in the app's fixed timezone
  context?: string | null; // prior turns of an in-progress clarification exchange
}): Promise<ParsedIntent> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("Missing GEMINI_API_KEY env var.");

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: "gemini-1.5-flash",
    systemInstruction: SYSTEM_INSTRUCTIONS,
    generationConfig: {
      responseMimeType: "application/json",
      responseSchema: responseSchema as any,
    },
  });

  const messageBody = params.context
    ? `Conversation so far:\n${params.context}\n\nNew reply: "${params.text}"`
    : `MESSAGE: "${params.text}"`;

  const result = await model.generateContent(
    `CURRENT DATE: ${params.currentDate}\n\n${messageBody}`
  );

  const raw = result.response.text();
  const parsed = JSON.parse(raw) as ParsedIntent;

  // Defensive defaults in case the model omits optional fields.
  return {
    intent: parsed.intent ?? "unknown",
    needs_clarification: parsed.needs_clarification ?? false,
    clarification_question: parsed.clarification_question ?? null,
    event: parsed.event ?? null,
    target: parsed.target ?? null,
  };
}
