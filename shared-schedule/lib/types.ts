import type { Person } from "./people";

export type Recurrence = {
  frequency: "daily" | "weekly" | "monthly" | null;
  interval: number | null;
  daysOfWeek: number[] | null; // 0=Sunday..6=Saturday
  endDate: string | null; // YYYY-MM-DD
};

export type EventRecord = {
  id: string;
  title: string;
  category: string;
  person: Person;
  start_date: string; // YYYY-MM-DD
  end_date: string; // YYYY-MM-DD
  start_time: string | null; // HH:MM:SS or null (untimed)
  duration_minutes: number;
  recurrence_frequency: Recurrence["frequency"];
  recurrence_interval: number | null;
  recurrence_days_of_week: number[] | null;
  recurrence_end_date: string | null;
  created_at: string;
};

// A concrete rendered occurrence of an event on a specific day
// (recurring events expand into one Occurrence per matching date).
export type Occurrence = {
  event: EventRecord;
  date: string; // YYYY-MM-DD, the specific day this occurrence falls on
  isMultiDaySpan: boolean;
  spanStart: string;
  spanEnd: string;
};

export type ParsedEventFields = {
  title: string | null;
  category: string | null;
  person: Person | null; // null only mid-parse; always normalized before saving
  start_date: string | null;
  end_date: string | null;
  start_time: string | null;
  duration_minutes: number | null;
  recurrence: {
    frequency: "daily" | "weekly" | "monthly" | null;
    interval: number | null;
    days_of_week: number[] | null;
    end_date: string | null;
  } | null;
};

export type ParsedIntent = {
  intent: "add" | "edit" | "delete" | "view" | "unknown";
  needs_clarification: boolean;
  clarification_question: string | null;
  event: ParsedEventFields | null;
  target: {
    date_hint: string | null;
    title_hint: string | null;
  } | null;
};

// A fully-parsed action (event fields resolved, target found if edit/delete)
// waiting on the user's explicit yes/no/correction before it's committed to
// the database. Round-tripped opaquely through the client between turns.
export type PendingAction = {
  intent: "add" | "edit" | "delete";
  event: ParsedEventFields | null; // for add/edit
  targetEventId: string | null; // resolved concrete event id, for edit/delete
  targetTitle: string | null; // resolved event's current title, for confirmation copy
  targetPerson: Person | null; // resolved event's current person tag, for confirmation copy
  targetIsRecurring: boolean; // for delete confirmation copy
};

// Result of asking Gemini to interpret a reply to a pending confirmation.
export type RevisedAction = {
  confirmed: boolean;
  cancelled: boolean;
  // Present when the user corrected details rather than plainly
  // confirming/cancelling — replaces pendingAction.event before re-asking.
  updatedEvent: ParsedEventFields | null;
};

export type ViewKind = "week" | "month" | "year";

// A single row in the manual event-list panel — one per occurrence in the
// current view's date range (recurring events appear once per occurrence,
// same as they're drawn in the rendered image).
export type EventListItem = {
  id: string;
  occurrenceDate: string;
  title: string;
  category: string;
  person: Person;
  startTime: string | null; // "HH:MM" or null
  durationMinutes: number;
  isRecurring: boolean;
  recurrenceFrequency: "daily" | "weekly" | "monthly" | null;
  recurrenceInterval: number | null;
  recurrenceDaysOfWeek: number[] | null;
  recurrenceEndDate: string | null;
  isMultiDaySpan: boolean;
  spanStart: string;
  spanEnd: string;
};
