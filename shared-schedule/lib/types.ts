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
  person: string | null;
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

export type ParsedIntent = {
  intent: "add" | "edit" | "delete" | "view" | "unknown";
  needs_clarification: boolean;
  clarification_question: string | null;
  event: {
    title: string | null;
    category: string | null;
    person: string | null;
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
  } | null;
  target: {
    date_hint: string | null;
    title_hint: string | null;
  } | null;
};

export type ViewKind = "week" | "month" | "year";
