-- Run this once in the Supabase SQL editor for your project.

create extension if not exists "pgcrypto";

create table if not exists events (
  id uuid primary key default gen_random_uuid(),

  title text not null,
  category text not null,       -- open-ended, free text (e.g. "dinner", "work", "gym")
  person text not null default 'both'
    check (person in ('itamar', 'hadas', 'both')),  -- who the event is tagged to

  start_date date not null,
  end_date date not null,       -- multi-day events: end_date > start_date
  start_time time,              -- null = untimed (todo); rendered at end-of-day
  duration_minutes int not null default 60,

  recurrence_frequency text,        -- 'daily' | 'weekly' | 'monthly' | null
  recurrence_interval int default 1,
  recurrence_days_of_week int[],    -- 0=Sunday .. 6=Saturday, used when frequency='weekly'
  recurrence_end_date date,         -- null = repeats indefinitely (capped at render time)

  created_at timestamptz not null default now()
);

create index if not exists events_date_range_idx on events (start_date, end_date);
create index if not exists events_recurring_idx on events (recurrence_frequency) where recurrence_frequency is not null;

-- No RLS policies: the app talks to Supabase using the service-role key from
-- server-side routes only (never exposed to the browser), and access to the
-- app itself is gated by the SHARED_SECRET link, not Supabase auth.
alter table events enable row level security;
-- Intentionally no policies added -> only the service-role key (which bypasses RLS) can read/write.
