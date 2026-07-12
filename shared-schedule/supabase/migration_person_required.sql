-- One-time migration: run this in the Supabase SQL editor against the LIVE
-- database to backfill/enforce the new required "person" tagging
-- (itamar / hadas / both). Safe to run even if some rows already have a
-- valid person value.

-- 1. Backfill any existing null/invalid values to 'both' (the safe default).
update events
set person = 'both'
where person is null or person not in ('itamar', 'hadas', 'both');

-- 2. Make the column required with a default going forward.
alter table events alter column person set default 'both';
alter table events alter column person set not null;

-- 3. Enforce the allowed values.
alter table events
  add constraint events_person_check check (person in ('itamar', 'hadas', 'both'));
