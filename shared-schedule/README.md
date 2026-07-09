# Shared Schedule

A shared visual calendar for two people. Add events by typing plain
English into a single input box; it renders as an actual PNG image with
week / month / year views. No login — one shared secret link.

Total cost: **$0** (Vercel free tier, Supabase free tier, Gemini free tier).

## How it works

1. You type a message ("dinner Friday at 7", "dance class every Monday 6pm",
   "cancel Thursday's shift").
2. The server sends it to Gemini with a strict JSON schema (title, date,
   time, category, recurrence, etc.) — Gemini never free-writes a reply,
   it only ever returns structured data.
3. If something's missing or ambiguous, the app asks a follow-up question
   instead of guessing.
4. Events are stored in Supabase (Postgres).
5. Each view (week/month/year) is rendered server-side as a PNG on request
   (via `next/og`) from whatever's currently in the database — there's no
   separate "build the calendar" step, the image is always live.

## One-time setup

### 1. Gemini (parsing)

Get a free API key at https://aistudio.google.com/apikey — no card required.

### 2. Supabase (database)

1. Create a free project at https://supabase.com.
2. Open the SQL editor and run `supabase/schema.sql` from this repo.
3. Grab your Project URL and **service_role** key from Project Settings → API.
   (Use the service_role key, not the anon key — there's no per-user auth
   here, the app relies on the shared link + RLS being locked down instead.)

### 3. Shared secret

Generate a random token, e.g.:

```
openssl rand -hex 24
```

This becomes your private link: `https://<your-deploy>.vercel.app/?key=<token>`.
Whoever opens that link once gets a cookie and won't need the `?key=` again
on that device.

### 4. Deploy to Vercel (free tier)

1. Push this repo to GitHub (already done if you're reading this from the repo).
2. Import it at https://vercel.com/new.
3. Add the environment variables from `.env.example`:
   - `GEMINI_API_KEY`
   - `NEXT_PUBLIC_SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `SHARED_SECRET`
   - `APP_TIMEZONE` (an IANA name, e.g. `America/New_York`)
4. Deploy. Visit `https://<your-deploy>.vercel.app/?key=<your SHARED_SECRET>` once each.

### Local development

```
npm install
cp .env.example .env.local   # fill in the values above
npm run dev
```

Visit `http://localhost:3000/?key=<your SHARED_SECRET>`.

## Known limitations (by design, for v1)

- **Deleting a recurring event cancels the whole series** — there's no
  per-occurrence exception table yet, so "cancel this Monday's class" removes
  every future Monday, not just one.
- **Single fixed timezone** — set once via `APP_TIMEZONE`, no per-viewer
  conversion.
- **No login** — anyone with the link has full read/write access.
- Overlap layout in week view uses a simple greedy column algorithm, not a
  fully optimal packing — fine at the event density this app expects.
