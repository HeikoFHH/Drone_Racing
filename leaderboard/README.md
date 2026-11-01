# Public leaderboard (Cloudflare)

Static UI in `public/` + Worker API in `src/worker.js`. Times stored in Workers KV.

## What Jaime must configure

1. **Cloudflare account** with Workers + Pages (or Workers + Assets). Free tier is enough.
2. **Install deps** (from this folder):

   ```bash
   cd leaderboard
   npm install
   npx wrangler login
   ```

3. **Create KV namespace** and wire it into `wrangler.toml`:

   ```bash
   npx wrangler kv namespace create RACE_TIMES
   ```

   Copy the printed `id` into `wrangler.toml`:

   ```toml
   [[kv_namespaces]]
   binding = "RACE_TIMES"
   id = "PASTE_ID_HERE"
   ```

4. **Optional write protection** — set a submit secret so random visitors cannot spam times:

   ```bash
   npx wrangler secret put SUBMIT_TOKEN
   ```

   Hackers submit with `Authorization: Bearer <token>` or `race/submit_time.py --token …`.

5. **Deploy**:

   ```bash
   npm run deploy
   ```

   Wrangler prints the `*.workers.dev` URL. Point a custom domain in the Cloudflare dashboard if you want.

6. **Tell racers** the public URL and (if set) the submit token out of band. Set:

   ```bash
   export LEADERBOARD_URL=https://<your-worker>.workers.dev
   export LEADERBOARD_TOKEN=...   # only if SUBMIT_TOKEN is set
   ```

## API

| Method | Path | Body | Notes |
|--------|------|------|-------|
| GET | `/api/times` | — | Sorted ascending by `time_s` |
| POST | `/api/submit` | `{team, time_s, drone?, notes?}` | Optional Bearer token |

No invented credentials are shipped in this repo. Create your own KV id and token.
