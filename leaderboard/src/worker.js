/**
 * Cloudflare Worker + static assets leaderboard.
 * KV binding: RACE_TIMES (JSON array of entries under key "times").
 */

const TIMES_KEY = "times";
const MAX_ENTRIES = 500;

function json(data, status = 200, extra = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "access-control-allow-origin": "*",
      "access-control-allow-headers": "content-type, authorization",
      "access-control-allow-methods": "GET, POST, OPTIONS",
      ...extra,
    },
  });
}

async function loadTimes(env) {
  if (!env.RACE_TIMES) return [];
  const raw = await env.RACE_TIMES.get(TIMES_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

async function saveTimes(env, times) {
  if (!env.RACE_TIMES) {
    throw new Error("RACE_TIMES KV binding missing — create namespace and update wrangler.toml");
  }
  await env.RACE_TIMES.put(TIMES_KEY, JSON.stringify(times.slice(0, MAX_ENTRIES)));
}

function sanitizeTeam(s) {
  return String(s || "")
    .trim()
    .slice(0, 64)
    .replace(/[<>&"']/g, "");
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return json({ ok: true });
    }

    if (url.pathname === "/api/times" && request.method === "GET") {
      const times = await loadTimes(env);
      const sorted = [...times].sort((a, b) => a.time_s - b.time_s);
      return json({
        course: env.COURSE_NAME || "two-pole-one-lap",
        count: sorted.length,
        times: sorted,
      });
    }

    if (url.pathname === "/api/submit" && request.method === "POST") {
      if (env.SUBMIT_TOKEN) {
        const auth = request.headers.get("authorization") || "";
        const token = auth.startsWith("Bearer ") ? auth.slice(7) : "";
        if (token !== env.SUBMIT_TOKEN) {
          return json({ error: "unauthorized" }, 401);
        }
      }

      let body;
      try {
        body = await request.json();
      } catch {
        return json({ error: "invalid json" }, 400);
      }

      const team = sanitizeTeam(body.team);
      const time_s = Number(body.time_s);
      if (!team || !Number.isFinite(time_s) || time_s <= 0 || time_s > 3600) {
        return json({ error: "need team and time_s in (0, 3600]" }, 400);
      }

      const entry = {
        id: crypto.randomUUID(),
        team,
        time_s: Math.round(time_s * 1000) / 1000,
        drone: sanitizeTeam(body.drone || "").slice(0, 32),
        notes: sanitizeTeam(body.notes || "").slice(0, 200),
        submitted_at: new Date().toISOString(),
      };

      const times = await loadTimes(env);
      times.push(entry);
      times.sort((a, b) => a.time_s - b.time_s);
      try {
        await saveTimes(env, times);
      } catch (e) {
        return json({ error: String(e.message || e) }, 500);
      }

      const rank = times.findIndex((t) => t.id === entry.id) + 1;
      return json({ ok: true, entry, rank, count: times.length });
    }

    if (env.ASSETS) {
      return env.ASSETS.fetch(request);
    }
    return json({ error: "not found" }, 404);
  },
};
