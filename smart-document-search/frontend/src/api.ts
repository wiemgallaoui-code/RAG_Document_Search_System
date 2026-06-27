import type { AskResponse, StatsResponse } from "./types";

export async function fetchStats(): Promise<StatsResponse | null> {
  try {
    const res = await fetch("/api/stats");
    if (!res.ok) return null;
    return res.json() as Promise<StatsResponse>;
  } catch {
    return null;
  }
}

export async function askQuestion(query: string, topK = 3): Promise<AskResponse> {
  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });

  if (!response.ok) {
    const err = (await response.json().catch(() => ({}))) as { detail?: string };
    throw new Error(err.detail || `HTTP ${response.status}`);
  }

  return response.json() as Promise<AskResponse>;
}
