// src/utils/fetcher.js
export async function fetchRecommendations({ apiBase, userId, k, apiKey }) {
  const url = `${apiBase.replace(/\/$/, "")}/recommend_for_me?k=${encodeURIComponent(k)}`;
  const resp = await fetch(url, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
      "x-user-id": userId,
      "x-api-key": apiKey,
    },
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }
  const data = await resp.json();
  return data;
}
