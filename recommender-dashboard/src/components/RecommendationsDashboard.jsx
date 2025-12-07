// RecommenderDashboard.jsx

import React, { useEffect, useState, useCallback, useMemo } from "react";

const DEFAULT_API = (import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000").replace(/\/+$/, "");

function joinUrl(base, path) {
  const b = (base || "").replace(/\/+$/, "");
  const p = (path || "").replace(/^\/+/, "");
  return `${b}/${p}`;
}

function useNow() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000 * 30);
    return () => clearInterval(id);
  }, []);
  return now;
}

function formatDate(d) {
  if (!d) return "—";
  const dt = new Date(d);
  return dt.toLocaleString();
}

function SkeletonCard() {
  return (
    <div className="animate-pulse p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex gap-4">
        <div className="w-36 h-36 bg-gray-200 rounded" />
        <div className="flex-1 space-y-3 py-1">
          <div className="h-6 bg-gray-200 rounded w-3/4" />
          <div className="h-4 bg-gray-200 rounded w-1/2" />
          <div className="h-4 bg-gray-200 rounded w-full" />
          <div className="h-4 bg-gray-200 rounded w-5/6" />
        </div>
      </div>
    </div>
  );
}

function ProductCard({ product, onGenerateImage }) {
  const [open, setOpen] = useState(false);
  const copyId = async () => {
    try {
      await navigator.clipboard.writeText(product.product_id || "");
      alert("Copied product id");
    } catch (e) {
      // fallback
    }
  };

  return (
    <article className="bg-white border rounded-lg p-4 shadow-sm flex flex-col" aria-labelledby={`title-${product.product_id}`}>
      <div className="flex">
        <div className="flex-1">
          <h3 id={`title-${product.product_id}`} className="text-xl font-semibold leading-tight">
            {product.title || "Untitled Product"}
          </h3>
          <div className="text-sm text-gray-600 mt-1">Score: {Number(product.score || 0).toFixed(3)} • Source: {product.explanation_source || "n/a"}</div>

          <p className="mt-3 text-gray-800">{product.blurb || product.title}</p>

          <div className="mt-2">
            <button
              onClick={() => setOpen((s) => !s)}
              className="inline-flex items-center gap-2 text-sm text-indigo-700 hover:underline"
              aria-expanded={open}
              aria-controls={`expl-${product.product_id}`}
            >
              {open ? "Hide explanation" : "Why this recommendation?"}
            </button>

            <button onClick={copyId} className="ml-4 inline-flex items-center gap-2 px-2 py-1 text-sm border rounded">
              Copy ID
            </button>

            {/* <button
              onClick={() => onGenerateImage && onGenerateImage(product)}
              className="ml-4 inline-flex items-center gap-2 px-2 py-1 text-sm bg-indigo-50 text-indigo-700 border rounded"
            >
              Generate Image
            </button> */}
          </div>
        </div>
      </div>

      <div className={`mt-3 ${open ? "block" : "hidden"}`} id={`expl-${product.product_id}`}>
        <p className="text-sm text-gray-700">{product.explanation || "No explanation"}</p>
      </div>

      <footer className="mt-auto pt-3 text-xs text-gray-500">Product ID: {product.product_id}</footer>
    </article>
  );
}

export default function RecommenderDashboard() {
  const now = useNow();

  // UI state
  const [userId, setUserId] = useState("U0000019");
  const [apiKey, setApiKey] = useState("dev-key");
  const [k, setK] = useState(5);

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const [updatedAt, setUpdatedAt] = useState(null);

  // controls: search, sort, pageSize
  const [q, setQ] = useState("");
  const [sortBy, setSortBy] = useState("score");
  const [pageSize, setPageSize] = useState(12);
  const [page, setPage] = useState(1);

  // simple client cache (in-memory)
  const cacheKey = `${userId}|${k}`;
  const cacheRef = useMemo(() => ({}), []);

  const apiBase = DEFAULT_API;

  const joinEndpoint = useCallback((path) => joinUrl(apiBase, path), [apiBase]);

  const fetchRecommendations = useCallback(async (opts = {}) => {
    const u = opts.userId || userId;
    const kk = opts.k || k;
    const key = `${u}::${kk}`;

    setError(null);

    // quick cache hit
    if (cacheRef[key] && Date.now() - cacheRef[key].ts < 1000 * 60 * 3) {
      setResults(cacheRef[key].data);
      setUpdatedAt(cacheRef[key].ts);
      return cacheRef[key].data;
    }

    setLoading(true);
    try {
      const url = `${joinEndpoint("recommend_for_me")}?k=${encodeURIComponent(kk)}`;
      const resp = await fetch(url, {
        method: "GET",
        headers: {
          "x-user-id": u,
          "x-api-key": apiKey,
          "Content-Type": "application/json",
        },
      });

      if (!resp.ok) {
        const txt = await resp.text().catch(() => "");
        throw new Error(`HTTP ${resp.status}: ${txt || resp.statusText}`);
      }

      const data = await resp.json();
      // some servers return {results: [...]}, some return {resp: ...}
      const list = data.results || data.recommendations || data.resp?.recommendations || data.resp || [];

      // normalize to an array of product objects
      const products = (Array.isArray(list) ? list : Object.values(list)).map((it) => {
        // if returned as {product_id, title, score, blurb, explanation}
        if (it.product_id) return it;
        // if tuple form [id, score]
        if (Array.isArray(it) && it.length >= 2) return { product_id: it[0], score: it[1] };
        return it;
      });

      setResults(products);
      setUpdatedAt(Date.now());
      cacheRef[key] = { ts: Date.now(), data: products };
      return products;
    } catch (e) {
      console.error("fetchRecommendations error", e);
      setError(e.message || String(e));
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [userId, k, apiKey, cacheRef, joinEndpoint]);

  useEffect(() => {
    // auto fetch on mount
    fetchRecommendations();
  }, []);

  const filtered = useMemo(() => {
    const ql = (q || "").trim().toLowerCase();
    let arr = results.slice();
    if (ql) arr = arr.filter((p) => (p.title || "").toLowerCase().includes(ql) || (p.blurb || "").toLowerCase().includes(ql));
    if (sortBy === "score") arr.sort((a, b) => (b.score || 0) - (a.score || 0));
    if (sortBy === "title") arr.sort((a, b) => String(a.title || "").localeCompare(String(b.title || "")));
    return arr;
  }, [results, q, sortBy]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const pageItems = filtered.slice((page - 1) * pageSize, page * pageSize);

  const onGenerateImage = useCallback(async (product) => {
    try {
      // call backend generate endpoint
      const url = `${joinEndpoint("generate_image")}/${encodeURIComponent(product.product_id)}`;
      const resp = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": apiKey,
        },
        body: JSON.stringify(product),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json = await resp.json();
      if (json.image_url) {
        // update local state (immutable update)
        setResults((r) => r.map((p) => (p.product_id === product.product_id ? { ...p, image_url: json.image_url } : p)));
      } else {
        alert("Image generation scheduled; refresh later.");
      }
    } catch (e) {
      console.error("generate image error", e);
      alert("Failed to generate image: " + (e.message || e));
    }
  }, [apiKey, joinEndpoint]);

  return (
    <main className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <header className="mb-6 flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-extrabold">Recommendations Dashboard</h1>
            <p className="text-sm text-gray-600">LLM explanations  • Live refresh</p>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center gap-2">
            <label className="flex items-center gap-2">
              <span className="text-sm">User ID</span>
              <input value={userId} onChange={(e) => setUserId(e.target.value)} className="px-2 py-1 border rounded" />
            </label>

            <label className="flex items-center gap-2">
              <span className="text-sm">k</span>
              <input type="number" value={k} onChange={(e) => setK(Number(e.target.value || 1))} className="w-20 px-2 py-1 border rounded" />
            </label>

            <label className="flex items-center gap-2">
              <span className="text-sm">API Key</span>
              <input value={apiKey} onChange={(e) => setApiKey(e.target.value)} className="px-2 py-1 border rounded w-36" />
            </label>

            <div className="flex gap-2">
              <button onClick={() => fetchRecommendations({ userId, k })} className="px-4 py-2 bg-indigo-600 text-white rounded">Refresh</button>
              <button onClick={() => { setResults([]); fetchRecommendations({ userId, k }); }} className="px-4 py-2 border rounded">Force Refresh</button>
            </div>
          </div>
        </header>

        <section className="mb-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div className="flex items-center gap-3">
            <input placeholder="Search products or blurb" value={q} onChange={(e) => setQ(e.target.value)} className="px-3 py-2 border rounded w-72" />
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value)} className="px-3 py-2 border rounded">
              <option value="score">Sort by score</option>
              <option value="title">Sort by title</option>
            </select>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-sm text-gray-600">Items: {results.length}</span>
            <label className="flex items-center gap-2">
              <span className="text-sm text-gray-600">Page size</span>
              <select value={pageSize} onChange={(e) => { setPageSize(Number(e.target.value)); setPage(1); }} className="px-2 py-1 border rounded">
                <option value={6}>6</option>
                <option value={12}>12</option>
                <option value={24}>24</option>
              </select>
            </label>
            <div className="text-sm text-gray-600">Updated: {updatedAt ? formatDate(updatedAt) : "—"}</div>
          </div>
        </section>

        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-800 rounded">Error: {error}</div>
        )}

        <section>
          {loading ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {Array.from({ length: 6 }).map((_, i) => (
                <SkeletonCard key={i} />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {pageItems.length === 0 ? (
                <div className="col-span-full p-6 bg-white border rounded text-gray-600">No results</div>
              ) : (
                pageItems.map((p) => (
                  <ProductCard key={p.product_id || p.id || Math.random()} product={p} onGenerateImage={onGenerateImage} />
                ))
              )}
            </div>
          )}
        </section>

        <footer className="mt-6 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button onClick={() => setPage(Math.max(1, page - 1))} className="px-3 py-1 border rounded">Prev</button>
            <div className="text-sm">Page {page} / {totalPages}</div>
            <button onClick={() => setPage(Math.min(totalPages, page + 1))} className="px-3 py-1 border rounded">Next</button>
          </div>

          <div className="text-sm text-gray-500">Built by your recommender • Ayush Mahant</div>
        </footer>
      </div>
    </main>
  );
}
