# services/llm_explainers.py
"""
LLM explainer module (batched, prompt-engineered) — Gemini-only.

Responsibilities:
- Given a user_id, list of product dicts, optional user interactions and product_catalog,
  produce for each product:
    - a short blurb (<= 20 words)
    - a short, realistic, behavior-driven explanation (1-4 short sentences)
- Uses Gemini (google.generativeai) in a single batched call when available; otherwise falls back
  to a deterministic generator.
- Enforces strict JSON output from the model and robustly parses it.
- Includes an in-process cache (EXPLANATION_CACHE_TTL seconds).
"""
from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta

# Optional DB helper to fetch recent interactions quickly (non-blocking with timeout)
try:
    from services.firebase_client import read_interactions_with_timeout
except Exception:
    read_interactions_with_timeout = None  # type: ignore

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_explainers")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Try to import Gemini SDK
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai  # type: ignore
    GENAI_AVAILABLE = True
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
        except Exception as e:
            logger.warning("Failed to configure google.generativeai with provided key: %s", e)
except Exception as e:
    GENAI_AVAILABLE = False
    logger.info("google.generativeai (Gemini) SDK not available. Will use deterministic fallback. %s", e)

# Cache TTL (seconds)
_EXPLANATION_CACHE: Dict[str, Tuple[Dict[str, str], datetime]] = {}
_CACHE_TTL = int(os.getenv("EXPLANATION_CACHE_TTL", "300"))


def _make_cache_key(user_id: str, product_id: str) -> str:
    return f"{user_id}||{product_id}"


def _get_from_cache(key: str) -> Optional[Dict[str, str]]:
    if _CACHE_TTL <= 0:
        return None
    rec = _EXPLANATION_CACHE.get(key)
    if not rec:
        return None
    payload, ts = rec
    if datetime.utcnow() - ts > timedelta(seconds=_CACHE_TTL):
        try:
            del _EXPLANATION_CACHE[key]
        except KeyError:
            pass
        return None
    return payload


def _save_to_cache(key: str, payload: Dict[str, str]):
    if _CACHE_TTL <= 0:
        return
    _EXPLANATION_CACHE[key] = (payload, datetime.utcnow())


# -----------------------
# JSON extractor
# -----------------------
def _extract_json(text: str) -> Any:
    """
    Extract first JSON array/object from text. Attempts a few salvage strategies.
    """
    if not text:
        raise ValueError("Empty model output")
    m = re.search(r"(\[.*?\]|\{.*?\})", text, flags=re.DOTALL)
    if not m:
        m2 = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
        if not m2:
            raise ValueError("No JSON-like structure found in model output")
        cand = m2.group(0)
    else:
        cand = m.group(0)
    try:
        return json.loads(cand)
    except Exception:
        last_obj = max(cand.rfind(']'), cand.rfind('}'))
        if last_obj != -1:
            try:
                return json.loads(cand[:last_obj + 1])
            except Exception:
                pass
    try:
        fixed = cand.replace("'", '"')
        fixed = re.sub(r",\s*([\]}])", r"\1", fixed)
        return json.loads(fixed)
    except Exception as e:
        logger.debug("JSON salvage failed. raw fragment: %s", cand[:500])
        raise ValueError(f"Failed to parse JSON from model output: {e}")


# -----------------------
# Prompt template (strict, behavior-first)
# -----------------------
BATCHED_PROMPT = r"""
System: You are an expert product analyst. For each product, answer this question clearly:
"Explain why product <product_id> is recommended to this user."

Important:
- IGNORE real user behavior unless explicitly provided.
- If user history is empty or irrelevant, EXPLAIN the recommendation based on the product’s qualities.
- You MAY assume the user has shown interest in similar products in the past, but phrase it honestly:
  e.g., "This is recommendated as you have recent browsing activity."

OUTPUT FORMAT (strict JSON):
Return ONLY a JSON ARRAY. Each object must contain:
- "product_id": string
- "blurb": string (≤ 20 words, short product summary)
- "explanation": string (2–4 short sentences)

EXPLANATION RULES:
1. Start with a sentence answering the question directly:
   - Example: "This product is recommended because you are likely interested in similar accessories."
   - If user history exists, you may reference it.
   - If history is empty, assume *possible* interest, but DO NOT say "you have no activity."
2. Use ONLY product fields: title, category, tags, price, rating_avg, rating_count, description.
3. Highlight special qualities (features, materials, rating, design, tags).
4. NEVER say: "matches your browsing patterns", "no recent activity", or fallback-like text.
5. Never invent missing product facts.

User context (may be empty):
{history_ctx}

Products:
{products_json}

Return JSON only.
"""
# Single-product retry prompt (stricter, asks only for explanation & blurb)
SINGLE_PRODUCT_PROMPT = r"""
System: You are an expert personalization analyst. For a single product produce a concise JSON object answering:
"Why is product {product_id} recommended to this user based on their behavior?"

Return a VALID JSON OBJECT with keys:
- "product_id": string
- "blurb": string (<=20 words)
- "explanation": string (1-4 short sentences; first sentence must be behavior-first)

User context:
{history_ctx}

Product:
{product_json}

Constraints: Use only provided product fields. Don't invent facts. If history_ctx is not "No strong history available", the explanation must begin with a behavior tie (e.g., "You recently viewed...").
"""


# -----------------------
# Deterministic fallback
# -----------------------
def _fallback_explanation(user_id: str, product: Dict, user_interactions: List[Tuple[str, float, Optional[float]]], product_catalog: Dict[str, Dict]) -> str:
    title = product.get("title") or product.get("description") or "This product"
    category = product.get("category")
    tags = product.get("tags") or []
    price = product.get("price")
    price_part = f"Priced at {price}." if price else ""
    parts: List[str] = []

    interacted = False
    for pid, *_ in (user_interactions or []):
        if pid == product.get("product_id"):
            parts.append("You previously interacted with this item.")
            interacted = True
            break
    if not interacted and user_interactions:
        for pid, *_ in user_interactions:
            p = product_catalog.get(pid, {})
            if p and category and p.get("category") == category:
                parts.append(f"Based on your interest in {category} items.")
                break

    if tags:
        try:
            t = tags[0] if isinstance(tags, (list, tuple)) else str(tags).split(",")[0]
            parts.append(f"It includes {t}.")
        except Exception:
            pass

    expl = " ".join(parts).strip()
    if not expl:
        expl = "Recommended because it matches your recent browsing patterns."
    sentences: List[str] = []
    sentences.append(f"{title} is recommended for you.")
    sentences.append(expl)
    if price_part:
        sentences.append(price_part)
    else:
        sentences.append("It matches products you've previously viewed or clicked.")
    return " ".join(sentences[:4])


# -----------------------
# Sanitizer helper
# -----------------------
def _sanitize_llm_output_blurb_and_expl(pid: str, blurb: str, expl: str, product: Dict) -> Tuple[str, str]:
    """
    Ensure blurb <= 20 words and explanation is 1-4 short sentences.
    If the LLM output violates constraints or is empty, use deterministic fallback.
    """
    blurb = (blurb or "").strip()
    expl = (expl or "").strip()

    # Trim blurb to 20 words
    words = blurb.split()
    if len(words) > 20:
        blurb = " ".join(words[:20]).rstrip(".,") + "..."

    # Split explanation into sentences (naive)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', expl) if s.strip()]

    if not sentences:
        fb = _fallback_explanation("user", product, [], {})
        return (blurb or (product.get("title") or "")[:60], fb)

    # Keep at most 4 sentences
    sentences = sentences[:4]

    expl = " ".join(sentences)
    if len(expl.split()) > 120:
        expl = " ".join(expl.split()[:120]).rstrip(".,") + "..."

    if not blurb:
        blurb = (product.get("title") or "")[:60]

    return blurb, expl


# -----------------------
# Consistency check
# -----------------------
def _llm_output_consistent_with_history(history_ctx: str, explanation: str) -> bool:
    """
    Return False if LLM claims no history while history_ctx indicates interactions,
    or if first sentence lacks behavior keywords when history exists.
    """
    if not history_ctx or history_ctx.strip().lower() == "no strong history available.":
        return True
    # reject explicit generic no-history sentence
    if re.search(r"this item is shown because you do not have recent activity", explanation, flags=re.I):
        return False
    # ensure first sentence has a behavior keyword
    first = explanation.split(".")[0]
    if not re.search(r'\b(you|your|recent|view|viewed|clicked|clicks|purchase|bought|added|cart|visited|interacted|favou?r)\b', first, flags=re.I):
        return False
    return True


# -----------------------
# Robust Gemini adapter
# -----------------------
def _call_gemini(prompt: str, max_tokens: int = 1500, temperature: float = 0.25) -> str:
    """
    Try multiple SDK call shapes for compatibility.
    """
    if not GENAI_AVAILABLE:
        raise RuntimeError("google.generativeai SDK not available")
    last_exc = None

    # modern SDK (preferred)
    try:
        if hasattr(genai, "GenerativeModel"):
            try:
                model = genai.GenerativeModel(GEMINI_MODEL)
                resp = model.generate_content(prompt)
                text = getattr(resp, "text", None)
                if text:
                    return text.strip()
                try:
                    return resp.output[0].content[0].text.strip()
                except Exception:
                    return str(resp)
            except Exception as e:
                last_exc = e
        else:
            last_exc = AttributeError("genai.GenerativeModel not present")
    except Exception as e:
        last_exc = e

    # older SDK shape
    try:
        if hasattr(genai, "generate"):
            try:
                resp = genai.generate(model=GEMINI_MODEL, input=prompt, max_output_tokens=max_tokens, temperature=temperature)
                try:
                    text = resp.output[0].content[0].text
                    if text:
                        return text.strip()
                except Exception:
                    pass
                try:
                    return str(resp)
                except Exception:
                    pass
            except Exception as e:
                last_exc = e
    except Exception as e:
        last_exc = e

    # chat-like
    try:
        if hasattr(genai, "chat"):
            try:
                resp = genai.chat(model=GEMINI_MODEL, messages=[{"role": "user", "content": prompt}])
                try:
                    return resp.output[0].content[0].text.strip()
                except Exception:
                    return str(resp)
            except Exception as e:
                last_exc = e
    except Exception as e:
        last_exc = e

    raise RuntimeError(f"Gemini call failed (no supported SDK method found). Last error: {last_exc}")


# -----------------------
# Single-product retry (one attempt)
# -----------------------
def _retry_single_product_llm(user_id: str, history_ctx: str, product: Dict) -> Optional[Tuple[str, str]]:
    """
    Ask LLM for a single product explanation + blurb. Returns (blurb, explanation) or None.
    """
    try:
        prod_json = json.dumps({
            "product_id": product.get("product_id", ""),
            "title": (product.get("title") or "")[:120],
            "brand": product.get("brand") or "",
            "category": product.get("category") or "",
            "tags": product.get("tags") or "",
            "price": str(product.get("price") or ""),
            "rating_avg": str(product.get("rating_avg") or ""),
            "rating_count": str(product.get("rating_count") or ""),
            "description": (product.get("description") or "")[:300],
        }, ensure_ascii=False)
        prompt = SINGLE_PRODUCT_PROMPT.format(product_id=product.get("product_id", ""), history_ctx=history_ctx, product_json=prod_json)
        raw = _call_gemini(prompt, max_tokens=500, temperature=0.2)
        # raw expected to be a JSON object
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            blurb = (parsed.get("blurb") or "").strip()
            expl = (parsed.get("explanation") or "").strip()
            return blurb, expl
        else:
            logger.debug("Single-product retry parsed non-dict: %s", type(parsed))
            return None
    except Exception as e:
        logger.debug("Single-product retry failed: %s", e)
        return None


# -----------------------
# Batched generation
# -----------------------
def generate_descriptions_and_explanations_batched(
    user_id: str,
    products: List[Dict],
    user_interactions: Optional[List[Tuple[str, float, Optional[float]]]] = None,
    product_catalog: Optional[Dict[str, Dict]] = None,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Returns: (descriptions_dict, explanations_dict, explanation_sources_dict)
    """
    descriptions: Dict[str, str] = {}
    explanations: Dict[str, str] = {}
    explanation_sources: Dict[str, str] = {}
    product_catalog = product_catalog or {}

    # Try to fetch interactions quickly if not provided
    if not user_interactions:
        try:
            if callable(read_interactions_with_timeout):
                fetched = read_interactions_with_timeout(user_id, timeout=1.0)
                if fetched:
                    user_interactions = fetched
                    logger.debug("Fetched %d interactions for user %s from DB", len(fetched), user_id)
                else:
                    user_interactions = []
                    logger.debug("No interactions returned for user %s (DB empty)", user_id)
            else:
                user_interactions = []
                logger.debug("No read_interactions_with_timeout available; continuing with empty interactions")
        except Exception as e:
            user_interactions = []
            logger.warning("Failed to fetch interactions for user %s: %s", user_id, e)

    user_interactions = user_interactions or []

    to_request: List[Dict] = []
    for p in products:
        pid = p.get("product_id")
        if not pid:
            continue
        key = _make_cache_key(user_id, pid)
        cached = _get_from_cache(key)
        if cached:
            descriptions[pid] = cached.get("blurb", "")
            explanations[pid] = cached.get("explanation", "")
            explanation_sources[pid] = "cache"
            logger.debug("Cache hit for %s", pid)
        else:
            to_request.append(p)

    if not to_request:
        return descriptions, explanations, explanation_sources

    # Build history_ctx (titles and categories)
    history_titles: List[str] = []
    history_cats: List[str] = []
    for pid, *_ in user_interactions:
        entry = product_catalog.get(pid) if product_catalog else None
        if entry:
            t = entry.get("title")
            c = entry.get("category")
            if t:
                history_titles.append(t)
            if c:
                history_cats.append(c)
    parts: List[str] = []
    if history_titles:
        parts.append("Recently interacted products: " + "; ".join(history_titles[:6]))
    if history_cats:
        parts.append("Recent categories: " + "; ".join(history_cats[:6]))
    history_ctx = " | ".join(parts) if parts else "No strong history available."

    # Prepare product JSON lines (truncate to keep prompt size reasonable)
    prod_lines: List[str] = []
    for p in to_request:
        info = {
            "product_id": p.get("product_id", ""),
            "title": (p.get("title") or "")[:120],
            "brand": p.get("brand") or "",
            "category": p.get("category") or "",
            "tags": p.get("tags") or "",
            "price": str(p.get("price") or ""),
            "rating_avg": str(p.get("rating_avg") or ""),
            "rating_count": str(p.get("rating_count") or ""),
            "description": (p.get("description") or "")[:300],
        }
        prod_lines.append(json.dumps(info, ensure_ascii=False))

    products_json = "\n".join(prod_lines)
    prompt = BATCHED_PROMPT.format(history_ctx=history_ctx, products_json=products_json)

    last_exc = None
    if GENAI_AVAILABLE and GEMINI_API_KEY and LLM_PROVIDER == "gemini":
        for attempt in range(2):
            try:
                raw = _call_gemini(prompt, max_tokens=1500, temperature=0.25)
                # logger.debug("Raw LLM output (truncated): %s", (raw or "")[:1000])
                parsed = _extract_json(raw)
                if not isinstance(parsed, list):
                    raise ValueError("Parsed Gemini output is not a list")

                # Build lookup for requested products
                prod_lookup: Dict[str, Dict] = {p.get("product_id"): p for p in to_request if p.get("product_id")}
                filled_pids = set()

                # Process LLM outputs mapped by product_id
                for obj in parsed:
                    pid = obj.get("product_id")
                    if not pid:
                        logger.warning("LLM returned an object without product_id: %s", obj)
                        continue

                    p_meta = prod_lookup.get(pid) or (product_catalog.get(pid) if product_catalog else None)
                    if not p_meta:
                        logger.warning("LLM returned explanation for unknown product_id '%s'. Recording anyway.", pid)

                    raw_blurb = (obj.get("blurb") or "").strip()
                    raw_expl = (obj.get("explanation") or "").strip()

                    # If model returned empty explanation or blurb, try single-product retry later
                    if not raw_expl:
                        logger.warning("LLM returned empty explanation for %s; will attempt retry/fallback", pid)

                    # sanitize first; this enforces blurb length and explanation sentence limits
                    try:
                        blurb, expl = _sanitize_llm_output_blurb_and_expl(pid, raw_blurb, raw_expl, p_meta or {})
                    except Exception as e:
                        logger.exception("Sanitizer failed for %s: %s", pid, e)
                        blurb, expl = None, None

                    # Validate consistency with history
                    if expl and not _llm_output_consistent_with_history(history_ctx, expl):
                        logger.info("LLM explanation inconsistent with history for %s; attempting single-product retry", pid)
                        retry = _retry_single_product_llm(user_id, history_ctx, p_meta or {})
                        if retry:
                            retry_blurb, retry_expl = retry
                            try:
                                r_blurb, r_expl = _sanitize_llm_output_blurb_and_expl(pid, retry_blurb, retry_expl, p_meta or {})
                                if _llm_output_consistent_with_history(history_ctx, r_expl):
                                    blurb, expl = r_blurb, r_expl
                                    logger.info("Single-product retry succeeded for %s", pid)
                                else:
                                    logger.info("Retry still inconsistent for %s; will fallback", pid)
                                    blurb, expl = None, None
                            except Exception:
                                blurb, expl = None, None
                        else:
                            blurb, expl = None, None

                    # If after retry we have usable blurb/expl, record; otherwise will fallback later
                    if expl:
                        # ensure blurb exists; if not use product title
                        if not blurb:
                            blurb = (p_meta or {}).get("title", "")[:60]
                        descriptions[pid] = blurb
                        explanations[pid] = expl
                        explanation_sources[pid] = "llm"
                        filled_pids.add(pid)
                        try:
                            key = _make_cache_key(user_id, pid)
                            _save_to_cache(key, {"blurb": blurb, "explanation": expl})
                        except Exception:
                            logger.exception("Failed saving to cache for %s", pid)
                        logger.info("Recorded LLM explanation for product_id=%s title=%s", pid, (p_meta or {}).get("title"))
                    else:
                        logger.info("No valid LLM explanation for %s after attempts; will fallback", pid)

                # Fill any omitted products with deterministic fallback
                for p in to_request:
                    pid = p.get("product_id")
                    if pid not in descriptions:
                        descriptions[pid] = (p.get("description") or p.get("title") or "")[:120]
                        explanations[pid] = _fallback_explanation(user_id, p, user_interactions, product_catalog)
                        explanation_sources[pid] = "fallback"
                        try:
                            key = _make_cache_key(user_id, pid)
                            _save_to_cache(key, {"blurb": descriptions[pid], "explanation": explanations[pid]})
                        except Exception:
                            logger.exception("Failed saving fallback to cache for %s", pid)
                return descriptions, explanations, explanation_sources

            except Exception as e:
                last_exc = e
                logger.warning("Gemini attempt %d failed: %s", attempt + 1, e)
        logger.warning("Gemini provider failed after attempts: %s", last_exc)

    # If Gemini not available or failed -> deterministic fallback
    logger.warning("Falling back to deterministic explanations (Gemini unavailable or failed). Last error: %s", last_exc)
    for p in to_request:
        pid = p.get("product_id")
        descriptions[pid] = (p.get("description") or p.get("title") or "")[:120]
        explanations[pid] = _fallback_explanation(user_id, p, user_interactions or [], product_catalog or {})
        explanation_sources[pid] = "fallback"
        try:
            key = _make_cache_key(user_id, pid)
            _save_to_cache(key, {"blurb": descriptions[pid], "explanation": explanations[pid]})
        except Exception:
            logger.exception("Failed saving fallback to cache for %s", pid)
    return descriptions, explanations, explanation_sources


# -----------------------
# Public API
# -----------------------
def generate_descriptions_and_explanations(
    user_id: str,
    products: List[Dict],
    user_interactions: Optional[List[Tuple[str, float, Optional[float]]]] = None,
    product_catalog: Optional[Dict[str, Dict]] = None,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    try:
        return generate_descriptions_and_explanations_batched(user_id, products, user_interactions, product_catalog)
    except Exception as e:
        logger.warning("generate_descriptions_and_explanations: batched path failed: %s", e)

    # deterministic fallback
    descriptions: Dict[str, str] = {}
    explanations: Dict[str, str] = {}
    explanation_sources: Dict[str, str] = {}
    for p in products:
        pid = p.get("product_id")
        if not pid:
            continue
        descriptions[pid] = (p.get("description") or p.get("title") or "")[:120]
        explanations[pid] = _fallback_explanation(user_id, p, user_interactions or [], product_catalog or {})
        explanation_sources[pid] = "fallback"
    return descriptions, explanations, explanation_sources
