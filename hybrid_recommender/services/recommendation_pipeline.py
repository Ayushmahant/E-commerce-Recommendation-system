# services/recommendation_pipeline.py
"""
Orchestration pipeline for recommendation + LLM explanations.

Public API:
  - load_artifacts_once()  # warm models/artifacts used by recommender
  - run_recommendation_pipeline(user_id, k, interactions) -> {"results": [...]}

This implementation uses your existing services:
  - services.recommender.recommend_for_user
  - services.llm_explainers.generate_descriptions_and_explanations
  - (optionally) product catalog lookups from a DB — stubbed here for you to replace
"""

import logging
from typing import Optional, List, Tuple, Dict, Any

# Import your existing recommender + llm explainer
from services.recommender import recommend_for_user, load_artifacts_once as recommender_load
from services.llm_explainers import generate_descriptions_and_explanations

# Optional: product catalog fetch (replace with your DB/catalog)
# For now we expect the recommender to include basic metadata; otherwise implement fetch_product_metadata()
def fetch_product_metadata_bulk(product_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Replace this with a real DB/catalog batch fetch.
    Returns dict mapping product_id -> metadata dict.
    """
    # STUB: return empty metadata — pipeline will still work (LLM uses provided product fields if available)
    return {}

logger = logging.getLogger("recommendation_pipeline")
logging.basicConfig(level=logging.INFO)


def load_artifacts_once():
    """
    Warm up heavy artifacts in recommender/LLM clients.
    Delegates to recommender warm-up.
    """
    try:
        recommender_load()
    except Exception as e:
        logger.warning("recommender warm-up failed: %s", e)


def _canonicalize_recommender_resp(resp: Any) -> List[Dict[str, Any]]:
    """
    Small copy of the canonicalizer: converts various recommender outputs into list of product dicts.
    Keep consistent with app.py canonicalizer or import it if you prefer.
    """
    products = []
    if resp is None:
        return products

    if isinstance(resp, dict):
        for key in ("recommendations", "products", "results", "items"):
            if key in resp and isinstance(resp[key], list):
                for item in resp[key]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        products.append({"product_id": item[0], "score": float(item[1])})
                    elif isinstance(item, dict):
                        pid = item.get("product_id") or item.get("id") or item.get("productId")
                        score = item.get("score") or item.get("rating") or 0.0
                        entry = {"product_id": pid, "score": float(score)}
                        for k in ("title", "description", "category", "tags", "price", "rating_avg", "rating_count"):
                            if k in item:
                                entry[k] = item[k]
                        products.append(entry)
                return products
        if all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in resp.items()):
            for pid, score in resp.items():
                products.append({"product_id": pid, "score": float(score)})
            return products
        if resp.get("product_id"):
            pid = resp.get("product_id")
            score = resp.get("score", 0.0)
            entry = {"product_id": pid, "score": float(score)}
            for k in ("title", "description", "category", "tags", "price"):
                if k in resp:
                    entry[k] = resp[k]
            products.append(entry)
            return products

    if isinstance(resp, list):
        for item in resp:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                products.append({"product_id": item[0], "score": float(item[1])})
            elif isinstance(item, dict):
                pid = item.get("product_id") or item.get("id") or item.get("productId")
                score = item.get("score") or item.get("rating") or 0.0
                entry = {"product_id": pid, "score": float(score)}
                for k in ("title", "description", "category", "tags", "price", "rating_avg", "rating_count"):
                    if k in item:
                        entry[k] = item[k]
                products.append(entry)
        return products

    return products


def run_recommendation_pipeline(user_id: str, k: int = 5, interactions: Optional[List[Tuple[str, float, Optional[float]]]] = None) -> Dict[str, Any]:
    """
    Full pipeline:
      - call recommender
      - canonicalize output
      - fetch/merge product metadata
      - call LLM explainer (batched)
      - return {"results": [...]}
    """
    # 1) Call recommender
    try:
        resp = recommend_for_user(user_id, k=k, interactions=interactions)
    except Exception as e:
        logger.exception("Recommender failed: %s", e)
        raise

    # 2) Canonicalize to product list
    products = _canonicalize_recommender_resp(resp)
    if not products:
        logger.info("Recommender returned no products for user %s", user_id)
        return {"results": []}

    # 3) Bulk fetch product metadata if needed and merge into products list
    pids = [p["product_id"] for p in products if p.get("product_id")]
    catalog_meta = fetch_product_metadata_bulk(pids) if pids else {}

    # Merge metadata (catalog_meta wins when present)
    for p in products:
        pid = p.get("product_id")
        meta = catalog_meta.get(pid, {})
        # Only copy keys if missing from recommender output
        for k in ("title", "description", "category", "tags", "price", "rating_avg", "rating_count"):
            if k not in p and meta.get(k) is not None:
                p[k] = meta.get(k)

    # 4) Build product_catalog for LLM explainer (map pid -> metadata)
    product_catalog: Dict[str, Dict[str, Any]] = {}
    for p in products:
        pid = p.get("product_id")
        product_catalog[pid] = {
            "title": p.get("title"),
            "description": p.get("description"),
            "category": p.get("category"),
            "tags": p.get("tags"),
            "price": p.get("price"),
            "rating_avg": p.get("rating_avg"),
            "rating_count": p.get("rating_count"),
        }

    # 5) Call LLM explainer (batched). It returns (descriptions, explanations, sources)
    try:
        descriptions_map, explanations_map, sources_map = generate_descriptions_and_explanations(
            user_id=user_id,
            products=products,
            user_interactions=interactions or [],
            product_catalog=product_catalog,
        )
    except Exception as e:
        logger.exception("LLM explainer failed: %s", e)
        # graceful fallback: construct deterministic explanations
        descriptions_map = {}
        explanations_map = {}
        sources_map = {}
        for p in products:
            pid = p.get("product_id")
            descriptions_map[pid] = (p.get("description") or p.get("title") or "")[:120]
            explanations_map[pid] = f"Recommended based on product attributes in category {p.get('category') or 'N/A'}."
            sources_map[pid] = "fallback"

    # 6) Build results list
    results = []
    for p in products:
        pid = p.get("product_id")
        results.append({
            "product_id": pid,
            "title": p.get("title"),
            "score": p.get("score"),
            "blurb": descriptions_map.get(pid, (p.get("title") or "")[:120]),
            "explanation": explanations_map.get(pid, ""),
            "explanation_source": sources_map.get(pid, "fallback"),
        })

    return {"results": results}
