# services/recommender.py
import numpy as np
from typing import List, Dict, Optional, Tuple
from inference_helper import load_once, get_recommendations, _build_subset_internal_indices, find_title_for_pid

# Module-level cache of artifacts (loaded once)
_ARTIFACTS = {
    "als": None,
    "user_index": None,
    "item_index": None,
    "products": None,
    "uim": None,
    "model_n": None,
    "subset_internal": None,
}

def load_artifacts_once():
    """
    Load and cache artifacts (idempotent).
    """
    if _ARTIFACTS["als"] is None:
        als, user_index, item_index, products, uim, model_n = load_once()
        _ARTIFACTS.update({
            "als": als,
            "user_index": user_index,
            "item_index": item_index,
            "products": products,
            "uim": uim,
            "model_n": model_n,
            "subset_internal": _build_subset_internal_indices(products, item_index) or []
        })
    return _ARTIFACTS

def _build_user_vector_from_interactions(interactions: Dict[str, float], item_index: Dict[str, int], als) -> Optional[np.ndarray]:
    """
    Weighted sum of item_factors -> normalized user vector.
    """
    if not interactions:
        return None
    item_map = {str(k).strip(): int(v) for k, v in item_index.items()}
    rows, weights = [], []
    for pid, w in interactions.items():
        pid_s = str(pid).strip()
        if pid_s in item_map:
            idx = item_map[pid_s]
            try:
                rows.append(np.asarray(als.item_factors)[idx])
                weights.append(float(w))
            except Exception:
                continue
    if not rows:
        return None
    rows = np.vstack(rows)
    weights = np.asarray(weights).reshape(-1,1)
    weighted = (rows * weights).sum(axis=0)
    user_vec = weighted / (np.linalg.norm(weighted) + 1e-9)
    return user_vec

def recommend_from_interactions(interactions: Dict[str,float], k:int=5) -> List[Dict]:
    """
    Score the catalog subset using a user vector built from interactions.
    Returns list of {internal_idx, product_id, title, score}.
    """
    art = load_artifacts_once()
    als = art["als"]
    item_index = art["item_index"]
    products = art["products"]
    subset_internal = art["subset_internal"] or []
    if als is None or not subset_internal:
        return []
    user_vec = _build_user_vector_from_interactions(interactions, item_index, als)
    if user_vec is None:
        return []
    item_mat = np.asarray(als.item_factors)[subset_internal]
    scores = item_mat.dot(user_vec)
    top_idx = np.argsort(-scores)[:k]
    rev_item_index = {int(v): k for k, v in item_index.items()}
    out = []
    for i in top_idx:
        iid = int(subset_internal[i])
        pid = rev_item_index.get(iid, iid)
        title = find_title_for_pid(products, pid)
        out.append({"internal_idx": iid, "product_id": pid, "title": title, "score": float(scores[i])})
    return out

def recommend_for_user(user_id: str, k:int=5, interactions:Optional[Dict[str,float]]=None) -> Dict:
    """
    Top-level recommend function:
      - Ignores live interactions here (keeps signature for compatibility).
      - If user in user_index -> use ALS .recommend().
      - Else fallback to popularity (first k from products).
    Returns dict: {"source": "...", "recommendations": [...]}
    """
    art = load_artifacts_once()
    user_index = art["user_index"]
    products = art["products"]

    # NOTE: live interactions branch removed intentionally.
    # We now prefer the trained ALS model for known users, otherwise popularity.

    # 1) if user seen during training -> ALS .recommend()
    if user_id is not None and str(user_id) in (user_index or {}):
        try:
            recs = get_recommendations(user_id, k=k)
            return {"source":"als_model", "recommendations": recs}
        except Exception:
            pass

    # 2) fallback popularity / product order
    pop = []
    if products is not None and "product_id" in products.columns:
        pop = products["product_id"].astype(str).tolist()[:k]
    recs = []
    for pid in pop:
        title = find_title_for_pid(products, pid)
        recs.append({"internal_idx": None, "product_id": pid, "title": title, "score": None})
    return {"source":"popularity_fallback", "recommendations": recs}
