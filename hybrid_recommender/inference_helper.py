# inference_helper.py
import os
import json
import pickle
import sys
import re
from typing import Optional, Tuple, List

import numpy as np
from scipy.sparse import load_npz, csr_matrix

# DATA_PATH is expected to be something like "/app/Data" inside container or "./Data" on host
DATA_PATH = os.environ.get("DATA_PATH", "Data")


def _debug(msg: str):
    # print to stdout so docker logs show it
    print(f"[inference_helper] {msg}", file=sys.stdout, flush=True)


def _try_paths(base: str, filename: str):
    """
    Generate a list of candidate absolute paths to look for filename.
    base: DATA_PATH default (may be relative)
    """
    candidates = []
    # direct join with base
    candidates.append(os.path.join(base, filename))
    # also try base's parent (in case file is in project root)
    parent = os.path.dirname(base.rstrip("/\\"))
    candidates.append(os.path.join(parent, filename))
    # also try base without "/Data" (i.e., project root)
    if base.endswith(("Data", "data")):
        root = base[: -len("Data")]
        candidates.append(os.path.join(root, filename))
    # also try just filename in cwd
    candidates.append(os.path.abspath(filename))
    # remove duplicates and non-sensical empties
    seen = []
    out = []
    for p in candidates:
        if not p:
            continue
        pabs = os.path.abspath(p)
        if pabs not in seen:
            seen.append(pabs)
            out.append(pabs)
    return out


def _locate_file(filename: str) -> Optional[str]:
    """
    Return first existing path or None. Also prints debug info.
    """
    tried = _try_paths(DATA_PATH, filename)
    _debug(f"Looking for {filename}. Tried paths (in order):")
    for p in tried:
        _debug(f"  - {p}")
        if os.path.exists(p):
            _debug(f"Found {filename} at: {p}")
            return p
    _debug(f"{filename} NOT found in any candidate path.")
    return None


def load_artifacts() -> Tuple:
    """
    Load model + indices + optional data, robustly searching likely paths.
    Returns: (als_model, user_index, item_index, products_df_or_None, user_item_matrix_or_None, model_n)
    """
    # 1) ALS model
    model_fname = "als_model.pkl"
    model_path = _locate_file(model_fname)
    if model_path is None:
        raise FileNotFoundError(
            f"Could not locate {model_fname}. Please ensure it exists in one of the expected locations. "
            f"Set DATA_PATH env var to point to the containing folder (e.g. /app/Data)."
        )

    _debug(f"Loading ALS model from: {model_path}")
    with open(model_path, "rb") as f:
        als = pickle.load(f)

    # 2) user_index.json
    user_index_path = _locate_file("user_index.json")
    if user_index_path is None:
        raise FileNotFoundError("user_index.json not found. Please place it alongside als_model.pkl or in project root.")
    with open(user_index_path, "r") as f:
        user_index = json.load(f)

    # 3) item_index.json
    item_index_path = _locate_file("item_index.json")
    if item_index_path is None:
        raise FileNotFoundError("item_index.json not found. Please place it alongside als_model.pkl or in project root.")
    with open(item_index_path, "r") as f:
        item_index = json.load(f)

    # 4) try load user_item_matrix.npz (optional but preferred)
    uim = None
    uim_path = _locate_file("user_item_matrix.npz")
    if uim_path:
        try:
            _debug(f"Loading user_item_matrix from: {uim_path}")
            uim = load_npz(uim_path)
        except Exception as e:
            _debug(f"Failed to load user_item_matrix.npz: {e}. Continuing without it.")

    # 5) products_preprocessed.csv (optional)
    products = None
    prod_path = _locate_file("products_preprocessed.csv")
    if prod_path:
        try:
            import pandas as pd

            _debug(f"Loading products CSV from: {prod_path}")
            products = pd.read_csv(prod_path)
            # normalize product_id column if present
            if "product_id" in products.columns:
                products["product_id"] = products["product_id"].astype(str).str.strip()
        except Exception as e:
            _debug(f"Failed to load products_preprocessed.csv: {e}. Titles lookup will be unavailable.")

    # infer model item count if available
    model_n = None
    if hasattr(als, "item_factors"):
        try:
            model_n = int(np.asarray(als.item_factors).shape[0])
            _debug(f"Detected model item count from als.item_factors: {model_n}")
        except Exception:
            _debug("als.item_factors exists but could not determine shape.")

    _debug("Artifacts loaded successfully.")
    return als, user_index, item_index, products, uim, model_n


# light wrapper to load once
_LOADED = None


def load_once():
    global _LOADED
    if _LOADED is None:
        _LOADED = load_artifacts()
    return _LOADED


# small helper mapping function used by app
def try_int(x):
    try:
        return int(x)
    except Exception:
        return x


# minimal recommend-only helpers 
def build_user_items_for_model(uim, uid_internal, als, item_index, products):
    if not hasattr(als, "item_factors"):
        if uim is None:
            return csr_matrix((1, 0))
        return uim[uid_internal]

    model_n = int(np.asarray(als.item_factors).shape[0])

    if uim is not None and uim.shape[1] >= model_n:
        try:
            return uim[uid_internal, :model_n]
        except Exception:
            pass

    # fallback: zero row
    return csr_matrix((1, model_n))


def call_als_recommend(als, uid_internal, user_items_row, N):
    # try the recommended signature
    try:
        out = als.recommend(uid_internal, user_items_row, N, filter_already_liked_items=False, recalculate_user=False)
    except TypeError:
        # fallback simpler signature
        out = als.recommend(uid_internal, user_items_row, N)
    # parse outputs
    if isinstance(out, tuple) and len(out) == 2:
        ids, scores = out
        return np.asarray(ids).astype(int), np.asarray(scores).astype(float)
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], (list, tuple)):
        ids = [int(x[0]) for x in out]
        scores = [float(x[1]) for x in out]
        return np.asarray(ids).astype(int), np.asarray(scores).astype(float)
    if isinstance(out, np.ndarray):
        ids = out.astype(int)
        return ids, np.zeros_like(ids, dtype=float)
    raise RuntimeError("als.recommend returned an unexpected format")


# BEGIN ADDITION: subset scoring helpers
def _build_subset_internal_indices(products_df, item_index) -> List[int]:
    """
    Given products_df (from products_preprocessed.csv) and item_index (product_id -> internal),
    returns a list of internal indices (integers) that exist in the model.
    """
    if products_df is None or item_index is None:
        return []

    # normalize keys to strings for robust matching
    item_index_map = {str(k).strip(): int(v) for k, v in item_index.items()}

    # get unique product ids from CSV (normalized)
    try:
        csv_pids = products_df["product_id"].astype(str).str.strip().unique().tolist()
    except Exception:
        csv_pids = []
    subset = [item_index_map[pid] for pid in csv_pids if pid in item_index_map]
    # dedupe & sort
    subset = sorted(set(subset))
    return subset


def _score_user_over_subset(als_model, uid_internal, subset_internal, top_k):
    """
    Score user over the item subset using model factors.
    Returns (ids_list, scores_list) where ids are internal indices in model-space.
    Falls back to raising RuntimeError if model doesn't expose usable factors.
    """
    if not subset_internal:
        return [], []

    # obtain user vector
    user_vec = None
    if hasattr(als_model, "user_factors"):
        user_vec = np.asarray(als_model.user_factors)[int(uid_internal)]
    elif hasattr(als_model, "_user_factor"):
        # some library expose private helper
        user_vec = np.asarray(als_model._user_factor(int(uid_internal)))
    else:
        raise RuntimeError("No accessible user_factors/_user_factor on model")

    # obtain item matrix for subset
    if not hasattr(als_model, "item_factors"):
        raise RuntimeError("No accessible item_factors on model")
    item_mat = np.asarray(als_model.item_factors)[subset_internal]  # shape (m, f)

    # compute dot products
    scores = item_mat.dot(user_vec)  # (m,)
    # get top-k indices relative to subset_internal
    if len(scores) == 0:
        return [], []
    top_idx = np.argsort(-scores)[:top_k]
    top_internal = [int(subset_internal[i]) for i in top_idx]
    top_scores = [float(scores[i]) for i in top_idx]
    return top_internal, top_scores
#  END ADDITION


# Robust title lookup with fallback placeholder
def find_title_for_pid(products_df, pid):
    """
    Robust lookup for product title given pid.
    Returns title (str) or a placeholder if not found.
    """
    if products_df is None:
        return f"Product {pid}"
    pid_s = str(pid).strip()

    # 1) exact string match
    try:
        m = products_df[products_df["product_id"].astype(str).str.strip() == pid_s]
        if not m.empty:
            return m.iloc[0].get("title") or f"Product {pid}"
    except Exception:
        pass

    # 2) numeric match
    if re.fullmatch(r"\d+", pid_s):
        try:
            pid_i = int(pid_s)
            m2 = products_df[products_df["product_id"].astype(str).str.isdigit() & (products_df["product_id"].astype(int) == pid_i)]
            if not m2.empty:
                return m2.iloc[0].get("title") or f"Product {pid}"
        except Exception:
            pass

    # 3) strip common non-digit prefixes (like 'P' or 'SKU-') and leading zeros
    stripped = re.sub(r"^[^\d]*", "", pid_s).lstrip("0")
    if stripped:
        try:
            m3 = products_df[products_df["product_id"].astype(str).str.strip().str.endswith(stripped)]
            if not m3.empty:
                return m3.iloc[0].get("title") or f"Product {pid}"
        except Exception:
            pass

    # 4) last-resort substring numeric match
    digits = "".join(ch for ch in pid_s if ch.isdigit())
    if digits:
        try:
            m4 = products_df[products_df["product_id"].astype(str).str.contains(digits, na=False)]
            if not m4.empty:
                return m4.iloc[0].get("title") or f"Product {pid}"
        except Exception:
            pass

    # fallback placeholder
    return f"Product {pid}"


def get_recommendations(user_id: str, k: int = 5, loaded=None):
    if loaded is None:
        als, user_index, item_index, products, uim, model_n = load_once()
    else:
        als, user_index, item_index, products, uim, model_n = loaded

    # resolve user internal id
    if str(user_id) in user_index:
        uid = user_index[str(user_id)]
    elif user_id in user_index:
        uid = user_index[user_id]
    else:
        raise KeyError("user_id not found")

    # build user's item row (sparse) if available
    user_items_row = build_user_items_for_model(uim, uid, als, item_index, products)

    # Try subset scoring: restrict candidates to those present in products CSV
    subset_internal = _build_subset_internal_indices(products, item_index)

    ids, scores = [], []
    if subset_internal:
        _debug(f"Subset has {len(subset_internal)} items. Using subset scoring for top-{k}.")
        try:
            ids, scores = _score_user_over_subset(als, uid, subset_internal, k)
        except Exception as e:
            _debug(f"Subset scoring failed ({e}), falling back to model.recommend().")
            ids, scores = call_als_recommend(als, uid, user_items_row, N=k)
    else:
        # no overlap -> fallback to normal recommend
        _debug("No overlap between products CSV and model item_index; using model.recommend()")
        ids, scores = call_als_recommend(als, uid, user_items_row, N=k)

    # build reverse index (internal -> product_id). Keep original product id types if possible
    rev_item_index = {int(v): k for k, v in item_index.items()}

    out = []
    for iid, sc in zip(ids, scores):
        pid = rev_item_index.get(int(iid), int(iid))
        title = find_title_for_pid(products, pid)
        out.append({"internal_idx": int(iid), "product_id": pid, "title": title, "score": float(sc)})
    return out
