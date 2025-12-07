# services/firebase_client.py
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import os
import logging

# your existing firebase helper functions (init_firestore, read_user_interactions, update_user_interaction_increment)
from firebase_helper import init_firestore, read_user_interactions, update_user_interaction_increment

logger = logging.getLogger("firebase_client")
_executor = ThreadPoolExecutor(max_workers=4)


def ensure_firestore() -> bool:
    """
    Initialize firebase client (idempotent). Returns True if initialized, False otherwise.
    Delegates to your firebase_helper.init_firestore().
    """
    try:
        initialized = init_firestore()
        if not initialized:
            logger.info("ensure_firestore: firebase_helper reported not initialized.")
        return bool(initialized)
    except Exception as e:
        logger.exception("ensure_firestore: failed to initialize Firestore via firebase_helper")
        return False


def read_interactions_with_timeout(user_id: str, timeout: float = 1.0) -> Dict[str, float]:
    """
    Read user interactions but don't block longer than `timeout` seconds.
    Returns a mapping product_id -> rating/weight. On timeout/error returns {}.

    Uses ThreadPoolExecutor to avoid blocking FastAPI worker threads.
    """
    if not user_id:
        return {}
    try:
        fut = _executor.submit(read_user_interactions, user_id)
        result = fut.result(timeout=timeout)
        # Expecting firebase_helper.read_user_interactions to return a dict-like mapping
        return result or {}
    except TimeoutError:
        logger.debug("read_interactions_with_timeout: timeout reached for user_id=%s", user_id)
        return {}
    except Exception:
        logger.exception("read_interactions_with_timeout: error reading interactions for user_id=%s", user_id)
        return {}


def increment_interaction(user_id: str, product_id: str, delta: float = 1.0, collection: str = "user_interactions"):
    """
    Atomic increment helper — uses your existing firebase_helper.update_user_interaction_increment.
    Returns whatever the helper returns (or None on failure).
    """
    if not user_id or not product_id:
        raise ValueError("user_id and product_id must be provided")
    try:
        return update_user_interaction_increment(user_id, product_id, delta, collection=collection)
    except Exception:
        logger.exception(
            "increment_interaction: failed to increment interaction for user_id=%s product_id=%s",
            user_id,
            product_id,
        )
        return None


def fetch_product_catalog_from_firestore() -> Dict[str, dict]:
    """
    Read the entire 'products' collection from Firestore and return a mapping:
       product_id -> product_dict (includes a 'product_id' key)

    This function tries to use firebase-admin's Firestore client if available and initialized.
    If Firestore is not initialized or an error occurs it returns an empty dict.
    NOTE: For very large catalogs consider pagination or caching instead of streaming the whole collection.
    """
    # Ensure Firestore init was attempted (idempotent). If your init is done on startup, this is a no-op.
    try:
        ensure_firestore()
    except Exception:
        logger.debug("fetch_product_catalog_from_firestore: ensure_firestore raised an exception (continuing)")

    try:
        # Import firebase_admin.firestore lazily to avoid hard dependency if not installed
        import firebase_admin  # noqa: F401 (we only need it to ensure installed)
        from firebase_admin import firestore  # type: ignore
    except Exception:
        logger.info("fetch_product_catalog_from_firestore: firebase-admin not available; returning empty catalog")
        return {}

    try:
        db = firestore.client()
        docs = db.collection("products").stream()
        out: Dict[str, dict] = {}
        for d in docs:
            data = d.to_dict() or {}
            # Ensure product_id is present and consistent with doc id
            data["product_id"] = d.id
            out[d.id] = data
        return out
    except Exception:
        logger.exception("fetch_product_catalog_from_firestore: failed to fetch products collection")
        return {}
