# firebase_helper.py
import os
import json
from typing import Dict, Optional
from datetime import datetime, timezone

# pip install firebase-admin
import firebase_admin
from firebase_admin import credentials, firestore

_FIRESTORE_CLIENT = None

def init_firestore(service_account_path: Optional[str] = None):
    """
    Initialize and return a Firestore client.
    Uses GOOGLE_APPLICATION_CREDENTIALS env var if service_account_path not provided.
    Safe to call multiple times.
    """
    global _FIRESTORE_CLIENT
    if _FIRESTORE_CLIENT is not None:
        return _FIRESTORE_CLIENT

    if service_account_path is None:
        service_account_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    if service_account_path is None or not os.path.exists(service_account_path):
        raise FileNotFoundError("Service account JSON not found. Set GOOGLE_APPLICATION_CREDENTIALS or pass path.")
    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(cred)
    _FIRESTORE_CLIENT = firestore.client()
    return _FIRESTORE_CLIENT

def write_user_interactions(user_id: str, interactions: Dict[str, float], collection: str = "user_interactions"):
    """
    Write (replace) a user's interactions document.
    interactions: dict product_id -> weight/int
    """
    client = init_firestore()
    doc_ref = client.collection(collection).document(user_id)
    payload = {
        "interactions": interactions,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    doc_ref.set(payload)
    return True

def update_user_interaction_increment(user_id: str, product_id: str, delta: float = 1.0, collection: str = "user_interactions"):
    """
    Increment a product count in user's interactions map atomically using a transaction.
    """
    client = init_firestore()
    doc_ref = client.collection(collection).document(user_id)

    def _txn_update(txn, ref):
        snapshot = ref.get(transaction=txn)
        if snapshot.exists:
            data = snapshot.to_dict()
            interactions = data.get("interactions", {}) or {}
        else:
            interactions = {}
        # update
        cur = float(interactions.get(product_id, 0.0))
        interactions[product_id] = cur + float(delta)
        txn.set(ref, {"interactions": interactions, "last_updated": datetime.now(timezone.utc).isoformat()})

    client.run_transaction(lambda txn: _txn_update(txn, doc_ref))
    return True

def read_user_interactions(user_id: str, collection: str = "user_interactions") -> Dict[str, float]:
    """
    Fetch the user's interactions as dict product_id -> float.
    """
    client = init_firestore()
    doc_ref = client.collection(collection).document(user_id)
    doc = doc_ref.get()
    if not doc.exists:
        return {}
    data = doc.to_dict()
    interactions = data.get("interactions") or {}
    # ensure numeric
    return {str(k): float(v) for k, v in interactions.items()}
