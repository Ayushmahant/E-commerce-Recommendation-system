# app.py  (minimal HTTP layer)
import os
import time
from typing import Optional, List, Tuple

from fastapi import FastAPI, Header, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Keep DB helpers for optional warm-up & logging
from services.firebase_client import ensure_firestore, read_interactions_with_timeout, increment_interaction

# Single entrypoint that runs the whole recommendation + enrichment pipeline
from services.recommendation_pipeline import run_recommendation_pipeline, load_artifacts_once

app = FastAPI(title="Recommender - minimal HTTP layer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("INFERENCE_API_KEY", "dev-key")
USER_HEADER = "x-user-id"


def api_key_auth(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def require_user_id(x_user_id: Optional[str] = Header(None)):
    if not x_user_id:
        raise HTTPException(status_code=400, detail=f"Missing header {USER_HEADER}")
    return x_user_id


@app.on_event("startup")
def startup():
    # Warm artifacts inside the pipeline (no logic here)
    try:
        load_artifacts_once()
    except Exception:
        pass

    # ensure firestore client ready (no-op if not configured)
    try:
        ensure_firestore()
    except Exception:
        pass


@app.get("/recommend_for_me")
def recommend_for_me(k: int = 5, auth=Depends(api_key_auth), user_id: str = Depends(require_user_id)):
    """
    Minimal endpoint:
    - best-effort fetch of recent interactions (1s timeout)
    - delegates full recommend + explain pipeline to services.recommendation_pipeline.run_recommendation_pipeline
    - returns pipeline result with latency
    """
    start = time.time()

    # best-effort quick read of interactions (may return [] or None)
    interactions = None
    try:
        interactions = read_interactions_with_timeout(user_id, timeout=1.0)
    except Exception:
        interactions = None

    try:
        pipeline_resp = run_recommendation_pipeline(user_id=user_id, k=k, interactions=interactions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation pipeline error: {e}")

    latency = time.time() - start
    # Ensure response shape
    results = pipeline_resp.get("results") if isinstance(pipeline_resp, dict) else pipeline_resp
    return {"user_id": user_id, "k": k, "results": results, "latency": latency}


@app.post("/log_interaction")
def log_interaction(payload: dict, auth=Depends(api_key_auth)):
    """
    Minimal handler to record user interactions.
    Expects JSON body: { "user_id": "...", "product_id": "...", "weight": number (optional) }
    """
    uid = payload.get("user_id")
    pid = payload.get("product_id")
    try:
        w = float(payload.get("weight", 1.0))
    except Exception:
        w = 1.0

    if not uid or not pid:
        raise HTTPException(status_code=400, detail="user_id and product_id required")

    try:
        increment_interaction(uid, pid, delta=w)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log interaction: {e}")

    return {"status": "ok"}
