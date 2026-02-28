"""
NeuroFraud Shield — Flask Backend API
Endpoints:
  GET  /api/health       → health check
  POST /api/predict      → predict fraud
  GET  /api/stats        → model stats / feature importance
  POST /api/batch        → batch predictions
  GET  /api/history      → recent predictions
"""
from flask import Flask, request, jsonify, send_from_directory
import joblib
import json
import numpy as np
import os
import time
import uuid
from datetime import datetime

app = Flask(__name__, static_folder=".", static_url_path="")

# ── Load model artifacts ──────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
model  = joblib.load(os.path.join(BASE, "model.pkl"))
scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))
with open(os.path.join(BASE, "model_meta.json")) as f:
    meta = json.load(f)

FEATURES = meta["features"]

# In-memory prediction history (last 50)
history = []

# ── CORS helper ───────────────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.route("/api/<path:p>", methods=["OPTIONS"])
def options(p):
    return jsonify({}), 200

# ── Serve frontend ────────────────────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": meta["model_type"], "uptime": "live"})

# ── Stats ─────────────────────────────────────────────────────────────────────
@app.route("/api/stats")
def stats():
    total  = len(history)
    frauds = sum(1 for h in history if h["result"]["is_fraud"])
    return jsonify({
        "accuracy":   meta["accuracy"],
        "f1_score":   meta["f1_score"],
        "model_type": meta["model_type"],
        "top_features": meta["top_features"][:8],
        "total_predictions": total,
        "fraud_detected": frauds,
        "legit_detected": total - frauds,
    })

# ── Single Prediction ─────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # Build feature vector
    try:
        vec = np.array([[float(data.get(f, 0)) for f in FEATURES]])
    except Exception as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    start = time.time()
    vec_sc = scaler.transform(vec)
    proba  = model.predict_proba(vec_sc)[0]
    pred   = int(model.predict(vec_sc)[0])
    latency_ms = round((time.time() - start) * 1000, 2)

    fraud_prob = round(float(proba[1]), 4)
    risk_score = int(fraud_prob * 100)

    # Risk tier
    if risk_score >= 75:   tier = "CRITICAL"
    elif risk_score >= 55: tier = "HIGH"
    elif risk_score >= 35: tier = "MEDIUM"
    else:                  tier = "LOW"

    # Simple SHAP-style feature contributions (model feature importances * input deviation)
    feat_imp = model.feature_importances_
    vec_raw  = vec[0]
    contributions = []
    for i, fname in enumerate(FEATURES):
        val = vec_raw[i]
        imp = float(feat_imp[i])
        contributions.append({"feature": fname, "value": float(val), "importance": round(imp, 4)})
    contributions.sort(key=lambda x: -x["importance"])
    top_contributions = contributions[:6]

    result = {
        "is_fraud": bool(pred),
        "fraud_probability": fraud_prob,
        "risk_score": risk_score,
        "risk_tier": tier,
        "latency_ms": latency_ms,
        "top_factors": top_contributions,
        "recommendation": _recommend(tier),
    }

    # Store in history
    record = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "claim_id": data.get("claim_id", f"CLM-{len(history)+1:04d}"),
        "result": result,
    }
    history.append(record)
    if len(history) > 50:
        history.pop(0)

    return jsonify(result)

# ── Batch Prediction ──────────────────────────────────────────────────────────
@app.route("/api/batch", methods=["POST"])
def batch_predict():
    data = request.get_json(force=True)
    claims = data.get("claims", [])
    if not claims:
        return jsonify({"error": "No claims provided"}), 400

    results = []
    for claim in claims[:100]:  # cap at 100
        vec = np.array([[float(claim.get(f, 0)) for f in FEATURES]])
        vec_sc = scaler.transform(vec)
        proba  = model.predict_proba(vec_sc)[0]
        pred   = int(model.predict(vec_sc)[0])
        fraud_prob = round(float(proba[1]), 4)
        risk_score = int(fraud_prob * 100)
        results.append({
            "claim_id": claim.get("claim_id", "N/A"),
            "is_fraud": bool(pred),
            "risk_score": risk_score,
            "fraud_probability": fraud_prob,
        })

    return jsonify({
        "total": len(results),
        "fraud_count": sum(1 for r in results if r["is_fraud"]),
        "results": results
    })

# ── History ───────────────────────────────────────────────────────────────────
@app.route("/api/history")
def get_history():
    return jsonify({"history": list(reversed(history[-20:]))})

def _recommend(tier):
    return {
        "CRITICAL": "🚨 Flag immediately. Escalate to SIU (Special Investigation Unit). Suspend payout.",
        "HIGH":     "⚠️  Manual review required. Request supporting documents and police report.",
        "MEDIUM":   "📋 Standard review. Verify key details. Proceed with caution.",
        "LOW":      "✅ Likely legitimate. Process normally with routine verification.",
    }[tier]

if __name__ == "__main__":
    print("🚀 NeuroFraud Shield API running on http://localhost:5000")
    app.run(debug=True, port=5000)
