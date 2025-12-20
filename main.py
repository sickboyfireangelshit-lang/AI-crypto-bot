from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# =========================
# HEALTH CHECK (ONLY ONE)
# =========================
@app.route("/health", methods=["GET"])
def api_health():
    return {
        "status": "ok",
        "service": "crypto-api",
        "timestamp": datetime.utcnow().isoformat()
    }, 200


# =========================
# ROOT (OPTIONAL)
# =========================
@app.route("/", methods=["GET"])
def root():
    return {"message": "API running"}, 200


# =========================
# PROTECTED ENDPOINT EXAMPLE
# =========================
@app.route("/trade", methods=["POST"])
def trade():
    return {"status": "trade accepted"}, 200


# DO NOT define app.run() for Gunicorn
