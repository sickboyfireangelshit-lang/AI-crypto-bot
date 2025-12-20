from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# =========================
# HEALTH CHECK (ONLY ONE)
# =========================
@app.route("/health", methods=["GET"], endpoint="api_health_check")
def api_health():
    """
    Health check endpoint
    """
    return {
        "status": "ok",
        "service": "crypto-api",
        "timestamp": datetime.utcnow().isoformat()
    }, 200


# =========================
# ROOT ENDPOINT
# =========================
@app.route("/", methods=["GET"])
def root():
    """
    Root endpoint to confirm API is running
    """
    return {"message": "API running"}, 200


# =========================
# PROTECTED TRADE ENDPOINT EXAMPLE
# =========================
@app.route("/trade", methods=["POST"])
def trade():
    """
    Example POST endpoint
    """
    data = request.json or {}
    # Here you can process trading data
    return {"status": "trade accepted", "received": data}, 200


# =========================
# Do NOT call app.run()
# Gunicorn will run this app
