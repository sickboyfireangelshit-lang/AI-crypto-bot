from flask import Flask, jsonify
from datetime import datetime

app = Flask(__name__)

@app.route("/health")
def health():
    return jsonify({
        "name": "AI Crypto Trading Bot",
        "status": "LIVE",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/signal")
def signal():
    return jsonify({"signal": "dynamic from analytics"})

@app.route("/portfolio")
def portfolio():
    return jsonify({"status": "tracked in worker"})

if __name__ == "__main__":
    app.run()
@app.route("/health"
def health():
    return {
        "status": "ok",
        "service": "crypto-api",
        "timestamp": datetime.utcnow().isoformat()
    }

