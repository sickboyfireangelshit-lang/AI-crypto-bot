import time
import os
from flask import request, abort

# Admin secret (set in Render env)
ADMIN_SECRET = os.getenv("ADMIN_SECRET")

# In-memory key store (free tier safe)
# Later replace with Redis / DB
API_KEYS = {
    # example structure
    # "sk_live_xxx": {
    #     "plan": "pro",
    #     "expires": 1735689600,
    #     "gumroad_license": "XXXX-YYYY",
    #     "rate_limit": 60,
    #     "usage": []
    # }
}

def require_api_key():
    key = request.headers.get("X-API-Key")
    if not key or key not in API_KEYS:
        abort(401, description="Invalid or missing API key")

    meta = API_KEYS[key]

    # expiration check
    if time.time() > meta["expires"]:
        abort(403, description="API key expired")

    # rate limit check
    now = time.time()
    window = 60
    meta["usage"] = [t for t in meta["usage"] if now - t < window]

    if len(meta["usage"]) >= meta["rate_limit"]:
        abort(429, description="Rate limit exceeded")

    meta["usage"].append(now)

    return meta

def require_admin():
    secret = request.headers.get("X-Admin-Secret")
    if not ADMIN_SECRET or secret != ADMIN_SECRET:
        abort(403, description="Admin access denied")
