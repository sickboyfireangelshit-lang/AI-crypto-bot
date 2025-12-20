import os
import secrets
from flask import request, abort

def generate_api_key(prefix: str = "sk_live_") -> str:
    # 48 hex chars (~192 bits) is plenty
    return prefix + secrets.token_hex(24)

def get_valid_keys() -> set[str]:
    raw = os.getenv("VALID_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}

def require_api_key():
    key = request.headers.get("X-API-Key", "")
    if key not in get_valid_keys():
        abort(401, description="Invalid or missing API key")

def require_admin():
    admin = os.getenv("ADMIN_SECRET", "")
    provided = request.headers.get("X-Admin-Secret", "")
    if not admin or provided != admin:
        abort(403, description="Admin access denied")
