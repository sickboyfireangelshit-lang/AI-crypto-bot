import os
import secrets
import inspect
from datetime import datetime
from typing import Any
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ccxt.async_support as ccxt
import asyncio

# -----------------------------
# Safe optional imports – never crash the oracle
# -----------------------------
def _safe_import():
    send_alert = None
    send_startup_alert = None
    get_ml_signal = None
    Portfolio = None
    logger = None
    
    try:
        from utils.telegram_alerts import send_alert as _sa, send_startup_alert as _ssa
        send_alert, send_startup_alert = _sa, _ssa
    except Exception as e:
        print(f"[WARN] Telegram alerts not available: {e}")
    
    try:
        from core.ml_predictor import get_ml_signal as _gms
        get_ml_signal = _gms
    except Exception as e:
        print(f"[WARN] ML predictor not available: {e}")
    
    try:
        from core.portfolio import Portfolio as _P
        Portfolio = _P
    except Exception as e:
        print(f"[WARN] Portfolio module not available: {e}")
    
    try:
        from analytics.logger import logger as _l
        logger = _l
    except Exception as e:
        print(f"[WARN] analytics.logger not available: {e}")
    
    return send_alert, send_startup_alert, get_ml_signal, Portfolio, logger

send_alert, send_startup_alert, get_ml_signal, Portfolio, logger = _safe_import()

def _log(msg: str):
    if logger:
        try:
            logger.info(msg)
            return
        except Exception:
            pass
    print(msg)

async def _maybe_await(fn, *args, **kwargs):
    if not fn:
        return None
    res = fn(*args, **kwargs)
    if inspect.isawaitable(res):
        return await res
    return res

# -----------------------------
# Auth – Render Free friendly
# -----------------------------
def _valid_keys() -> set[str]:
    raw = os.getenv("VALID_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}

def require_api_key(request: Request):
    key = request.headers.get("X-API-Key", "")
    if key not in _valid_keys():
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

def require_admin(request: Request):
    admin_secret = os.getenv("ADMIN_SECRET", "")
    provided = request.headers.get("X-Admin-Secret", "")
    if provided != admin_secret:
        raise HTTPException(status_code=403, detail="Admin access denied")

def generate_api_key(prefix: str = "sk_live_") -> str:
    return prefix + secrets.token_hex(24)

# -----------------------------
# Exchange & Portfolio
# -----------------------------
exchange = ccxt.binance({
    "apiKey": os.getenv("BINANCE_API_KEY"),
    "secret": os.getenv("BINANCE_API_SECRET"),
    "enableRateLimit": True,
})

portfolio = Portfolio() if Portfolio else None

# -----------------------------
# FastAPI App – The Oracle Awakens
# -----------------------------
app = FastAPI(title="AI Crypto Oracle – Autonomous Swarm 2025 🔥", version="1.1")

@app.on_event("startup")
async def startup_event():
    await _maybe_await(send_startup_alert)
    _log("Oracle awakened – startup complete")
    asyncio.create_task(background_trading_swarm())

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    _log(f"Uncaught exception: {exc}")
    await _maybe_await(send_alert, f"⚠️ Oracle Resilience Activated\nError: {exc}\nEmpire continues")
    return JSONResponse(status_code=500, content={"detail": "Oracle resilient – empire endures 🔥"})

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>AI Crypto Oracle 2025</title></head>
        <body style="background:#000;color:#0f0;font-family:monospace;text-align:center;padding:100px;">
            <h1>🤖 AI CRYPTO ORACLE – LIVE & RESILIENT</h1>
            <h2>Autonomous Swarm Active • Predicting • Trading • Conquering</h2>
            <p>
                <a href="/docs" style="color:#0f0;">Interactive Docs</a> •
                <a href="/redoc" style="color:#0f0;">ReDoc</a> •
                <a href="/charts" style="color:#0f0;">Profit Charts</a> •
                <a href="/health" style="color:#0f0;">Health</a>
            </p>
            <p>The empire self-heals and compounds eternal 🔥</p>
        </body>
    </html>
    """

@app.get("/health")
async def health():
    try:
        balance = await exchange.fetch_balance()
        return {"status": "Oracle Alive & Resilient", "balance": balance.get("total", {})}
    except Exception as e:
        _log(f"Health check failed: {e}")
        await _maybe_await(send_alert, f"⚠️ Health error: {e}")
        return {"status": "Resilient – recovering", "error": str(e)}

@app.get("/status", dependencies=[Depends(require_api_key)])
async def status():
    balance = await exchange.fetch_balance()
    return {
        "oracle": "LIVE & RESILIENT",
        "balance_total": balance.get("total", {}),
        "portfolio": portfolio.get_summary() if portfolio else {"warning": "portfolio not loaded"}
    }

@app.get("/signal", dependencies=[Depends(require_api_key)])
async def signal(symbol: str = "BTC/USDT"):
    if not get_ml_signal:
        raise HTTPException(status_code=500, detail="ML predictor unavailable")
    
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, "1h", limit=100)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        prediction = get_ml_signal(df)
        await _maybe_await(send_alert, f"🧠 PROPHECY: {prediction.upper()} on {symbol}")
        return {"signal": prediction, "symbol": symbol}
    except Exception as e:
        _log(f"Signal error: {e}")
        await _maybe_await(send_alert, f"❌ Signal failed: {e} → fallback HOLD")
        return {"signal": "hold", "error": str(e)}

@app.get("/portfolio", dependencies=[Depends(require_api_key)])
async def get_portfolio():
    if not portfolio:
        raise HTTPException(status_code=500, detail="Portfolio unavailable")
    return {"portfolio": portfolio.get_summary()}

@app.get("/trades", dependencies=[Depends(require_api_key)])
async def get_trades(limit: int = 10):
    if not portfolio:
        raise HTTPException(status_code=500, detail="Portfolio unavailable")
    return {"trades": portfolio.get_recent_trades(limit)}

@app.get("/charts", response_class=HTMLResponse, dependencies=[Depends(require_api_key)])
async def profit_charts():
    if not portfolio:
        raise HTTPException(status_code=500, detail="Portfolio unavailable")
    
    trades = portfolio.get_trades_df()
    if trades is None or len(trades) == 0:
        return HTMLResponse("<h2 style='font-family:monospace;color:#0f0;'>No trade data yet – empire building</h2>")
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Equity Curve", "Daily PnL"))
    fig.add_trace(go.Scatter(x=trades["date"], y=trades["equity"], name="Equity"), row=1, col=1)
    fig.add_trace(go.Bar(x=trades["date"], y=trades["pnl"], name="PnL"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=800, title_text="Profit Dashboard – Empire Growth")
    return HTMLResponse(fig.to_html(full_html=False, include_plotlyjs="cdn"))

@app.post("/admin/generate-key", dependencies=[Depends(require_admin)])
async def admin_generate_key(request: Request):
    body = await request.json()
    prefix = body.get("prefix", "sk_live_")
    return {"api_key": generate_api_key(prefix)}

# -----------------------------
# Background Swarm – Autonomous Heartbeat
# -----------------------------
async def background_trading_swarm():
    while True:
        try:
            if get_ml_signal:
                await signal(symbol="BTC/USDT")  # Internal call – no auth needed
            await asyncio.sleep(3600)  # Hourly prophecy
        except Exception as e:
            _log(f"Swarm error: {e} – self-healing")
            await _maybe_await(send_alert, f"⚠️ Swarm error: {e} → restarting")
            await asyncio.sleep(60)

# Local dev only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
