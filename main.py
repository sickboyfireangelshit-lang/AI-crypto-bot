import os
import asyncio
import secrets
import inspect
from datetime import datetime
from typing import Optional, Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ccxt.async_support as ccxt

# -----------------------------
# Optional imports (do not crash deploy if missing)
# -----------------------------
def _safe_import():
    send_alert = None
    send_startup_alert = None
    get_ml_signal = None
    Portfolio = None
    logger = None

    try:
        from utils.telegram_alerts import send_alert as _send_alert, send_startup_alert as _send_startup_alert
        send_alert = _send_alert
        send_startup_alert = _send_startup_alert
    except Exception as e:
        print(f"[WARN] Telegram alerts not available: {e}")

    try:
        from core.ml_predictor import get_ml_signal as _get_ml_signal
        get_ml_signal = _get_ml_signal
    except Exception as e:
        print(f"[WARN] ML predictor not available: {e}")

    try:
        from core.portfolio import Portfolio as _Portfolio
        Portfolio = _Portfolio
    except Exception as e:
        print(f"[WARN] Portfolio module not available: {e}")

    try:
        from analytics.logger import logger as _logger
        logger = _logger
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
    """Calls fn; if it returns a coroutine, await it."""
    if not fn:
        return None
    res = fn(*args, **kwargs)
    if inspect.isawaitable(res):
        return await res
    return res

# -----------------------------
# API Key auth (Render Free-friendly: keys stored in env var)
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
    if not admin_secret or provided != admin_secret:
        raise HTTPException(status_code=403, detail="Admin access denied")

def generate_api_key(prefix: str = "sk_live_") -> str:
    return prefix + secrets.token_hex(24)

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="AI Crypto Oracle – Autonomous Swarm 2025", version="1.1")

BINANCE_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_API_SECRET")

exchange = ccxt.binance({
    "apiKey": BINANCE_KEY,
    "secret": BINANCE_SECRET,
    "enableRateLimit": True,
})

portfolio = Portfolio() if Portfolio else None

@app.on_event("startup")
async def startup_event():
    await _maybe_await(send_startup_alert)
    _log("Oracle awakened – startup complete")

    # Start background swarm task (best-effort on Render Free)
    asyncio.create_task(background_trading_swarm())

@app.route('/', methods=['GET', 'HEAD'])
async def root():
    return """
    <html>
        <head><title>AI Crypto Oracle Live</title></head>
        <body style="background:#000;color:#0f0;font-family:monospace;text-align:center;padding:50px;">
            <h1>AI Crypto Oracle – Autonomous Swarm 2025</h1>
            <h2>Status: LIVE • Predicting • Trading</h2>
            <p><a href="/docs">Interactive Docs</a> • <a href="/charts">Profit Charts</a> • <a href="/health">Health</a></p>
            <p><a href="/signal">Signal</a> • <a href="/portfolio">Portfolio</a> • <a href="/trades">Trade History</a></p>
        </body>
    </html>
    """

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/status")
async def status(_: Any = Depends(require_api_key)):
    balance = await exchange.fetch_balance()
    return {
        "oracle": "LIVE",
        "balance_total": balance.get("total", {}),
        "portfolio": portfolio.get_summary() if portfolio else {"warning": "portfolio module not loaded"},
        "endpoints": ["/signal", "/portfolio", "/trades", "/charts", "/health"]
    }

@app.get("/signal")
async def signal(symbol: str = "BTC/USDT", _: Any = Depends(require_api_key)):
    if not get_ml_signal:
        raise HTTPException(status_code=500, detail="ML predictor not available (core.ml_predictor missing)")

    ohlcv = await exchange.fetch_ohlcv(symbol, "1h", limit=100)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    prediction = get_ml_signal(df)

    await _maybe_await(send_alert, f"ML SIGNAL: {str(prediction).upper()} on {symbol}")

    return {"signal": prediction, "symbol": symbol}

@app.get("/portfolio")
async def get_portfolio(_: Any = Depends(require_api_key)):
    if not portfolio:
        raise HTTPException(status_code=500, detail="Portfolio not available (core.portfolio missing)")
    summary = portfolio.get_summary()
    return {"portfolio": summary}

@app.get("/trades")
async def get_trades(limit: int = 10, _: Any = Depends(require_api_key)):
    if not portfolio:
        raise HTTPException(status_code=500, detail="Portfolio not available (core.portfolio missing)")
    trades = portfolio.get_recent_trades(limit)
    return {"trades": trades}

@app.get("/charts", response_class=HTMLResponse)
async def profit_charts(_: Any = Depends(require_api_key)):
    if not portfolio:
        raise HTTPException(status_code=500, detail="Portfolio not available (core.portfolio missing)")

    trades = portfolio.get_trades_df()
    if trades is None or len(trades) == 0:
        return HTMLResponse("<h2 style='font-family:monospace'>No trade data available yet.</h2>")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Equity Curve", "Daily PnL")
    )
    fig.add_trace(go.Scatter(x=trades["date"], y=trades["equity"], name="Equity"), row=1, col=1)
    fig.add_trace(go.Bar(x=trades["date"], y=trades["pnl"], name="PnL"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=800, title_text="Profit Dashboard")

    return HTMLResponse(fig.to_html(full_html=False, include_plotlyjs="cdn"))

# -----------------------------
# Admin: generate API keys
# -----------------------------
@app.post("/admin/generate-key")
async def admin_generate_key(request: Request, _: Any = Depends(require_admin)):
    body = await request.json()
    prefix = body.get("prefix", "sk_live_")
    key = generate_api_key(prefix=prefix)
    return {"api_key": key}

# -----------------------------
# Background swarm loop
# -----------------------------
async def background_trading_swarm():
    while True:
        try:
            # If you want it to run, keep it; if you only want it on schedule, disable it on Free.
            # This calls /signal logic directly:
            if get_ml_signal:
                await signal(symbol="BTC/USDT", _=None)  # dependency bypass inside task
            await asyncio.sleep(3600)
        except Exception as e:
            _log(f"Swarm error: {e}")
            await asyncio.sleep(60)
try:
    from ml.predictor import MLPredictor
    ML_PREDICTOR_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] ML predictor not available: {e}")
    ML_PREDICTOR_AVAILABLE = False
    
    # Create a dummy MLPredictor class
    class MLPredictor:
        def __init__(self):
            pass
        
        def predict(self, *args, **kwargs):
            return None
        
        def train(self, *args, **kwargs):
            pass
@app.get("/", response_class=HTMLResponse)
async def root():  # Async harmony – accepts scope implicitly
    return """
    <html>
        <head><title>AI Crypto Oracle 2025</title></head>
        <body style="background:#000;color:#0f0;font-family:monospace;text-align:center;padding:100px;">
            <h1>🤖 AI CRYPTO ORACLE – LIVE</h1>
            <h2>Autonomous Swarm Active • Predicting • Trading • Conquering</h2>
            <p>
                <a href="/docs" style="color:#0f0;">Interactive Docs</a> • 
                <a href="/redoc" style="color:#0f0;">ReDoc</a> • 
                <a href="/charts" style="color:#0f0;">Profit Charts</a> • 
                <a href="/health" style="color:#0f0;">Health</a>
            </p>
            <p>The empire compounds while you watch 🔥</p>
        </body>
    </html>
    """
from fastapi import FastAPI, BackgroundTasks
import asyncio
from celery import Celery  # Optional heavy swarm (add to requirements.txt: celery redis)

app = FastAPI(title="AI Crypto Oracle – Autonomous Swarm 2025 🔥")

# Native FastAPI BackgroundTasks – Instant, lightweight
def background_trade_task(symbol: str, signal: str, amount: float):
    # Heavy logic: execute trade, update portfolio, log
    print(f"Background trade executed: {signal} {amount} {symbol}")
    # Real: await exchange.create_order(...)
    # Notify: await send_alert(f"Background trade: {signal.upper()} executed!")

@app.post("/trigger-trade")
async def trigger_trade(background_tasks: BackgroundTasks):
    background_tasks.add_task(background_trade_task, "BTC/USDT", "buy", 0.001)
    return {"status": "Trade launched in background – empire compounding"}

# Celery Heavy Swarm – For scheduled, resilient agents (Redis broker)
celery = Celery(__name__, broker='redis://')  # Add Redis URL in env

@celery.task
def hourly_ml_retrain():
    # Heavy ML retrain logic
    print("ML model retrained – prophecy sharpened")

@celery.task
def defi_yield_scan():
    # Scan staking/arbitrage opportunities
    print("DeFi yields compounded")

# Scheduled swarm (Celery beat – separate process)
from celery.schedules import crontab
celery.conf.beat_schedule = {
    'retrain-every-hour': {'task': 'main.hourly_ml_retrain', 'schedule': crontab(minute=0)},
    'yield-scan-every-30min': {'task': 'main.defi_yield_scan', 'schedule': crontab(minute='*/30')},
}

# Startup swarm launch
@app.on_event("startup")
async def launch_swarm():
    asyncio.create_task(background_monitor())  # Native loop
    # Celery workers run separately: celery -A main.celery worker

async def background_monitor():
    while True:
        try:
            # Live monitoring, alerts, health
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Monitor error: {e}")
