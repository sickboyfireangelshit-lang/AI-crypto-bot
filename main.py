import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import ccxt.async_support as ccxt
import asyncio
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.telegram_alerts import send_alert, send_startup_alert
from core.ml_predictor import get_ml_signal
from core.portfolio import Portfolio
from analytics.logger import logger

app = FastAPI(title="AI Crypto Oracle – Autonomous Swarm 2025 🔥", version="1.1")

# Config & Exchange
BINANCE_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET = os.getenv('BINANCE_API_SECRET')

exchange = ccxt.binance({
    'apiKey': BINANCE_KEY,
    'secret': BINANCE_SECRET,
    'enableRateLimit': True,
})

portfolio = Portfolio()

@app.on_event("startup")
async def startup_event():
    await send_startup_alert()
    logger.info("Oracle awakened – ML models loading, swarm deploying")

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>AI Crypto Oracle Live</title></head>
        <body style="background:#000;color:#0f0;font-family:monospace;text-align:center;padding:50px;">
            <h1>🤖 AI Crypto Oracle – Autonomous Swarm 2025</h1>
            <h2>Status: LIVE • Predicting • Trading • Conquering</h2>
            <p><a href="/docs">Interactive Docs</a> • <a href="/charts">Profit Charts</a> • <a href="/health">Health</a></p>
            <p><a href="/signal">Signal Prophecy</a> • <a href="/portfolio">Portfolio Summary</a> • <a href="/trades">Trade History</a></p>
            <p>The empire compounds while you dream 🔥</p>
        </body>
    </html>
    """

@app.get("/health")
async def health():
    return {"status": "Oracle Alive – Vigilance Eternal 🔥", "timestamp": pd.Timestamp.now()}

@app.get("/status")
async def status():
    balance = await exchange.fetch_balance()
    return {
        "oracle": "LIVE",
        "balance": balance['total'],
        "portfolio": portfolio.get_summary(),
        "accuracy": "55.3%",  # Update from ML metrics
        "endpoints": ["/signal", "/portfolio", "/trades", "/charts", "/health"]
    }

@app.get("/signal")
async def signal(symbol: str = 'BTC/USDT'):
    df = await exchange.fetch_ohlcv(symbol, '1h', limit=100)
    df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    prediction = get_ml_signal(df)
    await send_alert(f"<b>🧠 ML PROPHECY</b>\n{prediction.upper()} signal on {symbol}\nConfidence rising")
    return {"signal": prediction, "symbol": symbol}

@app.get("/portfolio")
async def get_portfolio():
    summary = portfolio.get_summary()
    return {"portfolio": summary, "total_value": summary['total_value'], "pnl": summary['pnl']}

@app.get("/trades")
async def get_trades(limit: int = 10):
    trades = portfolio.get_recent_trades(limit)
    return {"trades": trades}

@app.get("/charts", response_class=HTMLResponse)
async def profit_charts():
    # Real trade history from portfolio
    trades = portfolio.get_trades_df()
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Equity Curve – Empire Rising', 'Daily PnL – Explosions'))
    fig.add_trace(go.Scatter(x=trades['date'], y=trades['equity'], name='Equity', line=dict(color='#00ff00')), row=1, col=1)
    fig.add_trace(go.Bar(x=trades['date'], y=trades['pnl'], name='PnL'), row=2, col=1)
    fig.update_layout(template='plotly_dark', height=800, title_text="Profit Dashboard – Visual Conquest 🔥")
    return HTMLResponse(fig.to_html(full_html=False, include_plotlyjs='cdn'))

async def background_trading_swarm():
    while True:
        try:
            await signal()  # Or full trade execution
            await asyncio.sleep(3600)  # Hourly prophecy cycle
        except Exception as e:
            logger.error(f"Swarm error: {e}")
            await asyncio.sleep(60)

# Launch the swarm on startup
app.add_event_handler("startup", lambda: asyncio.create_task(background_trading_swarm()))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
