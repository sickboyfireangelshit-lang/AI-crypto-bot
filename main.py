
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import ccxt.async_support as ccxt
import asyncio
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from telegram import Bot
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AI Crypto Bot – Autonomous Swarm", version="1.1")

# Credentials (set these in Render env vars or .env)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID")

# Exchanges
binance = ccxt.binance({
    'apiKey': os.getenv("BINANCE_API"),
    'secret': os.getenv("BINANCE_SECRET"),
    'enableRateLimit': True,
})
bybit = ccxt.bybit({
    'apiKey': os.getenv("BYBIT_API"),
    'secret': os.getenv("BYBIT_SECRET"),
    'enableRateLimit': True,
})

# Mock ML model (replace with your trained model load)
scaler = StandardScaler()
model = RandomForestClassifier(n_estimators=100)
# For demo: train on dummy data – replace with real training
dummy_X = np.random.rand(100, 4)
dummy_y = np.random.randint(0, 2, 100)
model.fit(scaler.fit_transform(dummy_X), dummy_y)

# Strategy flags
GRID_ACTIVE = True
REVERSION_ACTIVE = True
MOMENTUM_ACTIVE = True
SENTIMENT_ACTIVE = True

async def notify_telegram(message: str):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT:
        bot = Bot(token=TELEGRAM_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT, text=message)

def predict_direction(symbol: str) -> str:
    # Placeholder prediction – integrate your real model
    features = np.random.rand(1, 4)
    pred = model.predict(scaler.transform(features))[0]
    return "BUY" if pred == 1 else "SELL"

# 1. Grid Arbitrage (Binance vs Bybit)
async def grid_arbitrage():
    while GRID_ACTIVE:
        try:
            binance_price = (await binance.fetch_ticker('BTC/USDT'))['bid']
            bybit_price = (await bybit.fetch_ticker('BTCUSDT'))['ask']
            spread = (bybit_price - binance_price) / binance_price
            if spread > 0.003:  # 0.3%
                await notify_telegram(f"Grid Arb: Buying Binance @ {binance_price}, Selling Bybit @ {bybit_price}")
                # Place orders here
            await asyncio.sleep(30)
        except Exception as e:
            logging.error(f"Grid error: {e}")
            await asyncio.sleep(60)

# 2. Mean Reversion
async def mean_reversion():
    while REVERSION_ACTIVE:
        try:
            ohlcv = await binance.fetch_ohlcv('BTC/USDT', '1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            mean = df['close'].rolling(20).mean().iloc[-1]
            current = df['close'].iloc[-1]
            if current < mean * 0.97:
                await notify_telegram("Reversion: BTC undervalued – entering long")
                # Execute trade
            await asyncio.sleep(300)
        except Exception as e:
            logging.error(f"Reversion error: {e}")

# 3. Momentum Chase
async def momentum_chase():
    while MOMENTUM_ACTIVE:
        try:
            pred = predict_direction('BTC/USDT')
            if pred == "BUY":
                await notify_telegram("Momentum: Strong BUY signal – entering position")
                # Execute
            await asyncio.sleep(600)
        except Exception as e:
            logging.error(f"Momentum error: {e}")

# 4. Sentiment Spike (simple placeholder – add real Twitter/Reddit API)
async def sentiment_spike():
    while SENTIMENT_ACTIVE:
        try:
            # Placeholder: random sentiment trigger
            if np.random.rand() > 0.9:
                await notify_telegram("Sentiment Alert: Positive spike detected on SOL – sniping")
            await asyncio.sleep(3600)
        except Exception as e:
            logging.error(f"Sentiment error: {e}")

# Background runner
async def run_strategies():
    tasks = []
    if GRID_ACTIVE: tasks.append(asyncio.create_task(grid_arbitrage()))
    if REVERSION_ACTIVE: tasks.append(asyncio.create_task(mean_reversion()))
    if MOMENTUM_ACTIVE: tasks.append(asyncio.create_task(momentum_chase()))
    if SENTIMENT_ACTIVE: tasks.append(asyncio.create_task(sentiment_spike()))
    await asyncio.gather(*tasks)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(run_strategies())

@app.get("/")
async def root():
    return {"message": "AI Crypto Bot Swarm is live – autonomous strategies engaged."}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
