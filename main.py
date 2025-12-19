import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import ccxt.async_support as ccxt
import asyncio

app = FastAPI(title="AI Crypto Oracle 2025 🔥")

# Config from env
BINANCE_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET = os.getenv('BINANCE_API_SECRET')

exchange = ccxt.binance({
    'apiKey': BINANCE_KEY,
    'secret': BINANCE_SECRET,
    'enableRateLimit': True,
})

@app.get("/")
async def root():
    return {"message": "AI Crypto Oracle Live – Predicting. Trading. Conquering. ⚡"}

@app.get("/status")
async def status():
    balance = await exchange.fetch_balance()
    return {"balance": balance['total'], "oracle": "awakened"}

@app.post("/predict-and-trade")
async def predict_trade(background_tasks: BackgroundTasks):
    background_tasks.add_task(execute_ml_trade)
    return JSONResponse({"status": "ML prophecy launched in background – trade firing soon"})

async def execute_ml_trade():
    # Your ML + signal logic here (from previous evolutions)
    ticker = await exchange.fetch_ticker('BTC/USDT')
    price = ticker['last']
    # Predict → order
    order = await exchange.create_market_buy_order('BTC/USDT', 0.001)  # Example
    print(f"Prophetic trade executed at {price}: {order}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
