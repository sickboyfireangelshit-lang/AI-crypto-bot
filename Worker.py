import time
from datetime import datetime

# import your existing modules
from core.ml_predictor import predict
from data.exchange import get_market_data
from analytics.metrics import update_metrics
from config import settings

def run_worker():
    print("Worker started")

    while True:
        try:
            market = get_market_data()
            signal = predict(market)

            update_metrics(signal)

            # control loop speed
            time.sleep(30)

        except Exception as e:
            print(f"Worker error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run_worker()
