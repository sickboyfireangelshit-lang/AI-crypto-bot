import time
import traceback

def run_worker():
    print("Worker started successfully")

    while True:
        try:
            # === YOUR BOT LOGIC GOES HERE ===
            execute_trading_cycle()

            time.sleep(60)  # adjust interval as needed

        except Exception as e:
            print("Worker error detected:")
            print(str(e))
            traceback.print_exc()

            # prevent crash loop
            time.sleep(30)

def execute_trading_cycle():
    # placeholder for your existing logic
    print("Running trading cycle")

if __name__ == "__main__":
    run_worker()

