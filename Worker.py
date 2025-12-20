import time

def run_worker():
    while True:
        print("Worker running...")
        # Insert trading bot logic here
        time.sleep(30)


if __name__ == "__main__":
    run_worker()
