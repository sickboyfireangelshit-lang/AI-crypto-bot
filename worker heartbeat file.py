# in worker.py
with open("heartbeat.txt", "w") as f:
    f.write(str(time.time()))
