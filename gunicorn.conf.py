import multiprocessing
import os

bind = os.environ.get("PORT", "0.0.0.0:8000")  # Use Render's $PORT
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"  # Use "uvicorn.workers.UvicornWorker" for FastAPI
timeout = 120
keepalive = 5
loglevel = "info"
accesslog = "-"
errorlog = "-"
