import os
import multiprocessing

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 120
keepalive = 5
loglevel = "info"
accesslog = "-"
errorlog = "-"
proc_name = "ai-crypto-bot"
reload = False
