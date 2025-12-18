import os
import multiprocessing

# Server socket
bind = "0.0.0.0:" + os.getenv("PORT", "8000")
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Timeouts
timeout = 120
keepalive = 5

# Process naming
proc_name = "ai-crypto-bot"

# Reload for development (disable in prod if needed)
reload = False
