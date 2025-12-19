bind = "0.0.0.0:8000"
workers = 2  # Light & fast for free tier
worker_class = "uvicorn.workers.UvicornWorker"  # The async bridge awakens
timeout = 120
keepalive = 5
loglevel = "info"
