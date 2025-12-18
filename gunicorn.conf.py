import multiprocessing
import os

bind = os.environ.get("PORT", "0.0.0.0:8000")  # Render's dynamic $PORT [web:10]
workers = multiprocessing.cpu_count() * 2 + 1   # Scales to CPU cores [web:10]
worker_class = "uvicorn.workers.UvicornWorker"  # Required for FastAPI ASGI [web:15][memory:2]
timeout = 120                  # Handles long trading signal requests
keepalive = 5                  # Efficient for API polling endpoints
loglevel = "info"              # Production logging
accesslog = "-"                # Stream to stdout for Render
errorlog = "-"                 # Stream errors to stdout

