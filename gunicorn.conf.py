import multiprocessing

bind = "0.0.0.0:8000"          # Listen everywhere on port 8000
workers = multiprocessing.cpu_count() * 2 + 1  # Optimal for most CPUs
worker_class = "sync"          # Default; use "gevent" for async magic
timeout = 120                  # Kill hung workers after 2 mins
keepalive = 5                  # Persistent connections
loglevel = "info"              # Verbose logs
accesslog = "-"                # Stdout access logs
errorlog = "-"                 # Stdout errors
