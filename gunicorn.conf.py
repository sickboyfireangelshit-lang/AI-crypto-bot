import multiprocessing

bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
timeout = 300                  # Long grace for ML/backtests
graceful_timeout = 300        # Smooth shutdown
keepalive = 10                # Persistent life
preload_app = True            # Load app before workers – faster restarts
worker_connections = 1000     # If using gevent/async
loglevel = "info"
