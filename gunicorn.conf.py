bind = "0.0.0.0:8000"
workers = 2  # Light for free tiers
timeout = 0  # Infinite wait – no forced kills
graceful_timeout = 300  # Smooth restarts
keepalive = 65  # Hold connections through cycles
preload_app = True  # Load once, fast revives
max_requests = 1000  # Auto-restart after requests
max_requests_jitter = 50  # Randomize for balance
