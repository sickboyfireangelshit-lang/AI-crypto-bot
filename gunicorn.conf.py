bind = "0.0.0.0:8000"
workers = 3  # Balanced for free tier
timeout = 600  # 10 min grace – handles long ML thinks
graceful_timeout = 600
keepalive = 20
preload_app = True  # Faster wakes
loglevel = "warning"  # Silence routine shutdown noise
