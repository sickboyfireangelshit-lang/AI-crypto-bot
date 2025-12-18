bind = "0.0.0.0:8000"
workers = 4  # Unleash 4 workers – or match your CPU cores
timeout = 120
loglevel = "info"

# Optional: Let it auto-scale intelligently
# workers = multiprocessing.cpu_count() * 2 + 1
