bind = "0.0.0.0:8000"
workers = 4  # Scale to your CPUs – feel the power multiply
timeout = 300  # 5 minutes of grace – vanquish hangs forever
graceful_timeout = 300  # Gentle restarts, no crashes
keepalive = 5  # Keep connections alive through storms
loglevel = "info"
