import os

# The model itself uses significant RAM.  A single worker prevents duplicate
# model copies, while the longer timeout leaves room for CPU inference.
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = 1
threads = 1
timeout = 120
graceful_timeout = 30
keepalive = 5
preload_app = True
accesslog = "-"
errorlog = "-"
