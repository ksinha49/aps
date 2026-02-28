"""Container healthcheck script — returns exit 0 if API is healthy."""

import sys
import urllib.request

try:
    with urllib.request.urlopen("http://localhost:8080/health", timeout=5) as resp:
        if resp.status == 200:
            sys.exit(0)
except Exception:
    pass

sys.exit(1)
