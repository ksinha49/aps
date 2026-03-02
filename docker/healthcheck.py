"""Container healthcheck script — returns exit 0 if API is healthy."""

from __future__ import annotations

import os
import sys
import urllib.request

port = os.environ.get("SCOUT_API_PORT", "8080")

try:
    with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as resp:
        if resp.status == 200:
            sys.exit(0)
except Exception:
    pass

sys.exit(1)
