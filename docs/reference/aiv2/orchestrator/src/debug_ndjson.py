"""NDJSON debug sink for Cursor debug mode (no secrets)."""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional

DEFAULT_DEBUG_LOG_PATH = "/home/fulltime/Desktop/AIv2/.cursor/debug-32fb75.log"
FALLBACK_DEBUG_LOG_PATH = "/tmp/debug-32fb75.log"
DEBUG_SESSION_ID = "32fb75"
# kubectl logs … | grep ORCH_DEBUG_NDJSON — works in k8s without volume / kubectl cp
STDERR_PREFIX = "ORCH_DEBUG_NDJSON "


def debug_ndjson(
    hypothesis_id: str,
    location: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    line = {
        "sessionId": DEBUG_SESSION_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    raw = json.dumps(line, ensure_ascii=False) + "\n"
    # Always mirror to stderr so k8s users can: kubectl logs deploy/orchestrator -n aiv2 | grep ORCH_DEBUG_NDJSON
    try:
        sys.stderr.write(STDERR_PREFIX + raw)
        sys.stderr.flush()
    except Exception:
        pass
    primary = os.environ.get("ORCHESTRATOR_DEBUG_NDJSON_PATH", DEFAULT_DEBUG_LOG_PATH)
    paths = [primary]
    if primary != FALLBACK_DEBUG_LOG_PATH:
        paths.append(FALLBACK_DEBUG_LOG_PATH)
    for path in paths:
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(raw)
            return
        except Exception:
            continue
