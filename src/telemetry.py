from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import request


class TelemetryClient:
    """
    Minimal, optional telemetry sink for field monitoring.
    - Writes JSONL to a local file (for sync/collection).
    - Optionally POSTs to an HTTP endpoint if LUMENTACT_TELEMETRY_URL is set.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        log_path: Optional[str | Path] = None,
        timeout: float = 0.5,
    ):
        self.endpoint_url = endpoint_url or os.getenv("LUMENTACT_TELEMETRY_URL")
        self.log_path = Path(log_path) if log_path else self._default_log_path()
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint_url or self.log_path)

    def record_latency(
        self,
        latency_ms: float,
        budget_ms: float,
        breached: bool,
        frame_index: Optional[int] = None,
    ):
        payload = {
            "ts": time.time(),
            "type": "latency",
            "latency_ms": float(latency_ms),
            "budget_ms": float(budget_ms),
            "breached": bool(breached),
            "frame_index": frame_index,
        }
        self._emit(payload)

    def _emit(self, payload: Dict[str, Any]):
        if self.log_path:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")
            except Exception:
                pass

        if self.endpoint_url:
            thread = threading.Thread(target=self._post, args=(payload,), daemon=True)
            thread.start()

    def _post(self, payload: Dict[str, Any]):
        try:
            data = json.dumps(payload).encode("utf-8")
            req = request.Request(
                self.endpoint_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            request.urlopen(req, timeout=self.timeout)
        except Exception:
            # Telemetry failures should never break runtime.
            pass

    def _default_log_path(self) -> Optional[Path]:
        # Keep a local trace by default so field logs can be pulled if no network.
        default_path = Path("out") / "telemetry-latency.jsonl"
        return default_path
