from __future__ import annotations

import json
from pathlib import Path
import tempfile

from telemetry import TelemetryClient


def test_telemetry_client_logs_latency_events_locally():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "telemetry.jsonl"
        client = TelemetryClient(endpoint_url=None, log_path=log_path)
        client.record_latency(latency_ms=210.0, budget_ms=180.0, breached=True, frame_index=5)
        assert log_path.exists()
        data = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        assert data and data[0]["breached"] is True
        assert data[0]["frame_index"] == 5
