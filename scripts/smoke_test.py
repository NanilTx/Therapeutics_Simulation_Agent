import json
import sys
import time
from typing import Any, Dict

import requests


BASE = "http://127.0.0.1:8000"


def require_ok(resp: requests.Response, label: str) -> Dict[str, Any]:
    if not resp.ok:
        print(f"[FAIL] {label}: HTTP {resp.status_code}")
        print(resp.text)
        sys.exit(1)
    try:
        return resp.json()
    except Exception:
        print(f"[FAIL] {label}: Non-JSON response: {resp.text[:200]}")
        sys.exit(1)


def main() -> None:
    # 1) Health
    r = requests.get(f"{BASE}/status", timeout=5)
    j = require_ok(r, "status")
    assert j.get("ok") is True
    print("[OK] status")

    # 2) Bootstrap synthetic data (idempotent)
    r = requests.post(f"{BASE}/bootstrap_synth_data", timeout=30)
    _ = require_ok(r, "bootstrap_synth_data")
    print("[OK] bootstrap_synth_data")

    # 3) Plan-only
    r = requests.post(f"{BASE}/plan", json={"n": 3}, timeout=30)
    j = require_ok(r, "plan")
    assert isinstance(j.get("proposals"), list)
    print(f"[OK] plan: {len(j['proposals'])} proposals")

    # 4) Full pipeline
    r = requests.post(f"{BASE}/run_pipeline", json={"n": 3}, timeout=60)
    j = require_ok(r, "run_pipeline")
    assert "selected" in j and "validation" in j
    sel = j["selected"]
    rmse = j["validation"].get("rmse")
    print("[OK] run_pipeline: selected=", json.dumps(sel)[:120], " validation.rmse=", rmse)

    print("[DONE] Smoke test passed.")


if __name__ == "__main__":
    main()

