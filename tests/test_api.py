from fastapi.testclient import TestClient
from tsa.api.app import app

client = TestClient(app)

def test_status():
    r = client.get("/status")
    assert r.status_code == 200 and r.json()["ok"] is True

def test_pipeline():
    client.post("/bootstrap_synth_data")
    r = client.post("/run_pipeline", json={"n": 4})
    assert r.status_code == 200
    body = r.json()
    assert "selected" in body and "validation" in body
