from fastapi.testclient import TestClient

import app as app_module


def test_health_providers_returns_check_results(monkeypatch):
    fake_results = [
        {"agent_name": "researcher", "provider": "groq", "model": "llama-3.3-70b-versatile",
         "status": "ok", "detail": ""},
        {"agent_name": "researcher", "provider": "gemini", "model": "gemini-2.0-flash",
         "status": "error", "detail": "invalid api key"},
    ]
    monkeypatch.setattr("app.check_provider_health", lambda: fake_results)

    with TestClient(app_module.app) as client:
        response = client.get("/health/providers")

    assert response.status_code == 200
    assert response.json() == {"providers": fake_results}
