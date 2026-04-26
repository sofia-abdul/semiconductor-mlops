from fastapi.testclient import TestClient

from pipeline.deployment import app, expected_feature_count

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "threshold" in data
    assert "expected_feature_count" in data


def test_predict_rejects_wrong_feature_count():
    response = client.post("/predict", json={"features": [0.1, 0.2, 0.3]})

    assert response.status_code == 400


def test_predict_accepts_valid_feature_count():
    features = [0.0] * expected_feature_count

    response = client.post("/predict", json={"features": features})

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert "prediction_label" in data
    assert "failure_probability" in data
    assert "threshold" in data