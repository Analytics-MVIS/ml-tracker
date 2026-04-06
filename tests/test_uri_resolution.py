from mltracker.runtime.env import resolve_tracking_uri


def test_explicit_uri_wins(monkeypatch):
    monkeypatch.setenv("MLTRACK_TRACKING_URI", "http://mltrack-env")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-env")

    assert resolve_tracking_uri("http://explicit") == "http://explicit"


def test_mltrack_env_wins(monkeypatch):
    monkeypatch.setenv("MLTRACK_TRACKING_URI", "http://mltrack-env")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-env")

    assert resolve_tracking_uri(None) == "http://mltrack-env"


def test_mlflow_env_fallback(monkeypatch):
    monkeypatch.delenv("MLTRACK_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-env")

    assert resolve_tracking_uri(None) == "http://mlflow-env"


def test_internal_default(monkeypatch):
    monkeypatch.delenv("MLTRACK_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert resolve_tracking_uri(None) == "http://lab.l2m.internal:5000"
