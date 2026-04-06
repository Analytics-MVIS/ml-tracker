from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import pytest

from mltracker.configs import YoloConfig
from mltracker.trackers.base import BaseTracker


def test_log_best_model_logs_and_registers_with_custom_name(monkeypatch, tmp_path: Path):
    cfg = YoloConfig(
        project="vision",
        run_name="run-1",
        model="yolov8n.pt",
        data="data.yaml",
        epochs=1,
        imgsz=640,
        batch=4,
        lr0=0.01,
        optimizer="SGD",
    )

    model_dir = tmp_path / "weights"
    model_dir.mkdir()
    (model_dir / "best.pt").write_text("best")
    (model_dir / "last.pt").write_text("last")

    artifacts: list[tuple[str, str | None]] = []
    tags: dict[str, str] = {}
    register_calls: list[tuple[str, str]] = []

    monkeypatch.setattr("mltracker.trackers.base.mlflow.log_artifact", lambda p, artifact_path=None: artifacts.append((p, artifact_path)))
    monkeypatch.setattr(
        "mltracker.trackers.base.mlflow.active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="run123")),
    )
    monkeypatch.setattr(
        "mltracker.trackers.base.mlflow.register_model",
        lambda model_uri, name: register_calls.append((model_uri, name)) or SimpleNamespace(version=7),
    )
    monkeypatch.setattr("mltracker.trackers.base.mlflow.set_tag", lambda k, v: tags.__setitem__(k, v))

    tracker = BaseTracker("exp", cfg, tracking_uri="http://mlflow.test")
    tracker.log_best_model(model_dir, model_name="detector-v1", register_name="yolo-prod")

    assert len(artifacts) == 2
    assert artifacts[0][0].endswith("detector-v1.pt")
    assert artifacts[1][0].endswith("last.pt")
    assert register_calls == [("runs:/run123/model/detector-v1.pt", "yolo-prod")]
    assert tags["model.name"] == "detector-v1.pt"
    assert tags["model.registry_name"] == "yolo-prod"
    assert tags["model.version"] == "7"
    assert tags["model.best_path"] == "model/detector-v1.pt"
    assert tags["model.last_path"] == "model/last.pt"


def test_log_model_logs_with_explicit_name_and_registers(monkeypatch, tmp_path: Path):
    cfg = YoloConfig(
        project="vision",
        run_name="run-2",
        model="yolov8n.pt",
        data="data.yaml",
        epochs=1,
        imgsz=640,
        batch=4,
        lr0=0.01,
        optimizer="SGD",
    )

    model_file = tmp_path / "best.pt"
    model_file.write_text("weights")

    artifacts: list[tuple[str, str | None]] = []
    tags: dict[str, str] = {}
    register_calls: list[tuple[str, str]] = []

    monkeypatch.setattr("mltracker.trackers.base.mlflow.log_artifact", lambda p, artifact_path=None: artifacts.append((p, artifact_path)))
    monkeypatch.setattr(
        "mltracker.trackers.base.mlflow.active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="run999")),
    )
    monkeypatch.setattr(
        "mltracker.trackers.base.mlflow.register_model",
        lambda model_uri, name: register_calls.append((model_uri, name)) or SimpleNamespace(version=3),
    )
    monkeypatch.setattr("mltracker.trackers.base.mlflow.set_tag", lambda k, v: tags.__setitem__(k, v))

    tracker = BaseTracker("exp", cfg, tracking_uri="http://mlflow.test")
    tracker.log_model(model_file, model_name="classifier-main", register_name="cls-prod")

    assert artifacts[0][0].endswith("classifier-main.pt")
    assert artifacts[0][1] == "model"
    assert register_calls == [("runs:/run999/model/classifier-main.pt", "cls-prod")]
    assert tags["model.name"] == "classifier-main.pt"
    assert tags["model.path"] == "model/classifier-main.pt"
    assert tags["model.registry_name"] == "cls-prod"
    assert tags["model.version"] == "3"


def test_log_model_raises_for_missing_file(tmp_path: Path):
    cfg = YoloConfig(
        project="vision",
        run_name="run-3",
        model="yolov8n.pt",
        data="data.yaml",
        epochs=1,
        imgsz=640,
        batch=4,
        lr0=0.01,
        optimizer="SGD",
    )
    tracker = BaseTracker("exp", cfg, tracking_uri="http://mlflow.test")

    with pytest.raises(FileNotFoundError):
        tracker.log_model(tmp_path / "missing.pt", model_name="missing")
