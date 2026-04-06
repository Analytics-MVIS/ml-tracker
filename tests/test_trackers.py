from __future__ import annotations

from pathlib import Path

import pytest

from mltracker.configs import YoloConfig
from mltracker.trackers.base import BaseTracker


@pytest.fixture
def config() -> YoloConfig:
    return YoloConfig(
        project="vision",
        run_name="run-1",
        model="yolov8n.pt",
        data="data.yaml",
        epochs=3,
        imgsz=640,
        batch=8,
        lr0=0.01,
        optimizer="SGD",
    )


def test_start_logs_params_and_tags(monkeypatch, config):
    calls = {"params": None, "tags": None}

    monkeypatch.setattr("mltracker.trackers.base.mlflow.set_tracking_uri", lambda _uri: None)
    monkeypatch.setattr("mltracker.trackers.base.mlflow.set_experiment", lambda _e: None)
    monkeypatch.setattr("mltracker.trackers.base.mlflow.start_run", lambda **_k: None)
    monkeypatch.setattr("mltracker.trackers.base.mlflow.log_params", lambda p: calls.__setitem__("params", p))
    monkeypatch.setattr("mltracker.trackers.base.mlflow.set_tags", lambda t: calls.__setitem__("tags", t))
    monkeypatch.setattr("mltracker.trackers.base.collect_system_tags", lambda: {"system.python": "3.11"})

    tracker = BaseTracker("exp", config, tracking_uri="http://mlflow.test")
    tracker.start()

    assert calls["params"]["project"] == "vision"
    assert calls["tags"]["tracker.config"] == "YoloConfig"


def test_context_marks_failed_on_exception(monkeypatch, config):
    statuses: list[str] = []

    monkeypatch.setattr("mltracker.trackers.base.mlflow.set_tracking_uri", lambda _uri: None)
    monkeypatch.setattr("mltracker.trackers.base.mlflow.set_experiment", lambda _e: None)
    monkeypatch.setattr("mltracker.trackers.base.mlflow.start_run", lambda **_k: None)
    monkeypatch.setattr("mltracker.trackers.base.mlflow.log_params", lambda _p: None)
    monkeypatch.setattr("mltracker.trackers.base.mlflow.set_tags", lambda _t: None)
    monkeypatch.setattr("mltracker.trackers.base.collect_system_tags", lambda: {})
    monkeypatch.setattr("mltracker.trackers.base.mlflow.end_run", lambda status=None: statuses.append(status))

    tracker = BaseTracker("exp", config, tracking_uri="http://mlflow.test")
    with pytest.raises(RuntimeError):
        with tracker:
            raise RuntimeError("boom")

    assert statuses == ["FAILED"]


def test_log_helpers_route_artifacts(monkeypatch, config, tmp_path: Path):
    files: list[tuple[str, str | None]] = []
    dirs: list[tuple[str, str | None]] = []

    cm = tmp_path / "cm.png"
    cm.write_text("x")
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    (val_dir / "img1.png").write_text("x")

    monkeypatch.setattr("mltracker.trackers.base.mlflow.log_artifact", lambda p, artifact_path=None: files.append((p, artifact_path)))
    monkeypatch.setattr("mltracker.trackers.base.mlflow.log_artifacts", lambda p, artifact_path=None: dirs.append((p, artifact_path)))

    tracker = BaseTracker("exp", config, tracking_uri="http://mlflow.test")
    tracker.log_confusion_matrix(cm)
    tracker.log_validation_images(val_dir)

    assert files[0][1] == "evaluation"
    assert dirs[0][1] == "val_images"
