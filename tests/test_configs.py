import pytest
from pydantic import Field

from mltracker.configs import ClassificationConfig, YoloConfig


class TeamYoloConfig(YoloConfig):
    dataset_version: str = Field(..., min_length=1)


def test_yolo_config_logs_flattened_params():
    cfg = YoloConfig(
        project="vision",
        run_name="run-1",
        model="yolov8n.pt",
        data="data.yaml",
        epochs=5,
        imgsz=640,
        batch=16,
        lr0=0.01,
        optimizer="adam",
        extra_params={"owner": "team-a", "flags": {"aug": True}},
    )

    params = cfg.to_mlflow_params()
    assert params["project"] == "vision"
    assert params["optimizer"] == "ADAM"
    assert params["extra.owner"] == "team-a"
    assert params["extra.flags.aug"] == "True"


def test_team_subclass_fields_are_logged():
    cfg = TeamYoloConfig(
        project="vision",
        run_name="run-1",
        model="yolov8n.pt",
        data="data.yaml",
        epochs=5,
        imgsz=640,
        batch=16,
        lr0=0.01,
        optimizer="SGD",
        dataset_version="2026-04",
    )

    params = cfg.to_mlflow_params()
    assert params["dataset_version"] == "2026-04"


def test_classification_validates_num_classes():
    with pytest.raises(Exception):
        ClassificationConfig(
            project="cls",
            run_name="bad",
            model_name="resnet18",
            dataset_name="imagenet",
            epochs=10,
            learning_rate=0.001,
            batch_size=32,
            num_classes=1,
        )
