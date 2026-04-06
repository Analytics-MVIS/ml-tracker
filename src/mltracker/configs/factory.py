from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

from mltracker.configs.classification import ClassificationConfig
from mltracker.configs.yolo import YoloConfig


T = TypeVar("T")


def _first_present(payload: Mapping[str, Any], keys: Iterable[str], default: T | None = None) -> Any | T | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return default


def _normalize_identity(
    payload: Mapping[str, Any],
    project: str | None,
    run_name: str | None,
) -> tuple[str, str]:
    resolved_project = project or _first_present(payload, ("project", "experiment", "experiment_name"))
    resolved_run_name = run_name or _first_present(payload, ("run_name", "name", "run", "run_id"))

    if not isinstance(resolved_project, str) or not resolved_project.strip():
        raise ValueError("project is required. Pass project=... or include 'project'/'experiment_name' in params.")
    if not isinstance(resolved_run_name, str) or not resolved_run_name.strip():
        raise ValueError("run_name is required. Pass run_name=... or include 'run_name'/'name' in params.")

    return resolved_project.strip(), resolved_run_name.strip()


def yolo_config_from_dict(
    params: Mapping[str, Any],
    *,
    project: str | None = None,
    run_name: str | None = None,
    extra_params: Mapping[str, Any] | None = None,
) -> YoloConfig:
    """Build a YoloConfig from loose training params used by team scripts."""

    resolved_project, resolved_run_name = _normalize_identity(params, project, run_name)

    mapped: dict[str, Any] = {
        "project": resolved_project,
        "run_name": resolved_run_name,
        "model": _first_present(params, ("model", "weights", "model_name")),
        "data": _first_present(params, ("data", "dataset", "dataset_name", "data_path", "dataset_path")),
        "epochs": _first_present(params, ("epochs", "num_epochs", "n_epochs")),
        "imgsz": _first_present(params, ("imgsz", "image_size", "img_size", "input_size")),
        "batch": _first_present(params, ("batch", "batch_size", "train_batch_size")),
        "lr0": _first_present(params, ("lr0", "learning_rate", "lr", "initial_lr")),
        "optimizer": _first_present(params, ("optimizer", "optim", "optim_name"), default="SGD"),
    }

    used_keys = {
        "project",
        "experiment",
        "experiment_name",
        "run_name",
        "name",
        "run",
        "run_id",
        "model",
        "weights",
        "model_name",
        "data",
        "dataset",
        "dataset_name",
        "data_path",
        "dataset_path",
        "epochs",
        "num_epochs",
        "n_epochs",
        "imgsz",
        "image_size",
        "img_size",
        "input_size",
        "batch",
        "batch_size",
        "train_batch_size",
        "lr0",
        "learning_rate",
        "lr",
        "initial_lr",
        "optimizer",
        "optim",
        "optim_name",
    }
    extras = {k: v for k, v in params.items() if k not in used_keys}
    if extra_params:
        extras.update(dict(extra_params))
    mapped["extra_params"] = extras

    return YoloConfig(**mapped)


def classification_config_from_dict(
    params: Mapping[str, Any],
    *,
    project: str | None = None,
    run_name: str | None = None,
    extra_params: Mapping[str, Any] | None = None,
) -> ClassificationConfig:
    """Build a ClassificationConfig from loose training params used by team scripts."""

    resolved_project, resolved_run_name = _normalize_identity(params, project, run_name)

    mapped: dict[str, Any] = {
        "project": resolved_project,
        "run_name": resolved_run_name,
        "model_name": _first_present(params, ("model_name", "model", "architecture", "backbone")),
        "dataset_name": _first_present(params, ("dataset_name", "dataset", "data", "data_path", "dataset_path")),
        "epochs": _first_present(params, ("epochs", "num_epochs", "n_epochs")),
        "learning_rate": _first_present(params, ("learning_rate", "lr", "lr0", "initial_lr")),
        "batch_size": _first_present(params, ("batch_size", "batch", "train_batch_size")),
        "num_classes": _first_present(params, ("num_classes", "classes", "class_count")),
    }

    used_keys = {
        "project",
        "experiment",
        "experiment_name",
        "run_name",
        "name",
        "run",
        "run_id",
        "model_name",
        "model",
        "architecture",
        "backbone",
        "dataset_name",
        "dataset",
        "data",
        "data_path",
        "dataset_path",
        "epochs",
        "num_epochs",
        "n_epochs",
        "learning_rate",
        "lr",
        "lr0",
        "initial_lr",
        "batch_size",
        "batch",
        "train_batch_size",
        "num_classes",
        "classes",
        "class_count",
    }
    extras = {k: v for k, v in params.items() if k not in used_keys}
    if extra_params:
        extras.update(dict(extra_params))
    mapped["extra_params"] = extras

    return ClassificationConfig(**mapped)


def build_config(
    task_type: str,
    params: Mapping[str, Any],
    *,
    project: str | None = None,
    run_name: str | None = None,
    extra_params: Mapping[str, Any] | None = None,
) -> YoloConfig | ClassificationConfig:
    """Create the correct config type based on task_type."""

    normalized = task_type.strip().lower()
    if normalized in {"yolo", "detection", "detector"}:
        return yolo_config_from_dict(
            params,
            project=project,
            run_name=run_name,
            extra_params=extra_params,
        )

    if normalized in {"classification", "classifier", "cls"}:
        return classification_config_from_dict(
            params,
            project=project,
            run_name=run_name,
            extra_params=extra_params,
        )

    raise ValueError("Unsupported task_type. Use one of: yolo, detection, classification.")