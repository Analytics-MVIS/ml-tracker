from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mltracker.configs import build_config
from mltracker.trackers.classification import ClassificationTracker
from mltracker.trackers.yolo import YoloTracker


def build_tracker(
    task_type: str,
    *,
    experiment_name: str,
    params: Mapping[str, Any],
    project: str | None = None,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    extra_params: Mapping[str, Any] | None = None,
) -> YoloTracker | ClassificationTracker:
    """Build a tracker with config normalization for team training scripts."""

    normalized = task_type.strip().lower()
    cfg = build_config(
        normalized,
        params,
        project=project,
        run_name=run_name,
        extra_params=extra_params,
    )

    if normalized in {"yolo", "detection", "detector"}:
        return YoloTracker(
            experiment_name=experiment_name,
            config=cfg,
            tracking_uri=tracking_uri,
        )

    return ClassificationTracker(
        experiment_name=experiment_name,
        config=cfg,
        tracking_uri=tracking_uri,
    )