from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
import shutil
import tempfile
from typing import Any, Generic, TypeVar

import mlflow

from mltracker.configs.base import BaseConfig
from mltracker.runtime.env import resolve_tracking_uri
from mltracker.runtime.system_tags import collect_system_tags

C = TypeVar("C", bound=BaseConfig)


class BaseTracker(AbstractContextManager["BaseTracker[C]"], Generic[C]):
    def __init__(
        self,
        experiment_name: str,
        config: C,
        tracking_uri: str | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.config = config
        self.tracking_uri = resolve_tracking_uri(tracking_uri)
        self._active = False

    def start(self) -> "BaseTracker[C]":
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.config.run_name)

        params = self.config.to_mlflow_params()
        if params:
            mlflow.log_params(params)

        tags = collect_system_tags()
        tags["tracker.config"] = self.config.__class__.__name__
        mlflow.set_tags(tags)

        self._active = True
        return self

    def end(self, status: str = "FINISHED") -> None:
        if self._active:
            mlflow.end_run(status=status)
            self._active = False

    def fail(self) -> None:
        self.end(status="FAILED")

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if step is None:
            mlflow.log_metric(key, value)
            return
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if step is None:
            mlflow.log_metrics(metrics)
            return
        mlflow.log_metrics(metrics, step=step)

    def log_param(self, key: str, value: Any) -> None:
        mlflow.log_param(key, value)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        target = Path(path)
        if target.is_dir():
            mlflow.log_artifacts(str(target), artifact_path=artifact_path)
        else:
            mlflow.log_artifact(str(target), artifact_path=artifact_path)

    def _resolve_model_filename(self, model_name: str | None, default_name: str) -> str:
        if model_name is None:
            return default_name

        sanitized = model_name.strip().replace(" ", "-")
        if not sanitized:
            return default_name

        path_name = Path(sanitized).name
        if Path(path_name).suffix:
            return path_name

        default_suffix = Path(default_name).suffix
        if default_suffix:
            return f"{path_name}{default_suffix}"
        return f"{path_name}.pt"

    def _log_artifact_with_name(
        self,
        source_path: Path,
        artifact_path: str,
        artifact_name: str,
    ) -> str:
        if source_path.name == artifact_name:
            self.log_artifact(source_path, artifact_path=artifact_path)
            return f"{artifact_path}/{artifact_name}"

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / artifact_name
            shutil.copy2(source_path, target)
            self.log_artifact(target, artifact_path=artifact_path)

        return f"{artifact_path}/{artifact_name}"

    def log_model(
        self,
        model_path: str | Path,
        model_name: str,
        artifact_path: str = "model",
        register_name: str | None = None,
    ) -> None:
        source = Path(model_path)
        if not source.exists() or source.is_dir():
            raise FileNotFoundError(f"Model file not found: {source}")

        resolved_name = self._resolve_model_filename(model_name, default_name=source.name)
        logged_relative_path = self._log_artifact_with_name(source, artifact_path, resolved_name)

        mlflow.set_tag("model.name", resolved_name)
        mlflow.set_tag("model.path", logged_relative_path)

        if register_name:
            active_run = mlflow.active_run()
            if active_run is None:
                raise RuntimeError("No active MLflow run. Start a run before model registration.")

            model_uri = f"runs:/{active_run.info.run_id}/{logged_relative_path}"
            result = mlflow.register_model(model_uri=model_uri, name=register_name)
            mlflow.set_tag("model.registry_name", register_name)
            mlflow.set_tag("model.version", str(result.version))

    def log_best_model(
        self,
        weights_path: str | Path,
        model_name: str | None = None,
        register_name: str | None = None,
    ) -> None:
        weights_dir = Path(weights_path)
        best = weights_dir / "best.pt"
        last = weights_dir / "last.pt"

        best_relative_path: str | None = None
        if best.exists():
            resolved_name = self._resolve_model_filename(model_name, default_name="best.pt")
            best_relative_path = self._log_artifact_with_name(
                best,
                artifact_path="model",
                artifact_name=resolved_name,
            )
            mlflow.set_tag("model.name", resolved_name)
            mlflow.set_tag("model.best_path", best_relative_path)
        if last.exists():
            self.log_artifact(last, artifact_path="model")
            mlflow.set_tag("model.last_path", "model/last.pt")

        if register_name and best_relative_path:
            active_run = mlflow.active_run()
            if active_run is None:
                raise RuntimeError("No active MLflow run. Start a run before model registration.")

            model_uri = f"runs:/{active_run.info.run_id}/{best_relative_path}"
            result = mlflow.register_model(model_uri=model_uri, name=register_name)
            mlflow.set_tag("model.registry_name", register_name)
            mlflow.set_tag("model.version", str(result.version))

    def log_confusion_matrix(self, image_path: str | Path) -> None:
        self.log_artifact(image_path, artifact_path="evaluation")

    def log_validation_images(self, dir_path: str | Path) -> None:
        self.log_artifact(dir_path, artifact_path="val_images")

    def __enter__(self) -> "BaseTracker[C]":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc is not None:
            self.fail()
            return False
        self.end(status="FINISHED")
        return False

    def run(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with self:
                return fn(*args, **kwargs)

        return wrapped
