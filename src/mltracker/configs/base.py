from __future__ import annotations

from abc import ABC
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BaseConfig(BaseModel, ABC):
    """Base config for all tracker configurations."""

    model_config = ConfigDict(extra="forbid")

    project: str = Field(..., min_length=1)
    run_name: str = Field(..., min_length=1)
    extra_params: dict[str, Any] = Field(default_factory=dict)

    def to_mlflow_params(self) -> dict[str, str]:
        payload = self.model_dump(mode="python")
        extra = payload.pop("extra_params", {})

        flat: dict[str, str] = {}
        self._flatten(prefix="", value=payload, out=flat)
        self._flatten(prefix="extra", value=extra, out=flat)
        return flat

    @classmethod
    def _flatten(cls, prefix: str, value: Any, out: dict[str, str]) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                cls._flatten(next_prefix, nested, out)
            return

        if isinstance(value, (list, tuple)):
            out[prefix] = ",".join(str(item) for item in value)
            return

        out[prefix] = "" if value is None else str(value)
