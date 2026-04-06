from pydantic import Field, field_validator

from mltracker.configs.base import BaseConfig


class YoloConfig(BaseConfig):
    model: str = Field(..., min_length=1)
    data: str = Field(..., min_length=1)
    epochs: int = Field(..., ge=1)
    imgsz: int = Field(..., ge=32)
    batch: int = Field(..., ge=1)
    lr0: float = Field(..., gt=0)
    optimizer: str = Field(..., min_length=1)

    @field_validator("optimizer")
    @classmethod
    def normalize_optimizer(cls, value: str) -> str:
        return value.upper()
