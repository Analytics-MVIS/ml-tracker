from pydantic import Field, field_validator

from mltracker.configs.base import BaseConfig


class ClassificationConfig(BaseConfig):
    model_name: str = Field(..., min_length=1)
    dataset_name: str = Field(..., min_length=1)
    epochs: int = Field(..., ge=1)
    learning_rate: float = Field(..., gt=0)
    batch_size: int = Field(..., ge=1)
    num_classes: int = Field(..., ge=2)

    @field_validator("dataset_name")
    @classmethod
    def trim_dataset_name(cls, value: str) -> str:
        return value.strip()
