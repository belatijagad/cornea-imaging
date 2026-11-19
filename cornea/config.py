from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    TypeAdapter,
    field_validator,
)
from pydantic.dataclasses import dataclass

from cornea.model import available_backbones


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yml"

Precision = Literal["fp16", "bf16", "fp32"]
PositiveInt = Annotated[StrictInt, Field(gt=0)]
PositiveFloat = Annotated[StrictFloat, Field(gt=0)]
NonNegativeFloat = Annotated[StrictFloat, Field(ge=0)]
Probability = Annotated[StrictFloat, Field(ge=0, le=1)]


def _unique(values):
    if len(set(values)) != len(values):
        raise ValueError("must not contain duplicates")
    return values


type NonEmptyUniqueList[T] = Annotated[list[T], Field(strict=True, min_length=1), AfterValidator(_unique)]


@dataclass(config=ConfigDict(extra="forbid"))
class ModelConfig:
    backbones: NonEmptyUniqueList[str]
    pretrained: StrictBool
    freeze_backbone: NonEmptyUniqueList[StrictBool]

    @field_validator("backbones")
    @classmethod
    def validate_backbones(cls, names):
        supported = available_backbones()
        for name in names:
            if name not in supported:
                raise ValueError(f"Unknown backbone {name!r}; choose from {supported}")
        return names

    @field_validator("freeze_backbone", mode="before")
    @classmethod
    def normalize_freeze_backbone(cls, value):
        return [value] if isinstance(value, bool) else value


@dataclass(config=ConfigDict(extra="forbid"))
class TrainingConfig:
    output_dir: Path
    epochs: PositiveInt
    validation_interval: PositiveInt
    learning_rate: PositiveFloat
    weight_decay: NonNegativeFloat
    threshold: Probability
    seed: StrictInt
    precision: Precision = "fp16"
    embeddings: StrictBool = False


@dataclass(config=ConfigDict(extra="forbid"))
class DataConfig:
    data_root: Path
    labels_csv: Path | None
    labels: NonEmptyUniqueList[str]
    folder_labels: dict[str, list[Literal[0, 1]]]
    image_extensions: list[str]
    folds: Annotated[StrictInt, Field(ge=2)]
    batch_size: PositiveInt
    image_size: PositiveInt
    num_workers: Annotated[StrictInt, Field(ge=0)]
    mean: Annotated[list[StrictFloat], Field(min_length=3, max_length=3)]
    std: Annotated[list[PositiveFloat], Field(min_length=3, max_length=3)]
    horizontal_flip_probability: Probability
    rotation_degrees: StrictFloat

    def __post_init__(self):
        if any(len(values) != len(self.labels) for values in self.folder_labels.values()):
            raise ValueError("folder_labels must contain one binary target per label")


@dataclass(config=ConfigDict(extra="forbid"))
class ExperimentConfig:
    model: ModelConfig
    trainer: TrainingConfig
    data: DataConfig

    def __post_init__(self):
        if "maxvit_t" in self.model.backbones and self.data.image_size != 224:
            raise ValueError("maxvit_t requires data.image_size: 224")

    def save(self, path):
        values = _EXPERIMENT_CONFIG.dump_python(self, mode="json")
        Path(path).write_text(yaml.safe_dump(values, sort_keys=False))


_EXPERIMENT_CONFIG = TypeAdapter(ExperimentConfig)


def load_config(path=DEFAULT_CONFIG, overrides=None):
    values = yaml.safe_load(Path(path).read_text())
    if overrides:
        if not isinstance(values, dict) or set(overrides) - set(values):
            raise ValueError("Overrides must target existing configuration sections")
        for section, changes in overrides.items():
            if not isinstance(values[section], dict) or not isinstance(changes, dict):
                raise ValueError(f"{section} must be a YAML mapping")
            values[section] = {**values[section], **changes}
    return _EXPERIMENT_CONFIG.validate_python(values)
