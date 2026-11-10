from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from cornea.model import available_backbones


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yml"


@dataclass
class ModelConfig:
    backbones: list[str]
    pretrained: bool
    freeze_backbone: list[bool]

    def __post_init__(self):
        if not isinstance(self.backbones, list) or not self.backbones:
            raise ValueError("backbones must be a non-empty list")
        if len(set(self.backbones)) != len(self.backbones):
            raise ValueError("backbones must not contain duplicates")
        for name in self.backbones:
            if name not in available_backbones():
                raise ValueError(f"Unknown backbone {name!r}; choose from {available_backbones()}")
        if type(self.pretrained) is not bool:
            raise ValueError("pretrained must be a boolean")
        # Accept previous single-value configs as a one-option sweep.
        if type(self.freeze_backbone) is bool:
            self.freeze_backbone = [self.freeze_backbone]
        if (not isinstance(self.freeze_backbone, list) or not self.freeze_backbone
                or any(type(value) is not bool for value in self.freeze_backbone)):
            raise ValueError("freeze_backbone must be a non-empty list of booleans")
        if len(set(self.freeze_backbone)) != len(self.freeze_backbone):
            raise ValueError("freeze_backbone must not contain duplicates")


@dataclass
class TrainingConfig:
    output_dir: Path
    epochs: int
    validation_interval: int
    learning_rate: float
    weight_decay: float
    threshold: float
    seed: int
    precision: str = "fp16"

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if self.precision not in ("fp16", "bf16", "fp32"):
            raise ValueError("precision must be fp16, bf16, or fp32")
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if type(self.validation_interval) is not int or self.validation_interval < 1:
            raise ValueError("validation_interval must be a positive integer")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")


@dataclass
class DataConfig:
    data_root: Path
    labels_csv: Path | None
    labels: list[str]
    folder_labels: dict[str, list[int]]
    image_extensions: list[str]
    folds: int
    batch_size: int
    image_size: int
    num_workers: int
    mean: list[float]
    std: list[float]
    horizontal_flip_probability: float
    rotation_degrees: float

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        if self.labels_csv is not None:
            self.labels_csv = Path(self.labels_csv)
        for name in ("batch_size", "image_size", "folds"):
            value = getattr(self, name)
            if type(value) is not int or value < (2 if name == "folds" else 1):
                raise ValueError(f"Invalid {name}: {value}")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be between 0 and 1")
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be non-empty and unique")
        if any(len(values) != len(self.labels) or any(v not in (0, 1) for v in values)
               for values in self.folder_labels.values()):
            raise ValueError("folder_labels must contain one binary target per label")
        if len(self.mean) != 3 or len(self.std) != 3 or any(v <= 0 for v in self.std):
            raise ValueError("mean and std must have three channels, with positive std")


@dataclass
class ExperimentConfig:
    model: ModelConfig
    trainer: TrainingConfig
    data: DataConfig

    def __post_init__(self):
        if "maxvit_t" in self.model.backbones and self.data.image_size != 224:
            raise ValueError("maxvit_t requires data.image_size: 224")

    def save(self, path):
        values = asdict(self)
        for section, keys in (("data", ("data_root", "labels_csv")),
                              ("trainer", ("output_dir",))):
            for key in keys:
                if values[section][key] is not None:
                    values[section][key] = str(values[section][key])
        Path(path).write_text(yaml.safe_dump(values, sort_keys=False))


def load_config(path=DEFAULT_CONFIG, overrides=None):
    """Load model/trainer/data sections; relative paths use the working directory."""
    values = yaml.safe_load(Path(path).read_text())
    sections = {"model": ModelConfig, "trainer": TrainingConfig, "data": DataConfig}
    if not isinstance(values, dict) or set(values) != set(sections):
        raise ValueError("Configuration must contain model, trainer, and data sections")
    overrides = overrides or {}
    if set(overrides) - set(sections):
        raise ValueError("Overrides must use model, trainer, and data sections")
    configs = {}
    for name, config_class in sections.items():
        if not isinstance(values[name], dict) or not isinstance(overrides.get(name, {}), dict):
            raise ValueError(f"{name} must be a YAML mapping")
        configs[name] = config_class(**{**values[name], **overrides.get(name, {})})
    return ExperimentConfig(**configs)
