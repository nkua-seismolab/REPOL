"""Configuration loading and validation for REPOL (config.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from math import isfinite
from pathlib import Path
from types import UnionType
from typing import get_args, get_origin, get_type_hints

import yaml


class ConfigError(Exception):
    """Raised when config.yaml is missing, malformed or fails validation."""


@dataclass
class SeisCompConfig:
    """Options consumed by the scclient module."""

    host: str | None = None
    database: str | None = None
    wait_time: int = 300
    reprocess: bool = False


@dataclass
class SDSConfig:
    archive: str = "/data/archive"


@dataclass
class ModelConfig:
    path: str = "models/DiTingMotionJul.hdf5"


@dataclass
class WaveformConfig:
    t_before: float = 10.0
    t_after: float = 120.0
    target_sampling_rate: float = 100.0
    decimate: bool = True
    filter: list[float] = field(default_factory=list)
    # +- window around the pick fed to the DL model (128 samples at 100 Hz)
    model_half_window: float = 0.64


@dataclass
class SPRatioConfig:
    enabled: bool = True
    use_observed_s: bool = True
    vp: float = 6.0
    vs: float = 3.5
    p_window: float = 2.0
    s_window: float = 10.0
    highpass: float = 1.0
    cut_time: float = 3.0
    pre_pick: float = 5.0
    min_sp_time: float = 0.3


@dataclass
class LoggingConfig:
    level: str = "DEBUG"
    file: str | None = None


@dataclass
class ProcessingConfig:
    # threads for parallel waveform retrieval and S/P measurement
    fetch_workers: int = 4


@dataclass
class Config:
    seiscomp: SeisCompConfig = field(default_factory=SeisCompConfig)
    sds: SDSConfig = field(default_factory=SDSConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    waveform: WaveformConfig = field(default_factory=WaveformConfig)
    spratio: SPRatioConfig = field(default_factory=SPRatioConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)


def _matches_type(value, annotation) -> bool:
    if get_origin(annotation) is UnionType:
        return any(_matches_type(value, option) for option in get_args(annotation))
    if get_origin(annotation) is list:
        return isinstance(value, list) and all(
            _matches_type(item, get_args(annotation)[0]) for item in value
        )
    if annotation is float:
        return type(value) in (int, float) and isfinite(value)
    return type(value) is annotation


def _build_section(cls, name: str, data: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"Unknown keys in '{name}' section: {sorted(unknown)}")
    hints = get_type_hints(cls)
    for key, value in data.items():
        if not _matches_type(value, hints[key]):
            raise ConfigError(f"{name}.{key} has an invalid type or non-finite value")
    return cls(**data)


def _validate(config: Config) -> None:
    if config.logging.level.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ConfigError("logging.level must be DEBUG, INFO, WARNING, ERROR or CRITICAL")
    if config.seiscomp.wait_time < 0:
        raise ConfigError("seiscomp.wait_time must be >= 0")
    if config.waveform.filter and len(config.waveform.filter) != 2:
        raise ConfigError("waveform.filter must be empty or [freqmin, freqmax]")
    if config.waveform.target_sampling_rate <= 0:
        raise ConfigError("waveform.target_sampling_rate must be > 0")
    if config.waveform.model_half_window <= 0:
        raise ConfigError("waveform.model_half_window must be > 0")
    if min(config.waveform.t_before, config.waveform.t_after) < config.waveform.model_half_window:
        raise ConfigError("waveform acquisition must cover model_half_window on both sides")
    if config.waveform.filter and not (
        0
        < config.waveform.filter[0]
        < config.waveform.filter[1]
        < config.waveform.target_sampling_rate / 2
    ):
        raise ConfigError("waveform.filter must be positive, ordered and below Nyquist")
    if not (config.spratio.vp > config.spratio.vs > 0):
        raise ConfigError("spratio requires vp > vs > 0")
    if config.processing.fetch_workers < 1:
        raise ConfigError("processing.fetch_workers must be >= 1")


def load_config(path: str | Path) -> Config:
    """Load and validate a config.yaml file. Missing sections/keys use defaults."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path) as fid:
            raw = yaml.safe_load(fid)
    except yaml.YAMLError:
        raise ConfigError(f"Invalid YAML in {path}") from None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Top level of {path} must be a mapping")

    sections = {f.name: f.default_factory for f in fields(Config)}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ConfigError(f"Unknown top-level sections: {sorted(unknown)}")

    kwargs = {}
    for f in fields(Config):
        data = raw.get(f.name, {})
        if not isinstance(data, dict):
            raise ConfigError(f"Section '{f.name}' must be a mapping")
        section_cls = f.default_factory
        kwargs[f.name] = _build_section(section_cls, f.name, data)

    config = Config(**kwargs)
    _validate(config)
    return config
