"""Base configuration dataclass and utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus

CONFIG_SUFFIXES = (".yml", ".yaml", ".json", ".toml")


@dataclass(kw_only=True)
class DraccusConfig:
    name: str
    _kwargs: dict[str, Any] = draccus.field(default_factory=dict)

    @property
    def shortname(self) -> str:
        """Return a short name for the config."""
        return self.name.split("/")[-1]

    @classmethod
    def from_cli(cls, spec, *args: Any, **kwargs: Any):
        """Create a config instance from a CLI argument.

        ``spec`` may be an instance of ``cls`` (returned as is), a dict (decoded), a path
        to a config file (parsed), or a bare name such as a model slug. A path that looks
        like a config file but does not exist raises instead of being treated as a bare
        name — that mistake silently creates data directories like ``gemma-7b.yml/``.

        Extra keyword arguments override matching fields; the rest land in ``_kwargs``.
        """
        cli_args = kwargs.pop("args", None)
        if isinstance(spec, cls):
            cfg = spec
        elif isinstance(spec, dict):
            cfg = draccus.decode(cls, spec)
        elif isinstance(spec, (str, Path)) and Path(spec).exists():
            # File-only loads need no argparse setup (which rejects Optional types on Python 3.14).
            cfg = (draccus.parse(cls, config_path=Path(spec), args=cli_args)
                   if cli_args else draccus.load(cls, Path(spec)))
        elif str(spec).endswith(CONFIG_SUFFIXES):
            raise FileNotFoundError(f"config file not found: {spec}")
        else:
            cfg = cls(name=str(spec))

        for key, value in kwargs.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                cfg._kwargs[key] = value
        return cfg
