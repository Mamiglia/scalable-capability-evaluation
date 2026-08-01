"""Base configuration dataclass and utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type, TypeVar, Union

import draccus

T = TypeVar("T", bound="DraccusConfig")


@dataclass(kw_only=True)
class DraccusConfig:
    name: str
    _kwargs: dict[str, Any] = draccus.field(default_factory=dict)
    
    @property
    def shortname(self) -> str:
        """Return a short name for the config."""
        return self.name.split("/")[-1]

    @classmethod
    def from_cli(cls: Type[T], spec: Union[str, dict, T], *args: Any, **kwargs: Any) -> T:
        """Create a config instance from a CLI argument.
        If spec is a dict, it is used to create the config instance.
        If spec is a path to a config file, it is parsed and used to create the config instance.
        If spec is a config instance, it is returned as is.
        """
        cli_args = kwargs.pop("args", None)
        if isinstance(spec, cls):
            cfg = spec
        elif isinstance(spec, dict):
            cfg = draccus.decode(cls, spec)
        elif isinstance(spec, (str, Path)) and Path(spec).exists():
            cfg = draccus.parse(cls, config_path=Path(spec), args=[] if cli_args is None else cli_args)
        else:        
            cfg = cls(name=str(spec))
            
        if kwargs:
            for k, v in kwargs.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                else:
                    cfg._kwargs[k] = v
        return cfg
