"""Base configuration dataclass and utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type, TypeVar, Union

import draccus

T = TypeVar("T", bound="DraccusConfig")


@dataclass
class DraccusConfig:
    name: str = ""

    @classmethod
    def from_cli(cls: Type[T], spec: Union[str, dict, T], args: Any = None, **kwargs: Any) -> T:
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, dict):
            return draccus.decode(cls, spec)
        path = Path(spec)
        if path.exists():
            return draccus.parse(cls, config_path=path, args=args, **kwargs)
        return cls(name=str(spec))
