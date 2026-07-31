from functools import wraps
import sys
from typing import get_type_hints
from draccus import cfgparsing
from draccus.argparsing import ArgumentParser

def wrap(fn=None, *, repeatable=(), config_path_aliases=()):
    if fn is None:
        return lambda wrapped_fn: wrap(
            wrapped_fn,
            repeatable=repeatable,
            config_path_aliases=config_path_aliases,
        )

    @wraps(fn)
    def wrapped(*args, **kwargs):
        if args: return fn(*args, **kwargs)
        cls = next(iter(get_type_hints(fn).values()))
        argv = sys.argv[1:]
        for name in config_path_aliases:
            option = f"--{name}"
            if option in argv:
                index = argv.index(option)
                if index + 1 == len(argv):
                    raise ValueError(f"{option} requires a config path")
                argv[index] = "--config_path"
        cfg, extra = ArgumentParser(cls).parse_known_args(argv)
        if len(extra) % 2 or any(not key.startswith("--") for key in extra[::2]): raise ValueError("Extra arguments must be --key value pairs")
        cfg.kwargs = cfg.kwargs or {}
        for key, value in zip(extra[::2], extra[1::2]):
            key = key[2:]
            value = cfgparsing.parse_string(value)
            if key in repeatable:
                cfg.kwargs.setdefault(key, []).append(value)
            else:
                cfg.kwargs[key] = value
        return fn(cfg, **kwargs)
    return wrapped
