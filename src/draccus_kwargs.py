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
        cfg.kwargs = cfg.kwargs or {}

        idx = 0
        while idx < len(extra):
            item = extra[idx]
            if not item.startswith("--"):
                raise ValueError(f"Extra arguments must start with '--', got '{item}'")

            if "=" in item:
                key, value_str = item[2:].split("=", 1)
                key = key.replace("-", "_")
                value = cfgparsing.parse_string(value_str)
                idx += 1
            else:
                raw_key = item[2:]
                key = raw_key.replace("-", "_")
                if idx + 1 < len(extra) and not extra[idx + 1].startswith("--"):
                    value_str = extra[idx + 1]
                    value = cfgparsing.parse_string(value_str)
                    idx += 2
                else:
                    if key.startswith("no_"):
                        key = key[3:]
                        value = False
                    else:
                        value = True
                    idx += 1

            if key in repeatable:
                cfg.kwargs.setdefault(key, []).append(value)
            else:
                cfg.kwargs[key] = value
        return fn(cfg, **kwargs)
    return wrapped
