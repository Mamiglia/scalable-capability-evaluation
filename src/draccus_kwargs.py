from typing import get_type_hints
from draccus import cfgparsing
from draccus.argparsing import ArgumentParser

def wrap(fn):
    def wrapped(*args, **kwargs):
        if args: return fn(*args, **kwargs)
        cls = next(iter(get_type_hints(fn).values())); cfg, extra = ArgumentParser(cls).parse_known_args()
        if len(extra) % 2 or any(not key.startswith("--") for key in extra[::2]): raise ValueError("Extra arguments must be --key value pairs")
        cfg.kwargs = cfg.kwargs or {}; cfg.kwargs.update({key[2:]: cfgparsing.parse_string(value) for key, value in zip(extra[::2], extra[1::2])}); return fn(cfg, **kwargs)
    return wrapped
