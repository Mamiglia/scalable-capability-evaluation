import threading


class Multiton(type):
    """Create one instance of a class per key."""

    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, key, *args, **kwargs):
        instance_key = (cls, key)
        with cls._lock:
            if instance_key not in cls._instances:
                cls._instances[instance_key] = super().__call__(key, *args, **kwargs)
            return cls._instances[instance_key]
