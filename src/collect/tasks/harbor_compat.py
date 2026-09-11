"""Make Harbor-generated compose files load under inspect-ai's strict compose model.

Harbor's task compose files carry `build:` keys inspect-ai does not model (`no_cache`,
`args`, `target`, …). inspect-ai's `ComposeBuild` rejects any unmodelled key, so
`load_tasks("inspect_harbor/…")` raises `ValidationError: Unknown field: 'no_cache'`
before a sample runs. inspect-ai only uses `context`/`dockerfile` from build (it drives
the build itself), so the rest is inert — we strip unmodelled build keys from the compose
dict right before Harbor constructs the `ComposeConfig`.

Imported for its side effect by `collect.cli`; re-import is a no-op.
"""

from inspect_harbor._harbor import converters as _conv
from inspect_ai.util._sandbox.compose import ComposeBuild as _ComposeBuild

_KNOWN_BUILD = set(_ComposeBuild.model_fields)


def _install() -> None:
    real = _conv.ComposeConfig
    if getattr(real, "_harbor_compat_wrapped", False):
        return

    def sanitized(**compose_dict):
        for svc in (compose_dict.get("services") or {}).values():
            build = svc.get("build") if isinstance(svc, dict) else None
            if isinstance(build, dict):
                svc["build"] = {
                    k: v for k, v in build.items() if k in _KNOWN_BUILD or k.startswith("x-")
                }
        return real(**compose_dict)

    sanitized._harbor_compat_wrapped = True
    _conv.ComposeConfig = sanitized


_install()
