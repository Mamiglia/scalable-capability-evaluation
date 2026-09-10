"""Make AgentBench-OS's generated Docker build work under rootless podman.

Two incompatibilities, both host-side, neither affecting what a task asks or how it is
scored:

1. The generated compose pins each sandbox to `cpuset: "0-3"`. On a cgroup-v2 host the
   cpuset controller is not delegated to the user slice (loki's `user-1000004.slice` has
   only cpu/memory/pids), so every container start dies with
   `crun: the requested cgroup controller 'cpuset' is not available`. Root fixes it
   properly: `Delegate=cpu cpuset io memory pids` in
   `/etc/systemd/system/user@.service.d/`. Until then, drop the pin.

2. Tasks whose `create.init` is a multi-line script get a `COPY <<'EOF'` heredoc in their
   Dockerfile. Heredocs are a BuildKit feature; buildah 1.33 (podman 4.9) reads the
   `<<'INIT_SCRIPT_EOF'` as a literal path and fails with
   `copier: stat: "/<<INIT_SCRIPT_EOF": no such file or directory`. 100+ of the 125 test
   samples build this way. Rewrite the heredoc as a base64 `RUN`, which is equivalent and
   builds anywhere.

3. Tasks whose `create.init.code` contains a blank line get a bare `RUN ` (the template
   loops over `code.split("\n")`), which fails everywhere with
   `/bin/bash: -c: option requires an argument`. Drop those lines. This one is an upstream
   bug, not a podman limitation.

Imported for its side effect by `collect.cli`; re-import is a no-op.
"""

import base64
import re

from inspect_evals.agent_bench import utils as _utils

_HEREDOC = re.compile(
    r"COPY <<'INIT_SCRIPT_EOF' (?P<dst>\S+)\n(?P<body>.*?)\nINIT_SCRIPT_EOF\n", re.DOTALL
)
_EMPTY_RUN = re.compile(r"^RUN[ \t]*$\n", re.MULTILINE)


def _install() -> None:
    if getattr(_utils.create_docker_image, "_agentbench_compat", False):
        return

    _utils.COMPOSE_YAML_TEMPLATE = _utils.COMPOSE_YAML_TEMPLATE.replace('    cpuset: "0-3"\n', "")

    real = _utils.create_docker_image

    def create_docker_image(record):
        name = real(record)
        path = _utils.get_path(name)
        path.write_text(
            _EMPTY_RUN.sub(
                "",
                _HEREDOC.sub(
                    lambda m: "RUN echo {} | base64 -d > {}\n".format(
                        base64.b64encode(m["body"].encode()).decode(), m["dst"]
                    ),
                    path.read_text(),
                ),
            )
        )
        return name

    create_docker_image._agentbench_compat = True
    _utils.create_docker_image = create_docker_image


_install()
