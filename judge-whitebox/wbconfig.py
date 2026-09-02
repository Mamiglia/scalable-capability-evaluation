import os
import pathlib
import yaml


def cfg(path: str | None = None) -> dict:
    p = pathlib.Path(path or pathlib.Path(__file__).parent / "config.yaml")
    c = yaml.safe_load(p.read_text())
    root = c["paths"]["hpc_root"] if os.environ.get("WB_HPC") else c["paths"]["local_root"]
    for k, v in c["paths"].items():
        if isinstance(v, str):
            formatted = v.format(root=root) if "{root}" in v else v
            if formatted.startswith("/"):
                c["paths"][k] = formatted
            else:
                c["paths"][k] = str((p.parent / formatted).resolve())
    # WB_PAIRS overrides paths.pairs — alternate pair universes without touching config.
    if os.environ.get("WB_PAIRS"):
        c["paths"]["pairs"] = os.path.abspath(os.environ["WB_PAIRS"])
    return c


def resolve_model(model_name: str, c: dict) -> str:
    """Model identifier for AutoModel/AutoTokenizer.from_pretrained: a local dir under
    models.hpc_dir when WB_HPC is set (cluster nodes run HF_HUB_OFFLINE=1 and can't resolve a
    hub id), else the models.hf_ids hub id."""
    if os.environ.get("WB_HPC"):
        return str(pathlib.Path(c["models"]["hpc_dir"]) / model_name)
    return c["models"]["hf_ids"][model_name]
