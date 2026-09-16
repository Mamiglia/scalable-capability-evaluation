#!/usr/bin/env python3
"""Read-only check of the Hugging Face repos this pipeline uses (prints the username, never the token).

    python scripts/embeddings/hf_check.py [--create]   # --create: create the embeddings repo (private) if absent
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judging_behavior"))
from hf_download import secret  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

EMBEDDINGS_REPO = "LASR-G5/transcript-embeddings"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()
    api = HfApi(token=secret("HF_TOKEN"))
    print("user", api.whoami()["name"])
    visible = sorted(d.id for d in api.list_datasets(author="LASR-G5"))
    print("LASR-G5 datasets visible to this token:", visible)
    if EMBEDDINGS_REPO not in visible and args.create:
        print("creating", api.create_repo(EMBEDDINGS_REPO, repo_type="dataset", private=True, exist_ok=True))
    try:
        info = api.dataset_info(EMBEDDINGS_REPO, files_metadata=True)
        print(EMBEDDINGS_REPO, "private" if info.private else "public", "sha", info.sha, "files", len(info.siblings))
    except Exception as error:  # noqa: BLE001
        print(EMBEDDINGS_REPO, "not accessible:", type(error).__name__, str(error).splitlines()[0])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
