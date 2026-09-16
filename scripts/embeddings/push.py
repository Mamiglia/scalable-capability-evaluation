#!/usr/bin/env python3
"""Upload the local transcript-embedding cache to the private Hugging Face dataset repo.

    python scripts/embeddings/push.py [--root /fast/jyeon/smalljudge_embeddings]
        [--repo LASR-G5/transcript-embeddings]

Runs as an HTCondor CPU job (network) via scripts/embeddings/cpu.sub:
    condor_submit_bid 100 scripts/embeddings/cpu.sub -append 'arguments = scripts/embeddings/push.py'

Writes --root/README.md describing the cache layout, then uploads every .npz plus that README with
huggingface_hub.HfApi.upload_large_folder (resumable; skips files already on the hub).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judging_behavior"))
from hf_download import secret  # noqa: E402  HF_TOKEN from env or /home/jyeon/LASR/.env
from huggingface_hub import HfApi  # noqa: E402

README = """# transcript-embeddings

Qwen3-Embedding-8B vectors of small-judge evaluation transcripts. One file per collection log and
text variant, stored at the log's path with its extension replaced:

    <bench>/<hash>/<model>/<repeat>/<log>.<variant>.npz

Each `.npz` holds these arrays:

    ids        str    sample id
    epochs     int    sample epoch (1 = first attempt)
    correct    bool   evals.sample.extract_score
    n_tokens   int    Qwen3 tokens before chunking
    n_chunks   int    chunks embedded
    embedding  float16 [n, 4096], L2-normalised, token-weighted mean over 8192-token chunks
    meta       str    json: model, revision, max_tokens, overlap, variant

Variants: `full` is the transcript with reasoning included (what a judge shown reasoning sees);
`answer` has the reasoning stripped.

Model: Qwen/Qwen3-Embedding-8B, revision 1d8ad4ca9b3dd8059ad90a75d4983776a23d44af.

Produced by scripts/embeddings/embed.py of github.com/Mamiglia/small-judge and read by
src/protocols/embeddings.py. Source logs: hf.co/datasets/LASR-G5/benchmarks.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=Path("/fast/jyeon/smalljudge_embeddings"))
    parser.add_argument("--repo", default="LASR-G5/transcript-embeddings")
    args = parser.parse_args()
    token = secret("HF_TOKEN")

    (args.root / "README.md").write_text(README)
    api = HfApi(token=token)
    api.upload_large_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(args.root),
                             allow_patterns=["**/*.npz", "README.md"],
                             ignore_patterns=["condor/**", "smoke/**", ".*", "**/.*"])

    local = sum(1 for _ in args.root.rglob("*.npz"))
    remote = len(api.list_repo_files(args.repo, repo_type="dataset"))
    print(f"{local} local .npz files, {remote} files now in {args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
