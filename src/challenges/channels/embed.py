"""Embed channel-filtered transcripts with Qwen3-Embedding-8B, as the embedding protocol reads them.

    uv run python src/challenges/channels/embed.py plan --out jobs.json [--thirds [kept]] # laptop: which logs, items, variants
    python src/challenges/channels/embed.py run jobs.json --out DIR --shard k/n      # GPU: download and embed
    python src/challenges/channels/embed.py check DIR REFERENCE_DIR                 # compare a "full" re-embedding

The embedder is Tommy's scripts/embeddings/embed.py (methodology branch, 68d6db0), unchanged in its settings:
extract_transcript text, tokenised without truncation, 8,192-token chunks with 256 overlap (at most 16), each chunk
ending in <|endoftext|>, last-token pooled, L2-normalised, token-weighted mean, float16. A variant
"channels-<spec>" renders the transcript with JudgeConfig.channels = <spec> (reasoning shown); "thirds-<kept>" renders it
as "full" does and cuts it with judge.dataset.cut(truncate="thirds-<kept>"), the cut of the pairwise thirds runs (plan
--thirds for the removal variants thirds-23/13/12, --thirds kept for the kept-alone thirds-1/2/3); "full" is the existing
variant, re-embedded only to check that this runner reproduces the published vectors. Output:
<out>/<log path without .eval>.<variant>.npz with the arrays protocols.embeddings reads. Samples: the frozen train items
of the semesters the plan covers, first attempts only.
"""

import argparse
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

MODEL = "Qwen/Qwen3-Embedding-8B"
REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
DATA_REPO = "generality-labs/benchmarks"
SPECS = ["drop-reasoning", "drop-text", "drop-calls", "drop-outputs", "drop-tools",
         "only-reasoning", "only-text", "only-calls", "only-outputs", "only-tools"]
# domain -> (semesters, channel specs to embed)
PLAN = {"coding": (["2025-II", "2026-I"], SPECS), "cyber": (["2025-II", "2026-I"], SPECS),
        "math": (["2025-II", "2026-I"], ["only-reasoning"]), "science": (["2025-II", "2026-I"], ["only-reasoning"])}
THIRDS = {"removed": ["thirds-23", "thirds-13", "thirds-12"], "kept": ["thirds-1", "thirds-2", "thirds-3"]}


def plan(out: Path, thirds: str | None = None) -> None:
    """One job per collection log the planned cohorts read: its train items and the variants to embed."""
    import pandas as pd

    from analysis import loading

    jobs: dict[str, dict] = {}
    for domain, (semesters, specs) in (PLAN if thirds is None else {d: (s, THIRDS[thirds]) for d, (s, _) in PLAN.items()}).items():
        for semester in semesters:
            (task,) = loading.tasks(loading.domain(domain), [semester])
            models = pd.concat([task.train.models, task.test.models])
            for bench, items in task.references.items.groupby("bench").item:
                for path in models[f"path_{bench}"].dropna().astype(str):
                    job = jobs.setdefault(path, {"log": path, "items": set(), "variants": set()})
                    job["items"] |= set(map(str, items))
                    job["variants"] |= {spec if thirds else f"channels-{spec}" for spec in specs}
    rows = [{"log": j["log"], "items": sorted(j["items"]), "variants": sorted(j["variants"])} for j in jobs.values()]
    out.write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} logs, {sum(len(r['items']) * len(r['variants']) for r in rows)} item x variant embeddings (upper bound)")


def chunk(ids: list[int], size: int, overlap: int, max_chunks: int) -> list[list[int]]:
    if len(ids) <= size:
        return [ids]
    return [ids[i:i + size] for i in range(0, len(ids), size - overlap)][:max_chunks]


class Embedder:
    """Tommy's embedder (scripts/embeddings/embed.py @ 68d6db0), settings unchanged."""

    def __init__(self, max_tokens=8192, overlap=256, max_chunks=16, tokens_per_batch=32768, device="cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.max_tokens, self.overlap, self.max_chunks, self.tokens_per_batch = max_tokens, overlap, max_chunks, tokens_per_batch
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
        self.eos = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.device = device
        self.model = AutoModel.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16).to(device).eval()
        self.dim = self.model.config.hidden_size

    def __call__(self, texts: list[str]):
        torch = self.torch
        encoded = self.tok(texts, add_special_tokens=True, truncation=False)["input_ids"]
        n_tokens = np.array([len(ids) for ids in encoded], dtype=np.int32)
        flat, owner = [], []
        for i, ids in enumerate(encoded):
            for piece in chunk(ids, self.max_tokens - 1, self.overlap, self.max_chunks):
                flat.append(piece if piece and piece[-1] == self.eos else piece + [self.eos])
                owner.append(i)
        owner = np.asarray(owner)
        embeddings = np.zeros((len(flat), self.dim), dtype=np.float32)
        order = np.argsort([len(x) for x in flat], kind="stable")
        pad = self.tok.pad_token_id or 0
        start = 0
        with torch.inference_mode():
            while start < len(order):
                end = start
                while end < len(order) and (end - start + 1) * len(flat[order[end]]) <= self.tokens_per_batch:
                    end += 1
                end = max(end, start + 1)
                batch = [flat[j] for j in order[start:end]]
                length = max(len(x) for x in batch)
                inp = torch.full((len(batch), length), pad, dtype=torch.long)
                att = torch.zeros((len(batch), length), dtype=torch.long)
                for i, ids in enumerate(batch):  # left padding
                    inp[i, length - len(ids):] = torch.tensor(ids)
                    att[i, length - len(ids):] = 1
                out = self.model(input_ids=inp.to(self.device), attention_mask=att.to(self.device),
                                 use_cache=False).last_hidden_state[:, -1]
                embeddings[order[start:end]] = torch.nn.functional.normalize(out.float(), dim=-1).cpu().numpy()
                start = end
        weights = np.array([len(x) for x in flat], dtype=np.float32)
        mean = np.zeros((len(texts), self.dim), dtype=np.float32)
        n_chunks = np.zeros(len(texts), dtype=np.int32)
        for i in range(len(texts)):
            member = owner == i
            n_chunks[i] = member.sum()
            mean[i] = (embeddings[member] * weights[member, None]).sum(0) / weights[member].sum()
            mean[i] /= max(np.linalg.norm(mean[i]), 1e-12)
        return mean.astype(np.float16), n_tokens, n_chunks


def samples(log: Path, items: set[str]):
    """First-attempt samples of the log on `items`, one archive member at a time."""
    from inspect_ai.log import EvalSample

    with zipfile.ZipFile(log) as archive:
        for member in sorted(m for m in archive.namelist() if m.startswith("samples/") and m.endswith(".json")):
            sample = EvalSample.model_validate_json(archive.read(member))
            if str(sample.id) in items and int(getattr(sample, "epoch", 1) or 1) == 1:
                yield sample


def render(sample, variant: str) -> str:
    """The text embedded for `variant`: "full", "channels-<spec>" or "thirds-<kept>" (module docstring)."""
    from evals.sample import extract_transcript
    from judge.dataset import cut

    if variant.startswith("thirds-"):
        return cut(extract_transcript(sample, include_reasoning=True, channels=""), 0, variant)
    return extract_transcript(sample, include_reasoning=True, channels="" if variant == "full" else variant.removeprefix("channels-"))


def run(jobs_path: Path, out: Path, shard: str, variants: list[str] | None, limit_samples: int, revision: str) -> None:
    from huggingface_hub import hf_hub_download

    from evals.sample import extract_score

    k, n = map(int, shard.split("/"))
    jobs = json.loads(jobs_path.read_text())[k::n]
    embedder = Embedder()
    meta = {"model": MODEL, "revision": REVISION, "max_tokens": 8192, "overlap": 256, "max_chunks": 16}
    for number, job in enumerate(jobs):
        wanted = variants or job["variants"]
        targets = {v: out / (job["log"].rsplit(".eval", 1)[0] + f".{v}.npz") for v in wanted}
        targets = {v: p for v, p in targets.items() if not p.exists()}
        if not targets:
            continue
        started = time.time()
        log = Path(hf_hub_download(DATA_REPO, job["log"], repo_type="dataset", revision=revision))
        rows = []
        for sample in samples(log, set(job["items"])):
            texts = {v: render(sample, v) for v in targets}
            rows.append((str(sample.id), int(getattr(sample, "epoch", 1) or 1), bool(extract_score(sample)), texts))
            if limit_samples and len(rows) >= limit_samples:
                break
        for variant, path in targets.items():
            texts = [r[3][variant] for r in rows]
            embedding, n_tokens, n_chunks = embedder(texts) if texts else (
                np.zeros((0, embedder.dim), np.float16), np.zeros(0, np.int32), np.zeros(0, np.int32))
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".npz")
            os.close(fd)
            np.savez(tmp, ids=np.array([r[0] for r in rows], dtype=str), epochs=np.array([r[1] for r in rows], np.int32),
                     correct=np.array([r[2] for r in rows], bool), n_tokens=n_tokens, n_chunks=n_chunks,
                     embedding=embedding, meta=np.array(json.dumps(meta | {"variant": variant})))
            os.replace(tmp, path)
        print(f"[{shard} {number + 1}/{len(jobs)}] {job['log']}: {len(rows)} samples x {len(targets)} variants, "
              f"{time.time() - started:.0f}s", flush=True)
        log.unlink(missing_ok=True)  # the HF cache copy; keeps the pod disk small


def check(new_root: Path, reference_root: Path) -> None:
    """Cosine between re-embedded "full" vectors and the published ones, per shared sample."""
    cosines = []
    for new in sorted(new_root.rglob("*.full.npz")):
        reference = reference_root / new.relative_to(new_root)
        if not reference.exists():
            continue
        a, b = np.load(new), np.load(reference)
        index = {(i, e): row for row, (i, e) in enumerate(zip(b["ids"], b["epochs"]))}
        for row, key in enumerate(zip(a["ids"], a["epochs"])):
            if key in index:
                u, v = a["embedding"][row].astype(np.float32), b["embedding"][index[key]].astype(np.float32)
                cosines.append(float(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
    print(f"{len(cosines)} samples, cosine min {min(cosines):.5f} median {np.median(cosines):.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan"); p.add_argument("--out", type=Path, required=True); p.add_argument("--thirds", nargs="?", const="removed", choices=list(THIRDS))
    r = sub.add_parser("run"); r.add_argument("jobs", type=Path); r.add_argument("--out", type=Path, required=True)
    r.add_argument("--shard", default="0/1"); r.add_argument("--variants", nargs="*"); r.add_argument("--limit-samples", type=int, default=0)
    r.add_argument("--revision", required=True, help="generality-labs/benchmarks revision the frozen tables point at (git -C data rev-parse HEAD)")
    c = sub.add_parser("check"); c.add_argument("new", type=Path); c.add_argument("reference", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        plan(args.out, args.thirds)
    elif args.command == "run":
        run(args.jobs, args.out, args.shard, args.variants, args.limit_samples, args.revision)
    else:
        check(args.new, args.reference)
