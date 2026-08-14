"""Judge-side extraction of the frozen GPQA capability SAE features.

The 32 SAE features in the capability evidence (sae_s0_000..031 in
features_wimhf8) are the latents of the upstream WIMHF SparseAutoencoder
(rmovva/wimhf @ 13f705ea, unmodified), M=32 K=4 prefixes [8, 32], seed 0,
trained by the GPQA capability analysis with the extended schedule on
question-residualised Qwen3-Embedding-8B embeddings of the answer_masked view.
The frozen SAE implementation, checkpoint, question means, and capability
evidence are in this repository under scripts/hibayes. The embedding input is
Qwen3-Embedding's instruct format with the question included:

    Instruct: <task>
    Query: Question: <question>

    Response: <answer_masked view text>

On the capability side <question> is the record's own prompt. Capability
prompts differ across runs for the same question (option order and format), so
there is no canonical capability prompt to reuse; the judge side uses the
question exactly as the judge prompt presents it in its "Input question"
block, which is the text the judge actually read.

Two production stages mirror embed_instructed.py and wimhf_export.py:

  --stage embed     GPU. Embed every unique (question, view text) with
                    Qwen3-Embedding-8B: chunk at 8192 tokens with 256 overlap,
                    left padding, last-token pool, L2-normalise per chunk,
                    token-length-weighted mean over chunks, fp16.
  --stage features  CPU. Residualise with the emb8 capability-corpus
                    per-question means, encode through the seed-0 upstream
                    checkpoint, difference major minus minor, and write
                    analysis/hibayes/features_sae_judge/ with one marginal
                    config per latent.

  --stage check     CPU. Verify fixed assets, the checkpoint, and any existing
                    judge-embedding store before a feature run.

The per-question means come from the capability corpus, not from the judge
pairs: they are the residualisation the SAE was trained against, and GPQA
question ids match between the corpora exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
HIB = ROOT / "analysis" / "hibayes"
OUT = HIB / "features_sae_judge"
STORE = HIB / "sae_judge_embeddings"
EMBEDDINGS = STORE / "unique_embeddings.npy"
EMBEDDING_ROWS = STORE / "rows.parquet"
EMBEDDING_MANIFEST = STORE / "manifest.json"
ASSETS = pathlib.Path(__file__).with_name("sae_assets")
CKPT = ASSETS / "sae_M32_K4_s0.pt"
QUESTION_MEANS = ASSETS / "question_means.npz"
EVIDENCE = pathlib.Path(__file__).with_name("learned_capability_features.csv")
MODEL = "Qwen/Qwen3-Embedding-8B"
MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
M, K, PREFIXES = 32, 4, [8, 32]
MAX_TOKENS = 8192
CHUNK_OVERLAP = 256
ASSET_SHA256 = {
    ASSETS / "wimhf_sae.py": "caf97042972d2f60a46e2263b3d4252fecdf7d1cffdf0a6c30b8e913c7518622",
    CKPT: "a26a214dca3848420fe3d1623e76f05bf683bd3ff7ec8c63f5b4f93479c2662e",
    QUESTION_MEANS: "72c14a06944436dfee119471e1da9ad601737e34941558f18a279b5d41739423",
    EVIDENCE: "ca617a2aaf6d94a85ac9365b17f245048dd0af5cf48b9dd7eaad926bfe268118",
}

# Copied verbatim from gpqa_tfa embed_instructed.py.
TASK = ("Represent this scientific question and the model's response so that "
        "differences in reasoning approach, uncertainty, verification, "
        "structure, and explanatory style are preserved.")


def build_text(question: str, response: str) -> str:
    return (f"Instruct: {TASK}\n"
            f"Query: Question: {question}\n\nResponse: {response}")


def chunk(ids: list[int], size: int, overlap: int) -> list[list[int]]:
    if len(ids) <= size:
        return [ids]
    step = size - overlap
    return [ids[i:i + size] for i in range(0, len(ids), step) if i < len(ids)]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_embedding_store(
    embeddings: np.ndarray, rows: pd.DataFrame, *, n_texts: int, n_chunks: int
) -> None:
    """Write the related embedding files and publish their manifest last."""
    STORE.mkdir(parents=True, exist_ok=True)
    temporary: list[pathlib.Path] = []
    try:
        for suffix in (".npy", ".parquet", ".json"):
            with tempfile.NamedTemporaryFile(dir=STORE, suffix=suffix, delete=False) as handle:
                temporary.append(pathlib.Path(handle.name))
        emb_tmp, rows_tmp, manifest_tmp = temporary
        np.save(emb_tmp, embeddings)
        rows.to_parquet(rows_tmp, index=False)
        manifest = {
            "version": 1,
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "format": "instruct+question+response",
            "texts": n_texts,
            "chunks": n_chunks,
            "rows": len(rows),
            "dimension": embeddings.shape[1],
            "embeddings_sha256": sha256(emb_tmp),
            "rows_sha256": sha256(rows_tmp),
        }
        manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        emb_tmp.replace(EMBEDDINGS)
        rows_tmp.replace(EMBEDDING_ROWS)
        manifest_tmp.replace(EMBEDDING_MANIFEST)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def load_embedding_store() -> tuple[np.ndarray, pd.DataFrame]:
    """Load one validated judge-embedding run."""
    manifest = json.loads(EMBEDDING_MANIFEST.read_text())
    if manifest.get("model") != MODEL or manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError(f"unexpected embedding model in {EMBEDDING_MANIFEST}")
    if sha256(EMBEDDINGS) != manifest.get("embeddings_sha256"):
        raise ValueError(f"{EMBEDDINGS} does not match {EMBEDDING_MANIFEST}")
    if sha256(EMBEDDING_ROWS) != manifest.get("rows_sha256"):
        raise ValueError(f"{EMBEDDING_ROWS} does not match {EMBEDDING_MANIFEST}")
    embeddings = np.load(EMBEDDINGS).astype(np.float32)
    rows = pd.read_parquet(EMBEDDING_ROWS)
    if (sha256(EMBEDDINGS) != manifest.get("embeddings_sha256")
            or sha256(EMBEDDING_ROWS) != manifest.get("rows_sha256")):
        raise ValueError(f"embedding files changed while loading {EMBEDDING_MANIFEST}")
    indices = rows[["major_idx", "minor_idx"]].to_numpy()
    if (embeddings.shape != (manifest.get("texts"), manifest.get("dimension"))
            or len(rows) != manifest.get("rows") or len(rows) == 0
            or indices.min() < 0 or indices.max() >= len(embeddings)):
        raise ValueError(f"invalid embedding store described by {EMBEDDING_MANIFEST}")
    return embeddings, rows


def stage_embed(batch_size: int) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    from tfidf_features import collect_texts

    rows = collect_texts(with_question=True)
    print(f"judged samples: {len(rows):,}", flush=True)
    uniq: dict[str, int] = {}
    for _, q, a, b in rows:
        for t in (a, b):
            uniq.setdefault(build_text(q, t), len(uniq))
    texts = sorted(uniq, key=uniq.get)
    print(f"unique instructed texts: {len(texts):,}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(
        MODEL, revision=MODEL_REVISION, padding_side="left"
    )
    model = AutoModel.from_pretrained(
        MODEL, revision=MODEL_REVISION,
        dtype=torch.bfloat16 if dev == "cuda" else torch.float32
    ).to(dev).eval()
    dim = model.config.hidden_size
    print(f"device {dev}  dim {dim}", flush=True)

    enc = tok(texts, add_special_tokens=True, truncation=False)["input_ids"]
    flat, owner = [], []
    for i, ids in enumerate(enc):
        for c in chunk(ids, MAX_TOKENS, CHUNK_OVERLAP):
            flat.append(c)
            owner.append(i)
    print(f"chunks {len(flat):,}", flush=True)

    @torch.inference_mode()
    def encode(batch_ids):
        L = max(len(x) for x in batch_ids)
        pad = tok.pad_token_id or 0
        inp = torch.full((len(batch_ids), L), pad, dtype=torch.long)
        att = torch.zeros((len(batch_ids), L), dtype=torch.long)
        for i, ids in enumerate(batch_ids):          # left padding
            inp[i, L - len(ids):] = torch.tensor(ids)
            att[i, L - len(ids):] = 1
        out = model(input_ids=inp.to(dev), attention_mask=att.to(dev)).last_hidden_state
        emb = out[:, -1]                              # last-token pooling
        return torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy().astype(np.float16)

    embs = np.zeros((len(flat), dim), dtype=np.float16)
    order = np.argsort([len(x) for x in flat])
    for s in range(0, len(order), batch_size):
        sel = order[s:s + batch_size]
        embs[sel] = encode([flat[j] for j in sel])
        if (s // batch_size) % 50 == 0:
            print(f"  encoded {min(s + batch_size, len(order)):,}/{len(order):,}",
                  flush=True)

    owner = np.asarray(owner)
    weights = np.asarray([len(x) for x in flat], dtype=np.float32)
    mean_e = np.zeros((len(texts), dim), dtype=np.float32)
    for i in range(len(texts)):
        m = owner == i
        e = embs[m].astype(np.float32)
        w = weights[m]
        mean_e[i] = (e * w[:, None]).sum(0) / max(w.sum(), 1e-6)

    embedding_rows = pd.DataFrame(
        [(u, uniq[build_text(q, a)], uniq[build_text(q, b)]) for u, q, a, b in rows],
        columns=["sample_uuid", "major_idx", "minor_idx"],
    )
    write_embedding_store(
        mean_e.astype(np.float16), embedding_rows, n_texts=len(texts), n_chunks=len(flat)
    )
    print(f"wrote validated embedding store under {STORE}")


def question_means() -> pd.DataFrame:
    """Load the frozen per-question means from the repository."""
    with np.load(QUESTION_MEANS) as data:
        questions = data["question_ids"].astype(str)
        means = data["means"].astype(np.float32)
    if len(questions) != len(set(questions)) or means.shape != (198, 4096):
        raise ValueError(f"invalid SAE question means in {QUESTION_MEANS}: {means.shape}")
    return pd.DataFrame(means, index=questions)


def check_hashes() -> None:
    for path, expected in ASSET_SHA256.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{path} has SHA-256 {actual}, expected {expected}")


def stage_check() -> None:
    """Check the local assets and one checkpoint inference."""
    check_hashes()
    means = question_means()
    evidence = pd.read_csv(EVIDENCE)
    latents = sorted(evidence.loc[evidence["method"] == "SAE", "feature"].unique())
    expected_latents = [f"sae_s0_{i:03d}" for i in range(M)]
    if latents != expected_latents:
        raise ValueError(f"{EVIDENCE} does not contain the expected seed-0 SAE latents")

    import torch
    from sae_assets.wimhf_sae import SparseAutoencoder

    sae = SparseAutoencoder(means.shape[1], M, K, prefix_lengths=PREFIXES)
    sae.load_state_dict(torch.load(CKPT, map_location="cpu"))
    activations = sae.get_activations(means.iloc[:2].to_numpy(), show_progress=False)
    if activations.shape != (2, M) or not np.isfinite(activations).all():
        raise ValueError(f"invalid checkpoint activations: {activations.shape}")
    message = f"SAE ASSETS OK: {len(means)} questions, {len(latents)} latents"
    if EMBEDDING_MANIFEST.exists():
        embeddings, rows = load_embedding_store()
        message += f", {len(rows)} judge rows, {len(embeddings)} unique texts"
    print(message)


def marginal_config(predictor: str) -> dict:
    """Build the standalone fit config for one SAE latent."""
    from fit_features import POSITION, build_config

    csv = pathlib.Path("analysis/hibayes/features_sae_judge/complete_cases.csv")
    config = build_config(csv, [predictor, POSITION], "is_correct")
    config["model"]["models"][0]["config"]["pool_judge_effects"] = True
    del config["communicate"]["communicators"][0]["judge_feature_report"]["make_plots"]
    return config


def stage_features() -> None:
    import torch

    from sae_assets.wimhf_sae import SparseAutoencoder

    check_hashes()
    mean_e, rows = load_embedding_store()
    keys = pd.read_csv(HIB / "gpqa_correctness.csv", low_memory=False,
                       usecols=["sample_uuid", "is_correct", "judge", "item",
                                "model_pair", "transcript_condition",
                                "major_is_right", "self_comparison"])
    merged = keys.merge(rows, on="sample_uuid", how="inner")
    print(f"rows after join: {len(merged):,} (csv has {len(keys):,})")
    assert len(merged) == len(keys), "judge rows missing from the embedding store"

    qm = question_means()
    print(f"question means: {len(qm)} questions, dim {qm.shape[1]}")
    missing = set(merged["item"]) - set(qm.index)
    if missing:
        raise ValueError(f"{len(missing)} judge items have no capability-corpus "
                         f"question mean, e.g. {sorted(missing)[:3]}")
    Q = qm.loc[merged["item"]].to_numpy()
    Xa = mean_e[merged["major_idx"].to_numpy()] - Q
    Xb = mean_e[merged["minor_idx"].to_numpy()] - Q

    sae = SparseAutoencoder(mean_e.shape[1], M, K, prefix_lengths=PREFIXES)
    sae.load_state_dict(torch.load(CKPT, map_location="cpu"))
    Za = sae.get_activations(Xa, show_progress=False)
    Zb = sae.get_activations(Xb, show_progress=False)
    print(f"mean active latents: major {(np.abs(Za) > 0).sum(1).mean():.2f}  "
          f"minor {(np.abs(Zb) > 0).sum(1).mean():.2f}")

    ev = pd.read_csv(EVIDENCE)
    latents = sorted(ev[ev["method"] == "SAE"]["feature"].unique())
    idx = [int(name.rsplit("_", 1)[1]) for name in latents]
    print(f"evidence latents: {len(latents)}")

    for name, j in zip(latents, idx):
        merged[f"d_{name}"] = Za[:, j] - Zb[:, j]

    contested = merged["self_comparison"] == 0
    parts, dropped = {}, []
    for col in [c for c in merged.columns if c.startswith("d_sae")]:
        scale = merged.loc[contested, col].std()
        if not np.isfinite(scale) or scale == 0:
            dropped.append(col)
            continue
        scaled = merged[col] / scale
        pair_mean = scaled.groupby(merged["model_pair"]).transform("mean")
        parts[f"{col}_between"] = pair_mean
        parts[f"{col}_within"] = scaled - pair_mean
    merged = pd.concat(
        [merged.drop(columns=["major_idx", "minor_idx"]),
         pd.DataFrame(parts, index=merged.index)], axis=1)
    print(f"constant latents dropped: {len(dropped)}")

    OUT.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT / "complete_cases.csv", index=False)
    import yaml

    written = 0
    for name in latents:
        col = f"d_{name}_within"
        if f"d_{name}" in dropped or col not in merged.columns:
            continue
        config = marginal_config(col)
        (OUT / f"marginal__sae__d_{name}.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False)
        )
        written += 1
    print(f"wrote complete_cases.csv and {written} configs to {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["check", "embed", "features"], required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    if args.stage == "check":
        stage_check()
    elif args.stage == "embed":
        stage_embed(args.batch_size)
    else:
        stage_features()


if __name__ == "__main__":
    main()
