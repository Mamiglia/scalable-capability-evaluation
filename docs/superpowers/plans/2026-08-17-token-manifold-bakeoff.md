# Token-manifold recorder and method bake-off — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the smallest trustworthy pipeline that records per-token residual states from base
Qwen3-8B/Llama-3B, locates logit-gap information over layer × token time, and compares CEBRA,
block-sparse and SAE/Ising discovery on a bounded discovery-fold bake-off.

**Architecture:** Two focused modules. `token_manifold_record.py` owns deterministic item folds,
token metadata, storage estimates and immutable activation shards. `token_manifold_discover.py` owns
shard loading, grouped logit-gap decoding and thin adapters around pinned upstream discovery methods.
The first empirical gate ends with a small CSV comparison; causal interventions, naming and full runs
remain Plan 2 work.

**Tech Stack:** Python 3.12, PyTorch, transformers, pandas, NumPy, scikit-learn; pinned CEBRA v0.6.0,
Goodfire block-sparse-featurizer, and Goodfire sae-manifold.

**Spec:** `docs/superpowers/specs/2026-08-17-output-conditioned-token-manifolds-design.md`

## Global constraints

- Discovery uses base Qwen3-8B and base Llama-3.2-3B-Instruct only.
- The sole discovery target is each model's continuous signed `l1 - l2` logit gap.
- No hand-authored semantic label enters recording, fitting, ranking or method selection.
- Both orders and every row for one GPQA item remain in the same fold.
- Exact tokenizer positions are recorded; existing `token_ablate.token_groups` supplies structural spans.
- Prompt sampling is equal-weighted; longer prompts do not become extra independent observations.
- New data outputs are immutable and written through a temporary file plus atomic rename.
- No paid API, no RunPod, and no SWC submission in this plan.
- No heavy process or package install on the SWC gateway; the user reviews storage before any launch.
- Ceph files are never overwritten or deleted.
- Work inline without subagents, per the user's choice.

---

### Task 1: Pin the method environment and deterministic outer folds

**Files:**
- Create: `judge-whitebox/requirements-manifold.txt`
- Create: `judge-whitebox/token_manifold_record.py`
- Create: `judge-whitebox/tests/test_token_manifold_record.py`
- Modify: `judge-whitebox/config.yaml:9-15`

**Interfaces:**
- Consumes: pairs DataFrame from `wbpairs.load_pairs`.
- Produces: `balanced_item_folds(df: pd.DataFrame, seed: int) -> dict[str, str]` with values `A|B`.
- Produces: `select_fold(df, folds, name) -> pd.DataFrame` preserving input order.

- [ ] **Step 1: Add exact dependency pins**

```text
cebra @ git+https://github.com/AdaptiveMotorControlLab/CEBRA.git@493ed252e544b89aa0cff2c1e7af34d7f80261ba
bsf @ git+https://github.com/goodfire-ai/block-sparse-featurizer.git@219f121ea82d2b19200d1dac918396e6058d7eb9
sae-manifold-eval @ git+https://github.com/goodfire-ai/sae-manifold.git@f2632ddbb25f4c670254ceea999058b9ba4b0450
pyarrow>=25.0.0
pytest>=9.1.1
PyYAML>=6.0
```

- [ ] **Step 2: Write failing fold tests**

```python
def test_balanced_item_folds_are_deterministic_disjoint_and_row_balanced():
    df = pd.DataFrame({"item": ["heavy"] * 8 + ["a"] * 4 + ["b"] * 4 + ["c"] * 2})
    one = record.balanced_item_folds(df, seed=7)
    two = record.balanced_item_folds(df.sample(frac=1, random_state=9), seed=7)
    assert one == two
    assert set(one) == set(df.item)
    assert set(one.values()) == {"A", "B"}
    counts = df.assign(fold=df.item.map(one)).groupby("fold").size()
    assert abs(int(counts.A) - int(counts.B)) <= 2


def test_select_fold_keeps_whole_items_and_original_order():
    df = pd.DataFrame({"item": ["a", "a", "b", "b"], "row": range(4)})
    out = record.select_fold(df, {"a": "B", "b": "A"}, "A")
    assert out.row.tolist() == [2, 3]
    assert set(out.item) == {"b"}
```

Put these concrete helpers above the tests in `test_token_manifold_record.py`:

```python
def fake_row(uid="p0"):
    return {"pair_uid": uid, "order": 0, "item": "it0", "label": 1,
            "question": "Q", "options_1": "", "options_2": "",
            "text_1": "abc", "text_2": "de"}


def fake_cfg():
    return {"token_ablate": {"markers": {}}}


class OffsetTokenizer:
    def __call__(self, text, return_offsets_mapping=False, return_tensors=None):
        ids = [1 + i % 30 for i in range(len(text))]
        out = {"input_ids": torch.tensor([ids]) if return_tensors else ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


LengthTokenizer = OffsetTokenizer
```

- [ ] **Step 3: Run tests and confirm RED**

Run: `uv run pytest judge-whitebox/tests/test_token_manifold_record.py -q`
Expected: import or attribute failure for `balanced_item_folds`.

- [ ] **Step 4: Implement the minimum deterministic greedy split**

```python
def balanced_item_folds(df, seed):
    counts = df.groupby("item", sort=True).size()
    rng = np.random.default_rng(seed)
    tie = {item: value for item, value in zip(counts.index, rng.random(len(counts)))}
    ordered = sorted(counts.index, key=lambda item: (-int(counts[item]), tie[item], str(item)))
    totals = {"A": 0, "B": 0}
    folds = {}
    for item in ordered:
        fold = min(totals, key=lambda name: (totals[name], name))
        folds[item] = fold
        totals[fold] += int(counts[item])
    return folds


def select_fold(df, folds, name):
    assert name in {"A", "B"}
    assert set(df["item"]) <= set(folds)
    return df[df["item"].map(folds) == name].copy()
```

- [ ] **Step 5: Add the missing local Llama model mapping**

```yaml
    Llama-3.2-3B-Instruct: meta-llama/Llama-3.2-3B-Instruct
```

- [ ] **Step 6: Run task tests and the existing config tests**

Run: `uv run pytest judge-whitebox/tests/test_token_manifold_record.py judge-whitebox/tests/test_wbconfig.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add judge-whitebox/requirements-manifold.txt judge-whitebox/token_manifold_record.py \
  judge-whitebox/tests/test_token_manifold_record.py judge-whitebox/config.yaml
git commit -m "token manifolds: pin methods and item folds"
```

---

### Task 2: Token metadata, layer selection and storage estimate

**Files:**
- Modify: `judge-whitebox/token_manifold_record.py`
- Modify: `judge-whitebox/tests/test_token_manifold_record.py`

**Interfaces:**
- Consumes: `wbprompts.render_verdict_prompt`, `token_ablate.token_groups`.
- Produces: `layer_indices(n_states: int, n_select: int = 12) -> tuple[int, ...]`.
- Produces: `prompt_token_meta(row, tokenizer, c) -> dict[str, object]`.
- Produces: `estimate_storage(df, tokenizer, c, layers, d_model, dtype_bytes=2) -> dict`.

- [ ] **Step 1: Write failing layer and metadata tests**

```python
def test_layer_indices_include_ends_and_are_unique():
    assert record.layer_indices(33, 4) == (0, 11, 21, 32)
    assert record.layer_indices(3, 12) == (0, 1, 2)


def test_prompt_token_meta_aligns_ids_offsets_and_structural_groups(monkeypatch):
    row, tok, cfg = fake_row(), OffsetTokenizer(), fake_cfg()
    monkeypatch.setattr(record.wbprompts, "render_verdict_prompt", lambda row, tok: "QABCD")
    monkeypatch.setattr(record.token_ablate, "token_groups", lambda row, tok, c: {
        "question": [0], "transcript_1": [1], "transcript_2": [2],
        "instr_format": [3], "slot_labels": [], "answer_readout": [4],
    })
    meta = record.prompt_token_meta(row, tok, cfg)
    assert len(meta["input_ids"]) == len(meta["offsets"]) == len(meta["group_id"]) == 5
    assert meta["group_names"][meta["group_id"][1]] == "transcript_1"
```

- [ ] **Step 2: Write a failing storage-estimate test**

```python
def test_estimate_storage_counts_rendered_tokens_and_residual_bytes(monkeypatch):
    df = pd.DataFrame([fake_row("p0"), fake_row("p1")])
    monkeypatch.setattr(record.wbprompts, "render_verdict_prompt", lambda row, tok: row["text_1"])
    out = record.estimate_storage(df, LengthTokenizer(), fake_cfg(), layers=(0, 2), d_model=4)
    assert out["n_prompts"] == 2
    assert out["n_tokens"] == len(df.iloc[0].text_1) + len(df.iloc[1].text_1)
    assert out["residual_bytes"] == out["n_tokens"] * 2 * 4 * 2
```

- [ ] **Step 3: Run tests and confirm RED**

Run: `uv run pytest judge-whitebox/tests/test_token_manifold_record.py -q`
Expected: missing `layer_indices`/`prompt_token_meta`/`estimate_storage`.

- [ ] **Step 4: Implement by reusing the existing token map**

```python
STRUCTURAL_GROUPS = ("instr_format", "slot_labels", "question", "transcript_1",
                     "transcript_2", "answer_readout")


def layer_indices(n_states, n_select=12):
    assert n_states > 0 and n_select > 0
    points = np.rint(np.linspace(0, n_states - 1, min(n_states, n_select))).astype(int)
    return tuple(sorted(set(points.tolist())))


def prompt_token_meta(row, tokenizer, c):
    prompt = wbprompts.render_verdict_prompt(row, tokenizer)
    enc = tokenizer(prompt, return_offsets_mapping=True)
    groups = token_ablate.token_groups(row, tokenizer, c)
    group_id = np.full(len(enc["input_ids"]), -1, dtype=np.int8)
    for idx, name in enumerate(STRUCTURAL_GROUPS):
        group_id[groups[name]] = idx
    assert (group_id >= 0).all()
    return {"prompt": prompt, "input_ids": enc["input_ids"],
            "offsets": enc["offset_mapping"], "group_id": group_id,
            "group_names": STRUCTURAL_GROUPS}
```

`estimate_storage` renders each prompt once and computes
`tokens × selected_layers × d_model × dtype_bytes`; metadata bytes are reported separately as a
conservative `tokens × 32` estimate, not hidden inside the residual estimate.

- [ ] **Step 5: Run task tests**

Run: `uv run pytest judge-whitebox/tests/test_token_manifold_record.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add judge-whitebox/token_manifold_record.py judge-whitebox/tests/test_token_manifold_record.py
git commit -m "token manifolds: map tokens and estimate storage"
```

---

### Task 3: Immutable per-token activation shards and manifest CLI

**Files:**
- Modify: `judge-whitebox/token_manifold_record.py`
- Modify: `judge-whitebox/tests/test_token_manifold_record.py`

**Interfaces:**
- Produces: `record_rows(model, tokenizer, df, c, layers) -> Iterator[dict]`.
- Produces: `write_shards(rows, out_dir, shard_prompts, manifest_meta) -> pathlib.Path`.
- CLI: `--model --fold A|B --fold-seed --layers --limit --shard-prompts --out --estimate-only`.

- [ ] **Step 1: Write a failing real-transformer shape test**

```python
def test_record_rows_keeps_every_token_at_selected_layers(monkeypatch):
    model, tok, df = make_tiny_model(), OffsetTokenizer(), make_pairs_df(2)
    monkeypatch.setattr(record.wbprompts, "render_verdict_prompt", lambda row, tok: row["text_1"])
    monkeypatch.setattr(record, "prompt_token_meta", fake_token_meta)
    rows = list(record.record_rows(model, tok, df, fake_cfg(), layers=(0, 2)))
    assert rows[0]["resid"].shape == (len(df.iloc[0].text_1), 2, model.config.hidden_size)
    assert rows[0]["resid"].dtype == torch.float16
    assert rows[0]["gap"] == pytest.approx(rows[0]["l1"] - rows[0]["l2"])
```

Reuse `make_tiny_model`'s exact random Qwen configuration from
`judge-whitebox/tests/test_record_resid.py:13-21`; define `make_pairs_df` locally with the complete
columns returned by `fake_row`. Define the remaining test helpers explicitly:

```python
def fake_token_meta(row, tokenizer, c):
    n = len(row["text_1"])
    return {"prompt": row["text_1"], "input_ids": list(range(1, n + 1)),
            "offsets": [(i, i + 1) for i in range(n)],
            "group_id": np.ones(n, dtype=np.int8),
            "group_names": record.STRUCTURAL_GROUPS}


def fake_recorded_row(uid, n_tokens):
    return {"row_idx": int(uid[1:]), "pair_uid": uid, "item": "it0", "order": 0,
            "resid": torch.zeros(n_tokens, 2, 4, dtype=torch.float16),
            "input_ids": list(range(n_tokens)), "offsets": [(i, i + 1) for i in range(n_tokens)],
            "group_id": np.ones(n_tokens, dtype=np.int8), "l1": 1.0, "l2": 0.0, "gap": 1.0}
```

- [ ] **Step 2: Write failing shard/manifest tests**

```python
def test_write_shards_round_trips_and_hashes(tmp_path):
    rows = [fake_recorded_row("p0", 3), fake_recorded_row("p1", 2)]
    manifest = record.write_shards(iter(rows), tmp_path / "new", 1, {"model": "dev"})
    data = json.loads(manifest.read_text())
    assert data["n_prompts"] == 2 and data["n_tokens"] == 5
    assert len(data["shards"]) == 2
    for entry in data["shards"]:
        path = manifest.parent / entry["path"]
        assert record.sha256(path) == entry["sha256"]


def test_write_shards_refuses_existing_output(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(FileExistsError):
        record.write_shards(iter([]), out, 1, {})
```

- [ ] **Step 3: Run tests and confirm RED**

Run: `uv run pytest judge-whitebox/tests/test_token_manifold_record.py -q`
Expected: missing recorder/shard functions.

- [ ] **Step 4: Implement the forward path**

For each prompt, call the model once with `output_hidden_states=True`, stack only the selected states,
transpose to `[tokens, layers, d_model]`, cast to float16 on CPU, and use `verdicts.readout` on the final
logits. Keep `row_idx`, `pair_uid`, `item`, `order`, token metadata and gap beside the tensor.

- [ ] **Step 5: Implement atomic new-output writes**

```python
def atomic_torch_save(obj, path):
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".partial")
    if path.exists() or tmp.exists():
        raise FileExistsError(path if path.exists() else tmp)
    torch.save(obj, tmp)
    os.replace(tmp, path)
```

`write_shards` concatenates prompt tensors, adds `prompt_idx`, writes `shard_00000.pt`, and writes
`manifest.json` last through the same atomic pattern. It never resumes into a new fragment or replaces
an existing file.

- [ ] **Step 6: Implement CLI estimate and recording modes**

`--estimate-only` loads `AutoConfig`, renders the selected fold, prints JSON, and writes nothing.
Recording asserts base-only (`--adapter` does not exist), resolves 12 layers when `--layers auto`, and
records exact model/pairs/prompt hashes in the manifest.

- [ ] **Step 7: Run task and legacy recorder tests**

Run: `uv run pytest judge-whitebox/tests/test_token_manifold_record.py judge-whitebox/tests/test_record_resid.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add judge-whitebox/token_manifold_record.py judge-whitebox/tests/test_token_manifold_record.py
git commit -m "token manifolds: record immutable token shards"
```

---

### Task 4: Item-grouped continuous-gap decoder timeline

**Files:**
- Create: `judge-whitebox/token_manifold_discover.py`
- Create: `judge-whitebox/tests/test_token_manifold_discover.py`

**Interfaces:**
- Consumes: Task 3 manifest/shards.
- Produces: `load_manifest(path) -> dict[str, np.ndarray]`.
- Produces: `ridge_timeline(data, alphas, cv_folds) -> tuple[pd.DataFrame, np.ndarray]`.
- Produces one OOF row per `(prompt, token, layer)` with `pred_gap` and one summary row per
  `(layer, structural_group, relative_decile)`.

At the top of the test file, define the planted dataset used below:

```python
def planted_token_data(n_items, prompts_per_item, tokens, layers, d, signal_layer, seed):
    rng = np.random.default_rng(seed)
    n_prompts = n_items * prompts_per_item
    gap = rng.normal(size=n_prompts)
    item = np.repeat([f"it{i}" for i in range(n_items)], prompts_per_item)
    prompt_idx = np.repeat(np.arange(n_prompts), tokens)
    X = rng.normal(size=(n_prompts * tokens, layers, d))
    direction = rng.normal(size=d); direction /= np.linalg.norm(direction)
    X[:, signal_layer] += 4 * gap[prompt_idx, None] * direction
    return {"X": X.astype(np.float32), "gap": gap, "prompt_idx": prompt_idx,
            "item": item, "group_id": np.ones(len(X), dtype=np.int8),
            "position": np.tile(np.arange(tokens), n_prompts)}
```

- [ ] **Step 1: Write a failing planted-signal test**

```python
def test_ridge_timeline_finds_planted_layer_and_uses_item_groups():
    data = planted_token_data(n_items=20, prompts_per_item=2, tokens=6,
                              layers=4, d=10, signal_layer=2, seed=0)
    summary, oof = discover.ridge_timeline(data, alphas=(0.1, 1.0, 10.0), cv_folds=5)
    peak = summary.groupby("layer").r2.mean().idxmax()
    assert peak == 2
    assert summary.loc[summary.layer == 2, "r2"].mean() > 0.7
    assert set(oof["item"]) == set(data["item"])
```

- [ ] **Step 2: Write a failing equal-prompt-weight test**

```python
def test_prompt_weights_sum_to_one_per_prompt():
    prompt_idx = np.array([0, 0, 1, 1, 1, 1])
    weights = discover.prompt_weights(prompt_idx)
    assert weights[prompt_idx == 0].sum() == pytest.approx(1.0)
    assert weights[prompt_idx == 1].sum() == pytest.approx(1.0)
```

- [ ] **Step 3: Run tests and confirm RED**

Run: `uv run pytest judge-whitebox/tests/test_token_manifold_discover.py -q`
Expected: import or attribute failure.

- [ ] **Step 4: Implement manifest verification and loading**

Verify every shard hash before loading. Concatenate `resid`, `prompt_idx`, token positions, group IDs
and prompt metadata. Reject mixed dimensions/layer lists and a manifest whose totals disagree with the
shards.

- [ ] **Step 5: Implement grouped ridge regression**

```python
def prompt_weights(prompt_idx):
    _, inverse, counts = np.unique(prompt_idx, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]
```

For each layer, choose alpha by `GroupKFold` on item, fit `sklearn.linear_model.Ridge` with the prompt
weights, and store strictly out-of-fold predictions. Summaries use prompt-weighted `R²`; relative token
deciles are computed within each structural group and prompt.

- [ ] **Step 6: Add shuffled-gap null and CLI**

CLI: `timeline --manifest ... --out timeline.csv --alphas 0.1,1,10,100 --cv-folds 5 --seed 0`.
Shuffle gaps across prompts before token expansion, preserving one shuffled gap for all tokens of a
prompt, refit once, and write `null_r2` beside each summary row.

- [ ] **Step 7: Run task tests**

Run: `uv run pytest judge-whitebox/tests/test_token_manifold_discover.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add judge-whitebox/token_manifold_discover.py judge-whitebox/tests/test_token_manifold_discover.py
git commit -m "token manifolds: decode the token-time gap"
```

---

### Task 5: CEBRA-Behaviour adapter and curved synthetic gate

**Files:**
- Modify: `judge-whitebox/token_manifold_discover.py`
- Modify: `judge-whitebox/tests/test_token_manifold_discover.py`

**Interfaces:**
- Produces: `fit_cebra(X_train, gap_train, X_test, output_dim, seed, iterations,
  device="cpu") -> MethodResult`.
- `MethodResult` fields: `method`, `train_code`, `test_code`, `test_reconstruction`, `metadata`.

Define the result type before any adapter:

```python
@dataclasses.dataclass
class MethodResult:
    method: str
    train_code: np.ndarray
    test_code: np.ndarray
    test_reconstruction: np.ndarray | None
    metadata: dict
```

Define the curved synthetic data used by this task:

```python
def curved_gap_data(n, d, seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, n)
    intrinsic = np.column_stack([np.cos(theta), np.sin(theta), np.cos(2 * theta)])
    projection = rng.normal(size=(3, d))
    return (intrinsic @ projection + 0.02 * rng.normal(size=(n, d))).astype(np.float32), theta
```

- [ ] **Step 1: Create the isolated Python 3.12 method environment**

Run only after dependency-download approval:

```bash
uv venv --python 3.12 .venv-manifold
uv pip install --python .venv-manifold/bin/python -r judge-whitebox/requirements-manifold.txt
```

Expected: `.venv-manifold/bin/python` imports `cebra`, `bsf`, `saes`, `pandas`, `pyarrow`, `yaml` and
`pytest`. The project itself is not installed because its root `pyproject.toml` requires Python 3.13;
tests import `judge-whitebox/` directly through pytest's working-directory path.

- [ ] **Step 2: Write the failing adapter test**

```python
def test_fit_cebra_returns_requested_dimension_and_seed_metadata():
    X, y = curved_gap_data(n=240, d=12, seed=0)
    a = discover.fit_cebra(X[:180], y[:180], X[180:], 4, 3, 200, device="cpu")
    assert a.train_code.shape == (180, 4) and a.test_code.shape == (60, 4)
    assert np.isfinite(a.test_code).all()
    assert a.test_reconstruction is None
    assert a.metadata == {"output_dim": 4, "seed": 3}
```

- [ ] **Step 3: Run in the manifold venv and confirm RED**

Run: `.venv-manifold/bin/python -m pytest judge-whitebox/tests/test_token_manifold_discover.py -q`
Expected: missing `fit_cebra`.

- [ ] **Step 4: Implement the thin upstream wrapper**

```python
def fit_cebra(X_train, gap_train, X_test, output_dim, seed, iterations, device="cpu"):
    import cebra
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = cebra.CEBRA(model_architecture="offset1-model", conditional="time_delta",
                        output_dimension=output_dim, max_iterations=iterations,
                        batch_size=min(512, len(X_train)), device=device)
    model.fit(np.asarray(X_train, np.float32), np.asarray(gap_train, np.float32)[:, None])
    return MethodResult("cebra", model.transform(X_train), model.transform(X_test), None,
                        {"output_dim": output_dim, "seed": seed})
```

- [ ] **Step 5: Add held-out code-to-gap scoring**

Use a ridge regressor fitted on `train_code` only and report held-out `R²`. Compare dimensions
`{1, 2, 4, 8, 16}` so the bake-off can detect a trivial one-dimensional solution.

- [ ] **Step 6: Deliberately defer CEBRA-Hybrid**

Add this comment next to the adapter registry:

```python
# ponytail: Behaviour-only bake-off; add boundary-safe Hybrid sampling only if CEBRA clears
# the held-out/stability gate, otherwise a custom prompt-time sampler is dead code.
```

- [ ] **Step 7: Run the CEBRA test and record upstream non-determinism**

Run: `.venv-manifold/bin/python -m pytest judge-whitebox/tests/test_token_manifold_discover.py -q`
Expected: PASS. CEBRA v0.6.0 advertises `non_deterministic=True`; stability is therefore an empirical
multi-run metric, not byte equality for one seed.

- [ ] **Step 8: Commit**

```bash
git add judge-whitebox/token_manifold_discover.py judge-whitebox/tests/test_token_manifold_discover.py
git commit -m "token manifolds: add CEBRA behaviour discovery"
```

---

### Task 6: BSF and matched SAE/Ising adapters

**Files:**
- Modify: `judge-whitebox/token_manifold_discover.py`
- Modify: `judge-whitebox/tests/test_token_manifold_discover.py`

**Interfaces:**
- Produces: `fit_bsf(X_train, X_test, group_size, n_groups, l0, seed, epochs) -> MethodResult`.
- Produces: `fit_sae(X_train, X_test, width, k, seed, epochs) -> MethodResult`.
- Produces: `ising_clusters(train_code, alpha, method, n_clusters=None) -> np.ndarray`.
- Produces: `fit_method(name, X_train, gap_train, X_test, seed, quick) -> MethodResult`, a three-branch
  dispatcher used by the CLI and tests.

Define the synthetic helpers before the tests:

```python
def sparse_manifold_mix(n, d, seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, n)
    active = rng.integers(0, 2, n)
    z = np.column_stack([np.cos(theta), np.sin(theta), active, active * theta])
    return (z @ rng.normal(size=(4, d))).astype(np.float32), z


def planted_binary_feature_groups(n, groups, seed):
    rng = np.random.default_rng(seed)
    codes = np.zeros((n, 4), dtype=np.float32)
    choice = rng.integers(0, len(groups), n)
    for row, selected in enumerate(choice):
        codes[row, list(groups[selected])] = 1
    return codes
```

- [ ] **Step 1: Write failing reconstructive-adapter tests**

```python
@pytest.mark.parametrize("method", ["bsf", "sae"])
def test_reconstructive_method_returns_codes_and_test_reconstruction(method):
    X, _ = sparse_manifold_mix(n=256, d=12, seed=0)
    result = discover.fit_method(method, X[:192], None, X[192:], seed=2, quick=True)
    assert result.train_code.shape[0] == 192
    assert result.test_code.shape[0] == 64
    assert result.test_reconstruction.shape == X[192:].shape
    assert np.isfinite(result.test_reconstruction).all()
```

- [ ] **Step 2: Write a failing Ising-cluster test**

```python
def test_ising_clusters_groups_planted_feature_pairs():
    codes = planted_binary_feature_groups(n=400, groups=((0, 1), (2, 3)), seed=0)
    labels = discover.ising_clusters(codes, alpha=0.01, method="spectral", n_clusters=2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
```

- [ ] **Step 3: Run tests and confirm RED**

Run: `.venv-manifold/bin/python -m pytest judge-whitebox/tests/test_token_manifold_discover.py -q`
Expected: missing BSF/SAE adapters.

- [ ] **Step 4: Wrap Goodfire BSF directly**

Use `bsf.GrassmannianBSF`, `bsf.VanillaBSF` and `bsf.train`; Task 1's main bake-off uses
Grassmannian. Seed NumPy/Torch before construction. Return flattened block coordinates plus the
native reconstruction and block-shaped codes in metadata.

- [ ] **Step 5: Train the minimal matched SAE**

Instantiate `saes.BatchTopKSAE(d_in, width, k)` and train with Adam on reconstruction MSE. This repo is
inference-only, so keep the local loop to the following operations: shuffle, forward, MSE, backward,
optimizer step, then normalize `sae.decoder.weight` along `dim=0` because PyTorch stores it as
`[d_in, d_sae]`. Match SAE `k` to BSF active block count and choose width so the total
decoder-coordinate budget equals `n_groups × group_size`.

- [ ] **Step 6: Reuse Goodfire's Ising and clustering functions**

Import `compute_ising`, `cluster_spectral` and `cluster_leiden` from `unsupervised_clustering`; binarize
codes using `code > 0`. The CPU test uses spectral clustering; Leiden is exercised by an import smoke
only.

- [ ] **Step 7: Run task tests**

Run: `.venv-manifold/bin/python -m pytest judge-whitebox/tests/test_token_manifold_discover.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add judge-whitebox/token_manifold_discover.py judge-whitebox/tests/test_token_manifold_discover.py
git commit -m "token manifolds: add block and SAE baselines"
```

---

### Task 7: Bake-off report, cluster script and launch gate

**Files:**
- Modify: `judge-whitebox/token_manifold_discover.py`
- Modify: `judge-whitebox/tests/test_token_manifold_discover.py`
- Create: `judge-whitebox/sbatch/token_manifold.sbatch`
- Modify: `judge-whitebox/README.md`

**Interfaces:**
- CLI: `bakeoff --manifest ... --layer ... --methods cebra,bsf,sae --out ... --quick`.
- Produces: `bakeoff.csv`, one row per method/config/seed with reconstruction, code-gap `R²`,
  active-code count, stability-ready metadata and runtime.
- Produces: `bakeoff.pt` with fitted small models/codes; output path must be new.

- [ ] **Step 1: Write a failing end-to-end bake-off test**

```python
def test_bakeoff_writes_one_row_per_method(tmp_path, monkeypatch):
    manifest = write_fake_manifest(tmp_path / "recording")
    monkeypatch.setattr(discover, "fit_method", fake_fit_method)
    out = tmp_path / "bakeoff"
    discover.run_bakeoff(manifest, layer=1, methods=("cebra", "bsf", "sae"),
                         out_dir=out, seeds=(0,), quick=True)
    table = pd.read_csv(out / "bakeoff.csv")
    assert set(table.method) == {"cebra", "bsf", "sae"}
    assert set(["test_gap_r2", "test_recon_r2", "runtime_s"]) <= set(table)
    assert (out / "bakeoff.pt").exists()
```

Define its two helpers directly in `test_token_manifold_discover.py`:

```python
def write_fake_manifest(out):
    rows = []
    for prompt in range(8):
        rows.append({"row_idx": prompt, "pair_uid": f"p{prompt}",
                     "item": f"it{prompt // 2}", "order": prompt % 2,
                     "resid": torch.randn(4, 2, 6, generator=torch.Generator().manual_seed(prompt)),
                     "input_ids": [1, 2, 3, 4], "offsets": [(0, 1), (1, 2), (2, 3), (3, 4)],
                     "group_id": np.array([0, 1, 1, 5], dtype=np.int8),
                     "l1": float(prompt), "l2": 0.0, "gap": float(prompt)})
    return record.write_shards(iter(rows), out, shard_prompts=4,
                               manifest_meta={"model": "dev", "layers": [0, 1]})


def fake_fit_method(name, X_train, gap_train, X_test, seed, quick):
    width = 2 if name == "cebra" else 4
    train = np.asarray(X_train)[:, :width]
    test = np.asarray(X_test)[:, :width]
    recon = None if name == "cebra" else np.asarray(X_test)
    return discover.MethodResult(name, train, test, recon, {"seed": seed})
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `.venv-manifold/bin/python -m pytest judge-whitebox/tests/test_token_manifold_discover.py -q`
Expected: missing `run_bakeoff`.

- [ ] **Step 3: Implement the bounded comparison**

Use the manifest's discovery fold only. Split by item into internal train/test groups. Subsample one
token per prompt per structural group per draw, normalize on train only, and apply identical rows to
all methods. Write metrics and fitted objects atomically. `--quick` uses one seed, dimensions
`{1, 4}`, 200 CEBRA iterations and five BSF/SAE epochs; scientific defaults use the spec grids.

- [ ] **Step 4: Add a single-stage sbatch script**

`WB_STAGE=record|timeline|bakeoff` chooses one command. The script requests one H100, 8 CPUs, 80 GB,
24 hours, `--requeue`, append logs, and refuses an existing destination. It sources `wb_common.sh`;
timeline/bakeoff additionally require `WB_MANIFOLD_VENV` and activate that existing environment. It
never installs packages. Every example submission is commented and requires the user to choose the
storage destination/tag.

- [ ] **Step 5: Document the separate environment and gates**

Add to README:

```bash
uv venv --python 3.12 .venv-manifold
uv pip install --python .venv-manifold/bin/python -r judge-whitebox/requirements-manifold.txt
.venv-manifold/bin/python -m pytest judge-whitebox/tests/test_token_manifold_*.py -q
```

Document that no SWC submission occurs until `--estimate-only` output and ceph quota are reviewed.

- [ ] **Step 6: Run targeted and full whitebox tests**

Run: `.venv-manifold/bin/python -m pytest judge-whitebox/tests/test_token_manifold_*.py -q`
Expected: PASS.
Run: `uv run pytest judge-whitebox/tests -q`
Expected: PASS.

- [ ] **Step 7: Run a two-row CPU smoke**

Run the recorder with the tiny test model through its Python API, load its manifest, run `timeline`,
and run `bakeoff --quick`. Assert no-op logits equal the direct model readout and every CSV field is
finite. This smoke downloads no model.

- [ ] **Step 8: Print—but do not execute—the real storage estimate commands**

```bash
python token_manifold_record.py --model Qwen3-8B --fold A --estimate-only --out /unused
python token_manifold_record.py --model Llama-3.2-3B-Instruct --fold B --estimate-only --out /unused
```

Record the expected command shape in the handoff; actual outputs require the locally cached tokenizer
or a compute-node run and are the user's launch gate.

- [ ] **Step 9: Commit**

```bash
git add judge-whitebox/token_manifold_discover.py judge-whitebox/tests/test_token_manifold_discover.py \
  judge-whitebox/sbatch/token_manifold.sbatch judge-whitebox/README.md
git commit -m "token manifolds: add the method bakeoff gate"
```

---

## Plan 1 implementation gate

Before requesting any real recording job, provide the user:

- passing targeted and full whitebox tests;
- exact dependency revisions and environment command;
- local model-specific storage estimates when tokenizers are cached, otherwise the exact estimate-only
  compute commands;
- a synthetic small bake-off CSV proving all three adapters and metrics execute;
- no semantic feature names and no cluster submission.

After the user reviews storage and launches the bounded recording/bake-off jobs, the empirical gate
adds the real stability/reconstruction/decoding summary, a recommendation to retain CEBRA, BSF,
SAE/Ising or a complementary pair, and explicit reasons for rejecting any method. Plan 2 is written
only after that review.

## Deferred by design

- Boundary-safe CEBRA-Hybrid sampling: add only if CEBRA-Behaviour passes the first method gate.
- Phrase naming, re-extraction, manifold inversion, causal intervention and interaction runs: Plan 2.
- Fine-tuned checkpoints: Plan 3.
