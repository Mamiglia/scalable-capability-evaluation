from types import SimpleNamespace

import pytest
import torch
import screen


class ToyDecomp:
    """Minimal .entries/.meta object — not svd_components.Decomp, screen doesn't require that class."""

    def __init__(self, entries, meta):
        self.entries = entries
        self.meta = meta


def test_finite_difference_attribution():
    """Load-bearing: attribution for a component must match a finite-difference weight
    perturbation along that component's rank-1 direction, for a plain nn.Linear."""
    torch.manual_seed(0)
    d_in, d_out, n = 16, 12, 5
    lin = torch.nn.Linear(d_in, d_out, bias=False)
    U, S, Vh = torch.linalg.svd(lin.weight.data.float(), full_matrices=False)
    V = Vh.T
    decomp = ToyDecomp(entries={"w": (U, S, V)}, meta={"w": {"head": None}})
    entry_map = {"w": (lin, False)}

    x = torch.randn(n, d_in)
    a = torch.randn(d_out)

    def forward_fn():
        return (a * lin(x)).sum()

    stats = screen.screen_one(lin, decomp, entry_map, forward_fn)
    activity, attr_signed, _ = stats["w"]

    # Reference finite difference in float64: L is exactly linear in W (single matmul), so any
    # eps would do analytically, but float32 subtractive cancellation swamps a float32 estimate.
    W64, x64, a64 = lin.weight.data.double(), x.double(), a.double()
    U64, S64, V64 = U.double(), S.double(), V.double()
    base_loss = (a64 * (x64 @ W64.T)).sum().item()
    eps = 1e-4
    for i in (0, 5, 11):
        w_pert = W64 + eps * S64[i] * torch.outer(U64[:, i], V64[:, i])
        fd = ((a64 * (x64 @ w_pert.T)).sum().item() - base_loss) / eps
        assert abs(attr_signed[i].item() - fd) < 1e-3 * max(abs(fd), 1.0), (i, attr_signed[i].item(), fd)

    # activity is exactly S[i] * sum_t |x_t . V[:,i]| — no grad involved, check it directly.
    for i in (0, 5, 11):
        expected = (S[i] * (x @ V[:, i]).abs().sum()).item()
        assert abs(activity[i].item() - expected) < 1e-4 * max(abs(expected), 1.0), (i, activity[i].item(), expected)


def test_row_sliced_component_with_nonzero_head_offset():
    """A dropped/zeroed lo-offset bug on the grad-side slice would only show up for head != 0 —
    the whole-matrix (head=None) case above always has lo=0 and can't catch that."""
    torch.manual_seed(7)
    d_in, d_out = 16, 12
    lin = torch.nn.Linear(d_in, d_out, bias=False)
    head, width = 2, 4  # third of three width-4 output heads (d_out = 3 x 4)
    lo, hi = head * width, (head + 1) * width
    U, S, Vh = torch.linalg.svd(lin.weight.data[lo:hi, :].float(), full_matrices=False)
    V = Vh.T
    decomp = ToyDecomp(entries={"w": (U, S, V)}, meta={"w": {"head": head}})
    entry_map = {"w": (lin, False)}

    x = torch.randn(5, d_in)
    a = torch.randn(d_out)

    def forward_fn():
        return (a * lin(x)).sum()

    stats = screen.screen_one(lin, decomp, entry_map, forward_fn)
    _, attr_signed, _ = stats["w"]

    W64, x64, a64 = lin.weight.data.double(), x.double(), a.double()
    U64, S64, V64 = U.double(), S.double(), V.double()
    base_loss = (a64 * (x64 @ W64.T)).sum().item()
    eps = 1e-4
    i = 0
    w_pert = W64.clone()
    w_pert[lo:hi, :] += eps * S64[i] * torch.outer(U64[:, i], V64[:, i])
    fd = ((a64 * (x64 @ w_pert.T)).sum().item() - base_loss) / eps
    assert abs(attr_signed[i].item() - fd) < 1e-3 * max(abs(fd), 1.0)


def test_reconstruction_matches_module_weight_via_slice_bounds():
    """decomp.entries reconstruct exactly to the module weight slice that screen's own
    _slice_bounds picks out — cross-checks screen's lo/width derivation against svd_components'
    slicing convention on real (synthetic Qwen-shaped) matrices, not just the toy 1-matrix case.
    """
    import svd_components as svdc

    arch = {"n_heads": 4, "n_kv_heads": 2, "head_dim": 8}
    d_model = 32
    g = torch.Generator().manual_seed(3)
    sd = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(4 * 8, d_model, generator=g),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(d_model, 4 * 8, generator=g),
        "model.layers.0.mlp.down_proj.weight": torch.randn(d_model, 64, generator=g),
    }
    decomp = svdc.decompose(sd, arch)

    cases = [
        ("L0.q.h2", sd["model.layers.0.self_attn.q_proj.weight"], False),
        ("L0.o.h1", sd["model.layers.0.self_attn.o_proj.weight"], True),
        ("L0.down", sd["model.layers.0.mlp.down_proj.weight"], False),
    ]
    for key, weight, by_columns in cases:
        U, S, V = decomp.entries[key]
        head = decomp.meta[key]["head"]
        lo, hi = screen._slice_bounds(by_columns, head, U, V)
        expected = weight[:, lo:hi] if by_columns else weight[lo:hi, :]
        recon = U @ torch.diag(S) @ V.T
        assert torch.allclose(recon, expected, atol=1e-4), key


def test_column_sliced_component_reads_input_slice():
    """by_columns=True (o-style): x must be sliced, grad must not be."""
    torch.manual_seed(1)
    d_in, d_out = 8, 6
    lin = torch.nn.Linear(d_in, d_out, bias=False)
    head, width = 1, 4
    lo, hi = head * width, (head + 1) * width  # second of two width-4 input slices
    U, S, Vh = torch.linalg.svd(lin.weight.data[:, lo:hi].float(), full_matrices=False)
    V = Vh.T
    decomp = ToyDecomp(entries={"w": (U, S, V)}, meta={"w": {"head": head}})
    entry_map = {"w": (lin, True)}

    x = torch.randn(4, d_in)
    a = torch.randn(d_out)

    def forward_fn():
        return (a * lin(x)).sum()

    stats = screen.screen_one(lin, decomp, entry_map, forward_fn)
    activity, attr_signed, attr_abs = stats["w"]
    assert torch.isfinite(activity).all() and torch.isfinite(attr_signed).all()

    W64, x64, a64 = lin.weight.data.double(), x.double(), a.double()
    U64, S64, V64 = U.double(), S.double(), V.double()
    base_loss = (a64 * (x64 @ W64.T)).sum().item()
    eps = 1e-4
    i = 0
    w_pert = W64.clone()
    w_pert[:, lo:hi] += eps * S64[i] * torch.outer(U64[:, i], V64[:, i])
    fd = ((a64 * (x64 @ w_pert.T)).sum().item() - base_loss) / eps
    assert abs(attr_signed[i].item() - fd) < 1e-3 * max(abs(fd), 1.0)


def test_per_prompt_matrices_activity_is_length_normalised():
    """per_prompt_activity rows must be mean-per-token, not raw sums (which scale with prompt
    length and swamp the actual signal) — unchanged behaviour from before this task's merge of
    the per-prompt pass into a gradient-enabled one. Also pins the [n_prompts, k] shape shared
    by all three returned matrices."""
    torch.manual_seed(4)
    d_in, d_out, vocab = 6, 5, 15
    model = ToyCausalModel(vocab, d_in, d_out)
    U, S, Vh = torch.linalg.svd(model.lin.weight.data.float(), full_matrices=False)
    V = Vh.T
    decomp = ToyDecomp(entries={"w": (U, S, V)}, meta={"w": {"head": None}})
    entry_map = {"w": (model.lin, False)}

    kept_ids = [torch.randint(0, vocab, (1, 3)), torch.randint(0, vocab, (1, 9))]  # different lengths
    prompt_tokens = [3, 9]
    top_components = [("w", 0), ("w", 1)]

    act_mat, attr_mat, proj_mat = screen._per_prompt_matrices(
        model, decomp, entry_map, kept_ids, prompt_tokens, top_components, "nll", None)

    for mat in (act_mat, attr_mat, proj_mat):
        assert mat.shape == (2, 2)

    for p, (ids, n_tok) in enumerate(zip(kept_ids, prompt_tokens)):
        x = model.embed(ids).reshape(-1, d_in)
        raw_activity, _, _ = screen._project_key(U, S, V, False, None, x)
        assert torch.allclose(act_mat[p], raw_activity[:2] / n_tok, atol=1e-5)


def test_per_prompt_attr_matches_finite_difference():
    """per_prompt_attr for a single toy prompt must match a finite-difference weight
    perturbation of the screened Linear — same style as test_finite_difference_attribution at
    the top of this file, but through the embed->lin->head chain _per_prompt_matrices actually
    differentiates through. "gap" loss keeps that chain exactly linear in lin.weight (it reads
    only the last position, with no softmax/CE nonlinearity), so this is still an exact FD check,
    not an approximate one."""
    torch.manual_seed(6)
    d_in, d_out, vocab = 5, 4, 20
    model = ToyCausalModel(vocab, d_in, d_out)
    U, S, Vh = torch.linalg.svd(model.lin.weight.data.float(), full_matrices=False)
    V = Vh.T
    decomp = ToyDecomp(entries={"w": (U, S, V)}, meta={"w": {"head": None}})
    entry_map = {"w": (model.lin, False)}

    class GapTokenizer:
        def encode(self, text, add_special_tokens=False):
            return {"1": [0], "2": [1]}[text]

    ids = torch.randint(0, vocab, (1, 6))
    top_components = [("w", 0), ("w", 2)]

    _, attr_mat, _ = screen._per_prompt_matrices(
        model, decomp, entry_map, [ids], [6], top_components, "gap", GapTokenizer())

    embed_last64 = model.embed(ids)[0, -1].double()  # [d_in], fixed (not perturbed)
    head64 = model.head.weight.data.double()  # [vocab, d_out], fixed
    W64 = model.lin.weight.data.double()

    def gap_loss(W):
        logits_last = head64 @ (W @ embed_last64)  # [vocab]
        return (logits_last[0] - logits_last[1]).item()

    base = gap_loss(W64)
    eps = 1e-4
    for j, (key, i) in enumerate(top_components):
        w_pert = W64 + eps * S[i].double() * torch.outer(U[:, i].double(), V[:, i].double())
        fd = (gap_loss(w_pert) - base) / eps
        assert abs(attr_mat[0, j].item() - fd) < 1e-3 * max(abs(fd), 1.0), (j, attr_mat[0, j].item(), fd)


def test_per_prompt_proj_sign_flips_when_x_is_negated():
    """per_prompt_proj is the x-side-only signed projection: feeding -x instead of x must flip
    its sign exactly (activity/attr both fold in |.| or a gradient, proj is the one quantity
    that's pure x, so this is the cleanest place to check sign behaviour)."""
    torch.manual_seed(9)
    d_in, d_out, vocab = 4, 3, 2
    model = ToyCausalModel(vocab, d_in, d_out)
    x = torch.randn(d_in)
    with torch.no_grad():
        model.embed.weight[0] = x
        model.embed.weight[1] = -x
    U, S, Vh = torch.linalg.svd(model.lin.weight.data.float(), full_matrices=False)
    V = Vh.T
    decomp = ToyDecomp(entries={"w": (U, S, V)}, meta={"w": {"head": None}})
    entry_map = {"w": (model.lin, False)}

    # two 2-token prompts (repeat the single id so NLL's shift has something to predict):
    # token id 0 -> embedding x, token id 1 -> embedding -x.
    kept_ids = [torch.tensor([[0, 0]]), torch.tensor([[1, 1]])]
    prompt_tokens = [2, 2]
    top_components = [("w", 0), ("w", 1)]

    _, _, proj_mat = screen._per_prompt_matrices(
        model, decomp, entry_map, kept_ids, prompt_tokens, top_components, "nll", None)

    assert torch.allclose(proj_mat[0], -proj_mat[1], atol=1e-5)
    assert (proj_mat[0].abs() > 1e-6).all()  # nontrivial, not a both-zero false pass


def test_decomp_cache_identity_mismatch_raises(tmp_path):
    path = tmp_path / "decomp.pt"
    torch.save({"entries": {}, "meta": {}, "model": "Qwen3-0.6B", "adapter": None}, path)
    with pytest.raises(AssertionError, match="identity mismatch"):
        screen._load_or_build_decomp(None, str(path), "Qwen3-8B", None)


def test_decomp_cache_identity_match_loads(tmp_path):
    path = tmp_path / "decomp.pt"
    torch.save({"entries": {"k": 1}, "meta": {"k": 2}, "model": "Qwen3-0.6B", "adapter": "adap"}, path)
    decomp = screen._load_or_build_decomp(None, str(path), "Qwen3-0.6B", "adap")
    assert decomp.entries == {"k": 1} and decomp.meta == {"k": 2}


def test_decomp_cache_load_retries_once_on_vanished_file_then_succeeds(tmp_path, monkeypatch):
    """Simulates the array-task race: exists() sees the file, but the first load attempt hits a
    concurrent writer's os.replace window. One retry must recover instead of crashing the job."""
    path = tmp_path / "decomp.pt"
    torch.save({"entries": {"k": 1}, "meta": {"k": 2}, "model": "Qwen3-0.6B", "adapter": None}, path)

    calls = {"n": 0}
    real_load = torch.load

    def flaky_load(p, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise FileNotFoundError("simulated race")
        return real_load(p, *a, **kw)

    monkeypatch.setattr(torch, "load", flaky_load)
    monkeypatch.setattr(screen.time, "sleep", lambda s: None)  # skip the real wait in tests
    decomp = screen._load_or_build_decomp(None, str(path), "Qwen3-0.6B", None)
    assert decomp.entries == {"k": 1} and decomp.meta == {"k": 2}
    assert calls["n"] == 2


def test_decomp_cache_build_writes_atomically_no_leftover_tmp(tmp_path):
    """The build (cache-miss) path must land the final file via os.replace, not a bare torch.save
    to the destination — no partial file should ever be visible under decomp_path, and no .tmp*
    scratch file should be left behind once the write completes."""
    arch = {"n_heads": 2, "n_kv_heads": 1, "head_dim": 4}
    d_model = 8
    sd = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(2 * 4, d_model),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(1 * 4, d_model),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(1 * 4, d_model),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(d_model, 2 * 4),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(16, d_model),
        "model.layers.0.mlp.up_proj.weight": torch.randn(16, d_model),
        "model.layers.0.mlp.down_proj.weight": torch.randn(d_model, 16),
    }
    model = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=2, num_key_value_heads=1, hidden_size=d_model,
                                head_dim=4),
        state_dict=lambda: sd,
    )
    path = tmp_path / "decomp.pt"

    decomp = screen._load_or_build_decomp(model, str(path), "toy", None)

    assert path.exists()
    assert list(tmp_path.glob("*.tmp*")) == []
    blob = torch.load(path)
    assert blob["model"] == "toy" and blob["adapter"] is None
    assert set(decomp.entries.keys()) == set(blob["entries"].keys())


def test_select_top_k_orders_by_activity_across_keys():
    activity_sum = {
        "k1": torch.tensor([3.0, 1.0]),
        "k2": torch.tensor([2.0, 5.0, 0.5]),
    }
    top = screen._select_top_k(activity_sum, k=3)
    assert top == [("k2", 1), ("k1", 0), ("k2", 0)]


class StubTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"1": [11], "2": [22]}[text]


def test_loss_gap_is_answer_logit_diff():
    logits = torch.zeros(3, 50)
    logits[-1, 11] = 2.0
    logits[-1, 22] = 0.5
    loss = screen._loss(logits, torch.zeros(3, dtype=torch.long), "gap", StubTokenizer())
    assert loss.item() == 1.5


def test_loss_nll_is_shifted_cross_entropy():
    torch.manual_seed(2)
    logits = torch.randn(4, 10)
    ids = torch.randint(0, 10, (4,))
    loss = screen._loss(logits, ids, "nll", None)
    expected = torch.nn.functional.cross_entropy(logits[:-1], ids[1:])
    assert torch.allclose(loss, expected)


def test_resolve_text_dispatches_on_loss_kind_and_entry_shape():
    class T:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
            assert add_generation_prompt is False
            return "RENDERED:" + msgs[-1]["content"]

    assert screen._resolve_text("already rendered", "gap", T()) == "already rendered"
    msg_entry = {"kind": "summarise", "messages": [{"role": "user", "content": "hi"}]}
    assert screen._resolve_text(msg_entry, "nll", T()) == "RENDERED:hi"
    text_entry = {"kind": "continuation", "text": "raw text"}
    assert screen._resolve_text(text_entry, "nll", T()) == "raw text"


class ToyCausalModel(torch.nn.Module):
    """Minimal get_input_embeddings()/forward(inputs_embeds=...) surface run_screen needs, with a
    plain nn.Linear as the screened module — qwen_entry_map's real (Qwen-specific) layer traversal
    is monkeypatched out below, same toy-decomp-and-entry_map pattern as the rest of this file."""

    def __init__(self, vocab, d_in, d_out):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d_in)
        self.lin = torch.nn.Linear(d_in, d_out, bias=False)
        self.head = torch.nn.Linear(d_out, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None):
        # both of run_screen's passes are gradient-enabled now (see _per_prompt_matrices) and
        # call inputs_embeds via _logits_with_grad, like a real HF causal LM would; input_ids
        # support stays here only because it's the cheaper/more direct kwarg for tests that
        # don't need grad (this class is also used as a plain input_ids-only stand-in elsewhere).
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        return SimpleNamespace(logits=self.head(self.lin(inputs_embeds)))


class CharTokenizer:
    """1 char = 1 token, so text length directly controls token count for the length-cap test."""

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[ord(c) for c in text]])}


def test_run_screen_tracks_kept_row_idx_when_a_middle_prompt_is_skipped(monkeypatch):
    """The alignment logic this task exists to protect: run_screen must record, per surviving
    batch entry, its ORIGINAL position in `batches` (not just a count) — so a downstream consumer
    can re-join per_prompt_activity rows back to the pairs df those batches came from, even when
    an over-length prompt in the MIDDLE of the batch got dropped."""
    torch.manual_seed(5)
    d_in, d_out, vocab = 6, 4, 128
    model = ToyCausalModel(vocab, d_in, d_out)

    U, S, Vh = torch.linalg.svd(model.lin.weight.data.float(), full_matrices=False)
    decomp = ToyDecomp(entries={"w": (U, S, Vh.T)}, meta={"w": {"head": None}})
    entry_map = {"w": (model.lin, False)}
    monkeypatch.setattr(screen, "qwen_entry_map", lambda model, decomp: entry_map)

    # position 1 (length 8) exceeds the batch_tokens cap of 5; positions 0 and 2 don't.
    batches = [{"text": "aaa"}, {"text": "a" * 8}, {"text": "aaaa"}]
    c = {"screen": {"batch_tokens": 5, "store_top_k": 10, "per_prompt_budget_mb": 2000}}

    result = screen.run_screen(model, CharTokenizer(), decomp, batches, "nll", c)

    assert result.kept_row_idx == [0, 2]
    assert result.n_prompts == 2
    assert result.prompt_tokens == [3, 4]  # order preserved: batch[0] then batch[2]
    assert result.per_prompt_activity.shape == (2, min(d_in, d_out))
    assert result.per_prompt_attr.shape == (2, min(d_in, d_out))
    assert result.per_prompt_proj.shape == (2, min(d_in, d_out))
    assert result.invert is False  # default, not passed here


def test_run_screen_records_invert_flag(monkeypatch):
    torch.manual_seed(8)
    d_in, d_out, vocab = 4, 3, 128  # 128: CharTokenizer below produces ord(c)-valued ids
    model = ToyCausalModel(vocab, d_in, d_out)
    U, S, Vh = torch.linalg.svd(model.lin.weight.data.float(), full_matrices=False)
    decomp = ToyDecomp(entries={"w": (U, S, Vh.T)}, meta={"w": {"head": None}})
    entry_map = {"w": (model.lin, False)}
    monkeypatch.setattr(screen, "qwen_entry_map", lambda model, decomp: entry_map)

    batches = [{"text": "aaa"}]
    c = {"screen": {"batch_tokens": 10, "store_top_k": 5, "per_prompt_budget_mb": 2000}}

    result = screen.run_screen(model, CharTokenizer(), decomp, batches, "nll", c, invert=True)
    assert result.invert is True


def test_run_screen_single_pass_matches_two_pass_per_prompt_matrices(monkeypatch):
    """The whole point of this task: single-pass mode (per_prompt_budget_mb generous) must
    produce per_prompt_activity/attr/proj identical (within fp tolerance) to the pre-existing
    two-pass path (budget forced to 0) — same formulas, same fp32, just captured during pass 1
    instead of a second forward+backward. Two keys sliced from the same Linear (like real
    per-head q/k/v/o entries) exercise the sorted-key column-offset math, not just a single key
    at offset 0."""
    torch.manual_seed(11)
    d_in, d_out, vocab = 8, 8, 128  # 128: CharTokenizer produces ord(c)-valued ids
    model = ToyCausalModel(vocab, d_in, d_out)
    width = 4  # two width-4 "heads" sliced from the same Linear's output rows
    U0, S0, Vh0 = torch.linalg.svd(model.lin.weight.data[0:width, :].float(), full_matrices=False)
    U1, S1, Vh1 = torch.linalg.svd(model.lin.weight.data[width:2 * width, :].float(), full_matrices=False)
    decomp = ToyDecomp(
        entries={"L0.w.h0": (U0, S0, Vh0.T), "L0.w.h1": (U1, S1, Vh1.T)},
        meta={"L0.w.h0": {"head": 0}, "L0.w.h1": {"head": 1}},
    )
    entry_map = {"L0.w.h0": (model.lin, False), "L0.w.h1": (model.lin, False)}
    monkeypatch.setattr(screen, "qwen_entry_map", lambda model, decomp: entry_map)

    batches = [{"text": "a" * n} for n in (3, 5, 4, 6)]

    def run(budget_mb):
        c = {"screen": {"batch_tokens": 20, "store_top_k": 3, "per_prompt_budget_mb": budget_mb}}
        return screen.run_screen(model, CharTokenizer(), decomp, batches, "nll", c)

    result_single = run(budget_mb=10_000)  # generous -> single-pass
    result_two = run(budget_mb=0)  # any nonzero store exceeds a 0 budget -> two-pass

    assert result_single.top_k_components == result_two.top_k_components
    assert torch.allclose(result_single.per_prompt_activity, result_two.per_prompt_activity, atol=1e-5)
    assert torch.allclose(result_single.per_prompt_attr, result_two.per_prompt_attr, atol=1e-5)
    assert torch.allclose(result_single.per_prompt_proj, result_two.per_prompt_proj, atol=1e-5)


def test_screen_result_save_includes_new_keys(tmp_path):
    r = screen.ScreenResult({}, {}, {}, 0, 1, [("w", 0)], torch.zeros(1, 1), [3],
                             [0], None, None, "judging", "dev", None,
                             per_prompt_attr=torch.zeros(1, 1), per_prompt_proj=torch.zeros(1, 1),
                             invert=True)
    path = tmp_path / "r.pt"
    r.save(path)
    saved = torch.load(path)
    assert "per_prompt_attr" in saved and "per_prompt_proj" in saved and "invert" in saved
    assert saved["invert"] is True


def test_screen_result_defaults_for_new_fields_when_omitted():
    """A call site that builds ScreenResult without the new kwargs (mirrors any pre-task-14
    caller) still gets sane defaults, not a TypeError."""
    r = screen.ScreenResult({}, {}, {}, 0, 0, [], torch.zeros(0, 0), [], [], None, None, None, None, None)
    assert r.per_prompt_attr is None
    assert r.per_prompt_proj is None
    assert r.invert is False


def test_legacy_screen_dict_without_new_fields_still_consumed(monkeypatch):
    """Screens saved before this task lack per_prompt_attr/per_prompt_proj/invert entirely —
    case_profile._load_prompt_meta and importance_report.merge_table must not require them."""
    import case_profile
    import importance_report
    import pandas as pd

    legacy = {"activity": {"L0.gate": torch.tensor([1.0])}, "attr_signed": {"L0.gate": torch.tensor([0.5])},
              "attr_abs": {"L0.gate": torch.tensor([0.5])}, "n_tokens": 10, "n_prompts": 1,
              "top_k_components": [("L0.gate", 0)], "per_prompt_activity": torch.zeros(1, 1),
              "prompt_tokens": [10], "kept_row_idx": [0], "split": None, "limit": None,
              "set_name": "judging", "model": None, "adapter": None}
    assert "per_prompt_attr" not in legacy and "per_prompt_proj" not in legacy and "invert" not in legacy

    monkeypatch.setattr(case_profile.wbpairs, "load_pairs",
                         lambda c, split=None, limit=None: pd.DataFrame({"x": [1]}))
    meta = case_profile._load_prompt_meta(legacy, {})
    assert len(meta) == 1

    df = importance_report.merge_table(legacy, dict(legacy), eps=1e-9)
    assert len(df) == 1


def test_main_invert_rejects_control_set(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["screen.py", "--model", "dev", "--set", "control",
                                       "--invert", "--out", "out.pt"])
    with pytest.raises(AssertionError, match="judging-set only"):
        screen.main()
