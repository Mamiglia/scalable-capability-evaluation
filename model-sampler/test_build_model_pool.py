import json
from build_model_pool import extract_base, blank_record, norm, half_of, join_openrouter, join_epoch, best_scores, impute_eci, merge_overrides, inject_pool

FAKE_HTML = """<script>
const DATA = /*DATA*/[{"name":"A","dev":"OpenAI","eci":130.5,"release":"2025-01-01","half":"2025-H1","desc":"d","est":false}];
const EST = /*EST*/[{"name":"B","dev":"Other","eci":118.0,"release":"2024-06-01","half":"2024-H1","desc":"","est":true}];
</script>"""

def test_extract_base():
    recs = extract_base(FAKE_HTML)
    assert [r["name"] for r in recs] == ["A", "B"]
    assert recs[0]["eci_source"] == "real" and recs[1]["eci_source"] == "aa"
    assert recs[0]["eci"] == 130.5
    # join fields defaulted
    assert recs[0]["params"] is None and recs[0]["on_openrouter"] is False
    assert recs[0]["price"] is None and recs[0]["tier_manual"] is None

def test_extract_base_real_page():
    html = open("../interactive_viewer/model_select.html").read()
    recs = extract_base(html)
    assert len(recs) > 100
    assert all(r["eci_source"] in ("real", "aa") for r in recs)

def test_norm():
    assert norm("Qwen 3 32B") == norm("Qwen3-32B") == "qwen332b"
    assert norm("Llama 2-70B") == "llama270b"

def test_half_of():
    assert half_of("2023-07-18") == "2023-H2"
    assert half_of("2024-01-02") == "2024-H1"

def _or_model(mid, name, prompt, completion):
    return {"id": mid, "name": name,
            "pricing": {"prompt": str(prompt), "completion": str(completion)}}

def test_join_openrouter_prices_and_free_preference():
    pool = [blank_record("Qwen3 32B", "Alibaba (Qwen)", 130, "real", "2025-04-01", "2025-H1")]
    ors = [_or_model("qwen/qwen3-32b:free", "Qwen: Qwen3 32B (free)", 0, 0),
           _or_model("qwen/qwen3-32b", "Qwen: Qwen3 32B", 1e-7, 3e-7)]
    join_openrouter(pool, ors)
    m = pool[0]
    assert m["on_openrouter"] and m["or_id"] == "qwen/qwen3-32b"
    assert abs(m["price_in"] - 0.1) < 1e-9 and abs(m["price_out"] - 0.3) < 1e-9
    assert abs(m["price"] - 0.4) < 1e-9

def test_join_epoch_enriches_and_appends():
    pool = [blank_record("Llama 2-70B", "Meta", None, None, "2023-07-18", "2023-H2")]
    rows = [
        {"Model": "Llama 2-70B", "Organization": "Meta AI", "Publication date": "2023-07-18",
         "Parameters": "70000000000.0", "Open model weights?": "Yes", "Domain": "Language"},
        {"Model": "Qwen1.5-7B", "Organization": "Alibaba", "Publication date": "2024-02-04",
         "Parameters": "7000000000.0", "Open model weights?": "Yes", "Domain": "Language"},
        {"Model": "TooOld-1B", "Organization": "X", "Publication date": "2020-01-01",
         "Parameters": "1000000000.0", "Open model weights?": "Yes", "Domain": "Language"},
        {"Model": "Closed-2B", "Organization": "X", "Publication date": "2024-01-01",
         "Parameters": "2000000000.0", "Open model weights?": "No", "Domain": "Language"},
    ]
    join_epoch(pool, rows, min_release="2022-01-01", max_params=200e9,
               extra=("Qwen1.5-7B", "TooOld-1B", "Closed-2B"))
    assert pool[0]["params"] == 70e9 and pool[0]["open_weights"] is True
    names = [m["name"] for m in pool]
    assert "Qwen1.5-7B" in names and "TooOld-1B" not in names and "Closed-2B" not in names
    added = next(m for m in pool if m["name"] == "Qwen1.5-7B")
    assert added["eci"] is None and added["half"] == "2024-H1" and added["dev"] == "Alibaba (Qwen)"

def _bench(task, model, score):
    return {"task": task, "Model": model, "mean_score": str(score), "best_score": ""}

def test_best_scores_keeps_max():
    rows = [_bench("GPQA diamond", "A", 0.5), _bench("GPQA diamond", "A", 0.6)]
    assert best_scores(rows)["GPQA diamond"]["A"] == 0.6

def test_impute_linear():
    # eci = 100*score + 100 exactly; imputed model must land on the line
    pool, rows = [], []
    for i, s in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
        pool.append(blank_record(f"M{i}", "Other", 100 * s + 100, "real", "2025-01-01", "2025-H1"))
        rows.append(_bench("GPQA diamond", f"M{i}", s))
    pool.append(blank_record("New", "Other", None, None, "2023-01-01", "2023-H1"))
    rows.append(_bench("GPQA diamond", "New", 0.45))
    task = impute_eci(pool, rows, min_dual=8)
    assert task == "GPQA diamond"
    new = next(m for m in pool if m["name"] == "New")
    assert new["eci_source"] == "imputed" and abs(new["eci"] - 145.0) < 1e-6

def test_impute_insufficient_dual_exits():
    import pytest
    pool = [blank_record("A", "Other", 130, "real", "2025-01-01", "2025-H1")]
    with pytest.raises(SystemExit):
        impute_eci(pool, [_bench("GPQA diamond", "A", 0.5)], min_dual=8)

def test_merge_overrides():
    pool = [blank_record("A", "Other", 130, "real", "2025-01-01", "2025-H1")]
    merge_overrides(pool, {"A": "Weak", "Nope": "Strong"})
    assert pool[0]["tier_manual"] == "Weak"

def test_inject_pool_roundtrip():
    html = "<script>\nconst POOL = /*POOL*/[];\n</script>"
    out = inject_pool(html, [{"name": "A"}])
    assert '/*POOL*/[{"name": "A"}];' in out
    # idempotent: inject again over the previous payload
    out2 = inject_pool(out, [{"name": "B"}])
    assert '"B"' in out2 and '"A"' not in out2

import io, zipfile
from build_model_pool import (gpqa_mean_tokens, eval_stem_clean, join_gpqa_tokens,
                              join_reasoning)

def _fake_eval(tmp_path, name, toks):
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as z:
        for i, t in enumerate(toks):
            z.writestr(f"samples/rec{i}_epoch_1.json",
                       json.dumps({"output": {"usage": {"output_tokens": t}}}))
        z.writestr("_journal/start.json", "{}")
    return p

def test_gpqa_mean_tokens(tmp_path):
    p = _fake_eval(tmp_path, "m.eval", [100, 200])
    assert gpqa_mean_tokens(p, cap=60) == 150.0
    empty = _fake_eval(tmp_path, "e.eval", [])
    assert gpqa_mean_tokens(empty, cap=60) is None

def test_eval_stem_clean():
    assert eval_stem_clean("claude-3-5-haiku-20241022") == "claude-3-5-haiku"
    assert eval_stem_clean("Meta-Llama-3-70B-Instruct") == "Llama-3-70B"
    assert eval_stem_clean("Llama-2-70b-chat-hf") == "Llama-2-70b"
    assert eval_stem_clean("DeepSeek-V3") == "DeepSeek-V3"

def test_join_gpqa_tokens_skips_budget_variants(tmp_path):
    _fake_eval(tmp_path, "some-model.eval", [100])
    _fake_eval(tmp_path, "some-model_32K.eval", [999])
    pool = [blank_record("Some Model", "Other", 120, "real", "2025-01-01", "2025-H1")]
    assert join_gpqa_tokens(pool, tmp_path) == 1
    assert pool[0]["gpqa_tok"] == 100.0

def test_join_reasoning():
    pool = [blank_record("Alpha", "Other", 120, "real", "2025-01-01", "2025-H1"),
            blank_record("Beta R1", "Other", 120, "real", "2025-01-01", "2025-H1"),
            blank_record("Gamma", "Other", 120, "real", "2025-01-01", "2025-H1")]
    ors = [{"id": "x/alpha", "name": "X: Alpha",
            "pricing": {"prompt": "0", "completion": "0"},
            "supported_parameters": ["temperature", "reasoning"]}]
    join_reasoning(pool, ors)
    assert pool[0]["reasoning"] is True     # from OpenRouter params
    assert pool[1]["reasoning"] is True     # name heuristic
    assert pool[2]["reasoning"] is None     # unknown
