"""Build model_pool.json: base list from the selector page + OpenRouter pricing
+ Epoch models DB (params, open weights) + pseudo-ECI imputed from Epoch
benchmark runs. Injects the pool into index.html."""
import argparse, csv, json, re, statistics, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
SELECTOR_HTML = HERE.parent / "interactive_viewer" / "model_select.html"
CACHE_DIR = HERE / "cache"
OR_MODELS_URL = "https://openrouter.ai/api/v1/models"
EPOCH_MODELS_URL = "https://epoch.ai/data/all_ai_models.csv"
EPOCH_BENCH_URL = "https://epoch.ai/data/benchmarks.csv"

JOIN_DEFAULTS = dict(params=None, open_weights=False, on_openrouter=False,
                     or_id=None, price_in=None, price_out=None, price=None,
                     tier_manual=None, gpqa_tok=None, reasoning=None)

GPQA_EVAL_DIR = HERE.parent / "external_data" / "epoch-gpqa"
TOK_SAMPLE_CAP = 60
REASONING_NAME_KEYS = ("r1", "o1", "o3", "o4", "qwq", "think", "reason")

def blank_record(name, dev, eci, eci_source, release, half, desc=""):
    r = dict(name=name, dev=dev, eci=eci, eci_source=eci_source,
             release=release, half=half, desc=desc)
    r.update(JOIN_DEFAULTS)
    return r

def _array_after(html, marker):
    # arrays sit on one line; greedy-to-last "];" so ";" inside desc strings can't truncate
    m = re.search(re.escape(marker) + r"(\[.*\]);", html)
    return json.loads(m.group(1))

def extract_base(html_text):
    recs = []
    for marker, source in (("/*DATA*/", "real"), ("/*EST*/", "aa")):
        for d in _array_after(html_text, marker):
            recs.append(blank_record(d["name"], d["dev"], d["eci"], source,
                                     d["release"], d["half"], d.get("desc", "")))
    return recs

def fetch(url, dest, refresh=False):
    dest.parent.mkdir(exist_ok=True)
    if dest.exists() and not refresh:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "model-sampler/1.0"})
    dest.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    return dest

def norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())

# norm(other-source name) -> base name; extend when the build summary reports misses
ALIASES = {
    norm("Llama 3.1 70B Instruct"): "Llama 3.1-70B",
    norm("Llama 3.1 8B Instruct"): "Llama 3.1-8B",
    norm("Llama 3.3 70B Instruct"): "Llama 3.3 70B",
    norm("Qwen2.5 72B Instruct"): "Qwen2.5-72B",
    norm("Mixtral 8x22B Instruct"): "Mixtral 8x22B",
    norm("Mistral Small 3.1 24B"): "Mistral Small 3.1",
    norm("Mistral Small 3.2 24B"): "Mistral Small 3.2",
    norm("Mistral Large 2407"): "Mistral Large 2",
    norm("Mistral Large 3 2512"): "Mistral Large 3",
    norm("Ministral 3 14B 2512"): "Ministral 3 14B",
    norm("Ministral 3 3B 2512"): "Ministral 3 3B",
    norm("Ministral 3 8B 2512"): "Ministral 3 8B",
    norm("Qwen3 Coder 480B A35B"): "Qwen3 Coder 480B A35B Instruct",
    norm("Qwen3.5-Flash"): "Qwen 3.5 Flash (hosted 35B-A3B)",
    # GPQA eval-log stems (post eval_stem_clean) -> base name
    norm("Phi-3-medium-128k"): "phi-3-medium 14B",
    norm("claude-2.0"): "Claude 2",
    norm("claude-opus-4-7_xhigh"): "Claude Opus 4.7",
    norm("claude-sonnet-4-6_high"): "Claude Sonnet 4.6",
    norm("gemini-1.0-pro-001"): "Gemini 1.0 Pro",
    norm("gemini-1.5-flash-001"): "Gemini 1.5 Flash",
    norm("gemini-1.5-pro-001"): "Gemini 1.5 Pro",
    norm("gemini-2.0-flash-001"): "Gemini 2.0 Flash",
    norm("gemini-2.0-flash-thinking-exp-01-21"): "Gemini 2.0 Flash Thinking",
    norm("gemini-2.0-pro-exp-02-05"): "Gemini 2.0 Pro",
    norm("gemini-2.5-pro-exp-03-25"): "Gemini 2.5 Pro (Mar 2025)",
    norm("gemini-2.5-pro-preview-05-06"): "Gemini 2.5 Pro (May 2025)",
    norm("gemini-2.5-pro-preview-06-05"): "Gemini 2.5 Pro (Jun 2025)",
    norm("gemini-3-flash-preview"): "Gemini 3 Flash",
    norm("gemini-3-pro-preview"): "Gemini 3 Pro",
    norm("gemini-3.1-pro-preview"): "Gemini 3.1 Pro",
    norm("gemini-3.5-flash_high"): "Gemini 3.5 Flash",
    norm("gemma-2-27b-it"): "Gemma 2 27B",
    norm("gemma-2-9b-it"): "Gemma 2 9B",
    norm("gemma-3-27b-it"): "Gemma 3 27B",
    norm("gpt-4-0314"): "GPT-4 (Mar 2023)",
    norm("gpt-4-0613"): "GPT-4 (Jun 2023)",
    norm("gpt-4-1106-preview"): "GPT-4 Turbo (Nov 2023)",
    norm("gpt-4-turbo-2024-04-09"): "GPT-4 Turbo (Apr 2024)",
    norm("gpt-4.1-2025-04-14"): "GPT-4.1",
    norm("gpt-4.1-mini-2025-04-14"): "GPT-4.1 mini",
    norm("gpt-4.1-nano-2025-04-14"): "GPT-4.1 nano",
    norm("gpt-4.5-preview-2025-02-27"): "GPT-4.5",
    norm("gpt-4o-2024-05-13"): "GPT-4o",
    norm("gpt-4o-mini-2024-07-18"): "GPT-4o mini",
    norm("gpt-5-2025-08-07_high"): "GPT-5",
    norm("gpt-5-mini-2025-08-07_high"): "GPT-5 mini",
    norm("gpt-5-nano-2025-08-07_high"): "GPT-5 nano",
    norm("gpt-5.1-2025-11-13_high"): "GPT-5.1",
    norm("gpt-5.2-2025-12-11_high"): "GPT-5.2",
    norm("gpt-5.4-2026-03-05_high"): "GPT-5.4",
    norm("gpt-5.4-mini-2026-03-17_high"): "GPT-5.4 Mini",
    norm("gpt-5.4-nano-2026-03-17_high"): "GPT-5.4 Nano",
    norm("gpt-5.5-pre-release_xhigh"): "GPT-5.5",
    norm("gpt-5.5-pro-pre-release_xhigh"): "GPT-5.5 Pro",
    norm("grok-2-1212"): "Grok-2",
    norm("grok-3-beta"): "Grok 3",
    norm("grok-3-mini-beta_high"): "Grok-3 mini",
    norm("kimi-k2-thinking-turbo"): "Kimi K2 Thinking",
    norm("magistral-small-2506"): "Magistral Small 1.0",
    norm("mistral-large-2402"): "Mistral Large",
    norm("mistral-large-2411"): "Mistral Large 2",
    norm("mistral-medium-2505"): "Mistral Medium 3",
    norm("mistral-small-2501"): "Mistral Small 3",
    norm("mistral-small-2503"): "Mistral Small 3.1",
    norm("o1-2024-12-17_high"): "o1",
    norm("o1-mini-2024-09-12_high"): "o1-mini",
    norm("o1-preview-2024-09-12"): "o1-preview",
    norm("o3-2025-04-16_high"): "o3",
    norm("o3-mini-2025-01-31_high"): "o3-mini",
    norm("o4-mini-2025-04-16_high"): "o4-mini",
    norm("open-mistral-7b"): "Mistral 7B",
    norm("open-mistral-nemo-2407"): "Mistral NeMo",
    norm("open-mixtral-8x22b"): "Mixtral 8x22B",
    norm("open-mixtral-8x7b"): "Mixtral 8x7B",
    norm("qwen-max-2025-01-25"): "Qwen2.5-Max",
    norm("qwen3-max-2025-09-23"): "Qwen3-Max",
    norm("qwen3.5-plus"): "Qwen 3.5 Plus (hosted 397B-A17B)",
    norm("Yi-1.5-34B"): "Yi-34B",
}

ORG_MAP = {"Meta AI": "Meta", "OpenAI": "OpenAI", "Google": "Google",
           "Google DeepMind": "Google", "Anthropic": "Anthropic",
           "Alibaba": "Alibaba (Qwen)", "DeepSeek": "DeepSeek",
           "Mistral AI": "Mistral", "xAI": "xAI", "Microsoft": "Microsoft",
           "01.AI": "01.AI", "Zhipu AI": "Zhipu (GLM)", "Amazon": "Amazon",
           "IBM": "IBM (Granite)", "Allen Institute for AI": "AllenAI (OLMo)"}

def half_of(date):
    y, m = date[:4], int(date[5:7])
    return f"{y}-H{1 if m <= 6 else 2}"

def _index(pool):
    idx = {}
    for m in pool:
        idx[norm(m["name"])] = m
    return idx

def _base_lookup(idx, other_name):
    n = norm(other_name)
    if n in ALIASES:
        n = norm(ALIASES[n])
    return idx.get(n)

def gpqa_mean_tokens(eval_path, cap=TOK_SAMPLE_CAP):
    import zipfile
    toks = []
    with zipfile.ZipFile(eval_path) as z:
        for n in [x for x in z.namelist() if x.startswith("samples/")][:cap]:
            usage = (json.loads(z.read(n)).get("output") or {}).get("usage") or {}
            if usage.get("output_tokens") is not None:
                toks.append(usage["output_tokens"])
    return round(sum(toks) / len(toks), 1) if toks else None

def eval_stem_clean(stem):
    stem = re.sub(r"-\d{8}$", "", stem)
    stem = re.sub(r"^Meta-", "", stem)
    stem = re.sub(r"-(Instruct|instruct)([-_].*)?$", "", stem)
    stem = re.sub(r"-(chat-hf|chat|Chat)$", "", stem)
    return stem

def join_gpqa_tokens(pool, eval_dir=GPQA_EVAL_DIR):
    idx = _index(pool)
    matched = 0
    for p in sorted(eval_dir.glob("*.eval")):
        if re.search(r"_\d+K$", p.stem):
            continue
        m = _base_lookup(idx, eval_stem_clean(p.stem))
        if m is None or m["gpqa_tok"] is not None:
            continue
        tok = gpqa_mean_tokens(p)
        if tok is not None:
            m["gpqa_tok"] = tok
            matched += 1
    return matched

def join_reasoning(pool, or_models):
    idx = _index(pool)
    for om in or_models:
        name = re.sub(r"\s*\(free\)$", "", om["name"].split(": ", 1)[-1])
        m = _base_lookup(idx, name)
        if m is not None and m["reasoning"] is None:
            m["reasoning"] = "reasoning" in (om.get("supported_parameters") or [])
    for m in pool:
        if m["reasoning"] is None and any(k in norm(m["name"]) for k in REASONING_NAME_KEYS):
            m["reasoning"] = True

def join_openrouter(pool, or_models):
    idx = _index(pool)
    for om in or_models:
        # "Qwen: Qwen3 32B (free)" -> "Qwen3 32B"
        name = re.sub(r"\s*\(free\)$", "", om["name"].split(": ", 1)[-1])
        m = _base_lookup(idx, name)
        if m is None:
            continue
        free = om["id"].endswith(":free")
        if m["on_openrouter"] and free:
            continue  # keep the paid endpoint
        m["on_openrouter"] = True
        m["or_id"] = om["id"]
        m["price_in"] = float(om["pricing"]["prompt"]) * 1e6
        m["price_out"] = float(om["pricing"]["completion"]) * 1e6
        m["price"] = m["price_in"] + m["price_out"]

# open-weights models added beyond the selector base list; research staples only
EXTRA_OPEN_WEIGHTS = ("Olmo 3 32B Instruct", "OLMo 2 32B")

def join_epoch(pool, epoch_rows, min_release, max_params, extra=EXTRA_OPEN_WEIGHTS):
    idx = _index(pool)
    for row in epoch_rows:
        name, date = row.get("Model", ""), row.get("Publication date", "")
        try:
            params = float(row.get("Parameters") or "")
        except ValueError:
            params = None
        ow = row.get("Open model weights?") == "Yes"
        m = _base_lookup(idx, name)
        if m is not None:
            if m["params"] is None and params is not None:
                m["params"] = params
            m["open_weights"] = m["open_weights"] or ow
            continue
        if not (ow and name in extra and date >= min_release
                and params is not None and params <= max_params):
            continue
        dev = ORG_MAP.get(row.get("Organization", ""), "Other")
        rec = blank_record(name, dev, None, None, date, half_of(date))
        rec["params"], rec["open_weights"] = params, True
        pool.append(rec)
        idx[norm(name)] = rec

MIN_DUAL_POINTS = 8

def best_scores(bench_rows):
    """Extract max score per task per model from benchmark rows."""
    out = {}
    for r in bench_rows:
        raw = r.get("mean_score") or r.get("best_score") or ""
        try:
            score = float(raw)
        except ValueError:
            continue
        task = out.setdefault(r.get("task", ""), {})
        name = r.get("Model", "")
        task[name] = max(task.get(name, score), score)
    return out

def impute_eci(pool, bench_rows, min_dual=MIN_DUAL_POINTS):
    """Fit linear regression on benchmark scores vs ECI; impute for missing ECIs."""
    by_task = best_scores(bench_rows)
    idx = _index(pool)

    def dual(task_scores):
        pairs = []
        for name, score in task_scores.items():
            m = _base_lookup(idx, name)
            if m is not None and m["eci"] is not None:
                pairs.append((score, m["eci"]))
        return pairs

    task = max(by_task, key=lambda t: len(dual(by_task[t])))
    pairs = dual(by_task[task])
    if len(pairs) < min_dual:
        sys.exit(f"imputation: only {len(pairs)} models have both a '{task}' score "
                 f"and an ECI (need {min_dual})")
    fit = statistics.linear_regression([p[0] for p in pairs], [p[1] for p in pairs])
    for name, score in by_task[task].items():
        m = _base_lookup(idx, name)
        if m is not None and m["eci"] is None:
            m["eci"] = round(fit.slope * score + fit.intercept, 2)
            m["eci_source"] = "imputed"
    return task

def merge_overrides(pool, overrides):
    idx = _index(pool)
    missing = []
    for name, tier in overrides.items():
        m = idx.get(norm(name))
        if m is None:
            missing.append(name)
        else:
            m["tier_manual"] = tier
    if missing:
        print("overrides not matched:", ", ".join(missing))

def inject_pool(html_text, pool):
    # replace to end of line (json.dumps emits no newlines); a non-greedy ".*?;" would
    # stop at the first ";" inside a desc string and corrupt the page
    payload = "/*POOL*/" + json.dumps(pool) + ";"
    return re.sub(r"/\*POOL\*/.*", lambda _: payload, html_text, count=1)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="refetch cached downloads")
    ap.add_argument("--min-release", default="2022-01-01",
                    help="oldest publication date for Epoch-added open-weights models")
    ap.add_argument("--max-params", type=float, default=200e9,
                    help="largest Epoch-added model, params")
    ap.add_argument("--min-dual", type=int, default=MIN_DUAL_POINTS,
                    help="min models with both benchmark score and ECI for the imputation fit")
    args = ap.parse_args()

    pool = extract_base(SELECTOR_HTML.read_text())
    or_models = json.loads(fetch(OR_MODELS_URL, CACHE_DIR / "or_models.json",
                                 args.refresh).read_text())["data"]
    epoch_rows = list(csv.DictReader(open(fetch(EPOCH_MODELS_URL,
                                                CACHE_DIR / "epoch_models.csv", args.refresh))))
    bench_rows = list(csv.DictReader(open(fetch(EPOCH_BENCH_URL,
                                                CACHE_DIR / "epoch_benchmarks.csv", args.refresh))))

    n_base = len(pool)
    join_openrouter(pool, or_models)
    join_epoch(pool, epoch_rows, args.min_release, args.max_params)
    n_tok = join_gpqa_tokens(pool)
    join_reasoning(pool, or_models)
    task = impute_eci(pool, bench_rows, args.min_dual)

    ovr_path = HERE / "tier_overrides.json"
    if ovr_path.exists():
        merge_overrides(pool, json.loads(ovr_path.read_text()))

    (HERE / "model_pool.json").write_text(json.dumps(pool, indent=1))
    page = HERE / "index.html"
    if page.exists():
        page.write_text(inject_pool(page.read_text(), pool))

    from collections import Counter
    src = Counter(m["eci_source"] or "unscored" for m in pool)
    print(f"{len(pool)} models ({n_base} from selector page, "
          f"{len(pool) - n_base} added from Epoch)")
    print(f"eci_source: {dict(src)}")
    print(f"on_openrouter: {sum(m['on_openrouter'] for m in pool)}, "
          f"open_weights: {sum(m['open_weights'] for m in pool)}")
    print(f"imputation task: {task}")

    n_evals = len(list(GPQA_EVAL_DIR.glob("*.eval")))
    rsn = Counter("true" if m["reasoning"] else ("false" if m["reasoning"] is False else "unknown")
                  for m in pool)
    print(f"gpqa_tok matched: {n_tok} (of {n_evals} eval logs)")
    print(f"reasoning: {dict(rsn)}")

if __name__ == "__main__":
    main()
