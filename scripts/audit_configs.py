"""Check every model config against the three rules in configs/models/PROVIDER_PINNING.md.

    uv run python scripts/audit_configs.py [--json]

1. Pinned      -- exactly one `only:` endpoint tag plus `allow_fallbacks: false`.
2. Capacity    -- `max_tokens` fits the pinned endpoint, and no endpoint on the slug
                  offers materially more room (this outranks rule 3).
3. Precision   -- nothing on the slug is served at a higher precision.
4. Reasoning   -- enabled whenever the pinned endpoint supports it.

A `# deliberate:` comment anywhere in a config downgrades its findings to notes: the
tradeoff was looked at and taken on purpose. Exits 1 if any config fails a rule.
"""

import argparse
import json
import urllib.request
from pathlib import Path

import yaml

CONFIGS = Path("configs/models")
# higher is better; anything undeclared is unranked and never counts as an improvement
RANK = {"fp32": 4, "fp16": 3, "bf16": 3, "fp8": 2, "int8": 2, "fp6": 2,
        "fp4": 1, "int4": 1, "mxfp4": 1, "nvfp4": 1}
SLACK = 1.5  # a longer endpoint is only worth flagging at this multiple


def endpoints(slug: str) -> list[dict]:
    url = f"https://openrouter.ai/api/v1/models/{slug}/endpoints"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)["data"]["endpoints"]


def room(ep: dict) -> int | None:
    """Declared output cap. None means the provider declares none -- unknown, not unlimited."""
    return ep.get("max_completion_tokens")


def check(path: Path) -> dict | None:
    cfg = yaml.safe_load(path.read_text()) or {}
    slug = str(cfg.get("slug") or "")
    prov = ((cfg.get("model_args") or {}).get("provider")) or {}
    if not slug.startswith("openrouter/") and not prov:
        return None  # local vLLM or a first-party SDK route; no routing to check
    gen = cfg.get("generate_args") or {}
    only, fails = prov.get("only") or [], []

    if len(only) != 1 or prov.get("allow_fallbacks") is not False:
        fails.append(f"not pinned: only={only or None} allow_fallbacks={prov.get('allow_fallbacks')}")

    eps = endpoints(slug.split("/", 1)[1] if slug.startswith("openrouter/") else cfg["name"])
    tags = {e["tag"]: e for e in eps}
    # an exact tag match is the pin; a prefix match means the name still spans several endpoints
    exact = [tags[o] for o in only if o in tags]
    spanned = [e for o in only for e in eps if o not in tags and e["tag"].split("/")[0] == o.lower()]
    if only and not exact and not spanned:
        fails.append(f"pin {only[0]!r} matches no live endpoint")
    if len(spanned) > 1:  # spanning exactly one endpoint is unambiguous, just unlabelled
        fails.append(f"pin {only[0]!r} spans {len(spanned)}: {[e['tag'] for e in spanned]}")
    if not (ep := (exact or spanned or [None])[0]):
        return {"config": str(path), "fails": fails, "deliberate": False}

    # 2. capacity, then 3. precision -- capacity first, it is the one that truncates output.
    # Only compare declared caps against declared caps; an undeclared cap is unknown, not better.
    if (mt := gen.get("max_tokens")) and room(ep) and mt > room(ep):
        fails.append(f"max_tokens {mt} exceeds {ep['tag']} cap {room(ep)}")
    if room(ep) and (roomier := [e for e in eps if (room(e) or 0) > room(ep) * SLACK]):
        best = max(roomier, key=room)
        fails.append(f"{best['tag']} ({best.get('quantization')}) caps at {room(best)} "
                     f"vs {ep['tag']} {room(ep)}")
    if mine := RANK.get(ep.get("quantization") or "", 0):  # undeclared: nothing to compare against
        if finer := [e for e in eps if RANK.get(e.get("quantization") or "", 0) > mine]:
            fails.append(f"{finer[0]['tag']} is {finer[0]['quantization']}, pin is {ep['quantization']}")

    # 4. reasoning -- set it explicitly rather than inheriting whatever the provider defaults to
    if "reasoning" in (ep.get("supported_parameters") or []):
        if "reasoning" not in (gen.get("extra_body") or {}):
            fails.append(f"{ep['tag']} takes a reasoning param; config leaves it at the provider default")

    return {"config": str(path), "fails": fails, "deliberate": "# deliberate:" in path.read_text()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = [r for p in sorted(CONFIGS.rglob("*.yml")) if (r := check(p))]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            for f in r["fails"]:
                print(f"{'note ' if r['deliberate'] else 'FAIL '} {r['config']}: {f}")
        bad = sum(1 for r in rows if r["fails"] and not r["deliberate"])
        print(f"\n{len(rows)} routed configs, {bad} failing")
    return 1 if any(r["fails"] and not r["deliberate"] for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
