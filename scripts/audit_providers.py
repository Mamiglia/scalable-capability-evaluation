"""Which provider actually served each request, and what it actually cost.

OpenRouter returns both in every response body, and inspect stores that body in each
sample's ModelEvent, so this is recoverable from logs already collected — nothing needs
to be instrumented. Use it to verify that a config's provider pin held:

    uv run python scripts/audit_providers.py data/gpqa [--since '2026-08-19 22:20'] [--samples 50]

A pin written as `order:` is only a preference: OpenRouter falls back to any other
provider when the preferred one is busy or errors, so a run can be served by an
endpoint at a different quantization than the config names. Only `only:` combined with
`allow_fallbacks: false` is binding.
"""

import argparse
import datetime
import os
from collections import Counter, defaultdict
from pathlib import Path

import re

import yaml

from inspect_ai.log import read_eval_log_samples

from configs.paths import is_successful_eval


def _key(name: str) -> str:
    """Provider names are spelled differently in pins and responses (NovitaAI/Novita, z-ai/Z.AI)."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def pinned_providers() -> dict[str, tuple[list[str], bool]]:
    """Per model shortname: the providers its config names, and whether the pin is binding."""
    pins = {}
    for line in Path("configs/experiments/pairs/models.txt").read_text().split():
        cfg = yaml.safe_load(open(line))
        prov = ((cfg.get("model_args") or {}).get("provider")) or {}
        named = prov.get("only") or prov.get("order") or []
        binding = bool(prov.get("only") and prov.get("allow_fallbacks") is False)
        # pins read like "venice/fp8" or "Novita"; compare on the provider part, lowercased
        pins[cfg["name"].split("/")[-1].lower()] = (
            [_key(str(p).split("/")[0]) for p in named],
            binding,
        )
    return pins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--since")
    parser.add_argument("--samples", type=int, default=1000, help="samples read per log")
    args = parser.parse_args()

    cutoff = datetime.datetime.fromisoformat(args.since).timestamp() if args.since else 0.0
    pins = pinned_providers()

    print(f"{'bench':10s}{'model':44s}{'rep':4s}{'served by (requests)':44s}{'cost$':>8s}  pin")
    for root in args.roots:
        for path in sorted(root.glob("*/*/*/*.eval")):
            if path.stat().st_mtime < cutoff or not is_successful_eval(path):
                continue
            providers: Counter[str] = Counter()
            cost = 0.0
            for i, sample in enumerate(read_eval_log_samples(str(path), all_samples_required=False)):
                if i >= args.samples:
                    break
                for event in sample.events or []:
                    call = getattr(event, "call", None)
                    if type(event).__name__ != "ModelEvent" or call is None:
                        continue
                    response = call.response or {}
                    # an error body names the provider that rejected under metadata
                    # (a 429 from the pinned provider is a retry, not a fallback)
                    served = response.get("provider") or (response.get("metadata") or {}).get("provider_name")
                    label = str(served) + ("" if "provider" in response else " [error]")
                    providers[label] += 1
                    cost += float((response.get("usage") or {}).get("cost") or 0.0)
            model_dir = path.parent.parent.name
            named, binding = pins.get(model_dir.replace("-rescored", ""), ([], False))
            served = ", ".join(f"{p} ({n})" for p, n in providers.most_common())
            # only a *successful* response from an unnamed provider is an off-pin run;
            # an error produced no generation, whoever rejected it
            off_pin = named and any(
                "[error]" not in p and _key(p) not in named for p in providers
            )
            flag = "  <== OFF PIN" if off_pin else ""
            pin_desc = ("binding " if binding else "preference ") + str(named or "none")
            print(f"{path.parts[1]:10s}{model_dir:44s}{path.parent.name:4s}{served:44s}{cost:8.2f}  {pin_desc}{flag}")


if __name__ == "__main__":
    main()
