"""Export per-response cost and served-provider provenance from frontier logs."""

import argparse
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log

EXPECTED = {
    "deepseek-v4-flash-0731": "DeepInfra",
    "gemini-2.5-pro": "Google AI Studio",
    "gpt-4o-2024-11-20": "OpenAI",
    "gpt-5.2": "OpenAI",
    "o4-mini": "OpenAI",
}


def collect(roots):
    rows = []
    for root in roots:
        for path in sorted(root.rglob("*.eval")):
            log = read_eval_log(str(path))
            if log.status != "success":
                continue
            judge = log.eval.model.split("/")[-1]
            meta = log.eval.metadata or {}
            for sample in log.samples or []:
                for event in sample.events:
                    response = event.call.response if event.event == "model" and event.call else None
                    if not response:
                        continue
                    usage = response.get("usage") or {}
                    details = usage.get("completion_tokens_details") or {}
                    rows.append({
                        "source": root.name, "log": str(path), "judge": judge,
                        "sample_id": str(sample.id), "response_id": response.get("id"),
                        "served_model": response.get("model"), "provider": response.get("provider"),
                        "cost": float(usage.get("cost") or 0),
                        "input_tokens": int(usage.get("prompt_tokens") or 0),
                        "output_tokens": int(usage.get("completion_tokens") or 0),
                        "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
                        "major_path": meta.get("major_path"), "minor_path": meta.get("minor_path"),
                    })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    calls = collect(args.roots)
    bad = calls[calls.provider != calls.judge.map(EXPECTED)]
    if not bad.empty:
        raise ValueError(f"{len(bad)} responses came from an unexpected provider")
    args.out.mkdir(parents=True, exist_ok=True)
    calls.to_parquet(args.out / "api_calls.parquet", index=False)
    summary = (calls.groupby(["source", "judge", "provider"])
               .agg(calls=("response_id", "size"), cost=("cost", "sum"),
                    input_tokens=("input_tokens", "sum"), output_tokens=("output_tokens", "sum"),
                    reasoning_tokens=("reasoning_tokens", "sum")).reset_index())
    summary.to_csv(args.out / "api_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"logged cost: ${calls.cost.sum():.4f}")


if __name__ == "__main__":
    main()
