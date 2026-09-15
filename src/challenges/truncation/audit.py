"""Audit a truncation condition's saved judgments against frozen inputs using Inspect log APIs."""
import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values
from inspect_ai.log import read_eval_log, read_eval_log_sample, read_eval_log_sample_summaries
from judge.dataset import DIRECT_USER_PROMPT
from judge.scorer import parse_direct_judge_output
from challenges.truncation.run import condition_summary, tag

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "artifacts/direct-judging-truncation"  # frozen inputs, logs, results


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def baseline_request(log_dir):
    """Generation settings and served provider of the full-transcript run this condition mirrors."""
    for path in sorted(Path(log_dir).glob("*.eval")):
        if read_eval_log(path, header_only=True).status != "success":
            continue
        for sample in read_eval_log(path, resolve_attachments=True).samples:
            for event in sample.events:
                if event.event == "model" and event.call and not event.error and event.output.choices:
                    req, resp = event.call.request, event.call.response
                    return {"max_tokens": req.get("max_tokens"), "temperature": req.get("temperature"),
                            "reasoning": (req.get("extra_body") or {}).get("reasoning"),
                            "provider": resp.get("provider")}
    raise ValueError(f"No API trace in baseline logs: {log_dir}")


def audit(domain, log_dir=None):
    """`domain` is the condition: <domain>/<judge>/<semester>/<max_length>."""
    condition = domain
    payload = json.loads((ROOT / "inputs" / condition / "job.json").read_text())
    summary = condition_summary(condition)
    state = json.loads((ROOT / "results" / condition / "status.json").read_text())
    domain, semester, judge = payload["domain"], payload["semester"], payload["judge"]
    max_length, mode = payload["max_length"], payload["truncate"]
    suffix = DIRECT_USER_PROMPT.split("{transcript}", 1)[1]
    slug = payload["config"]["judge"]["slug"]
    request_model = slug.split("/", 1)[1]
    pin = payload["config"]["judge"]["model_args"]["provider"]
    baseline_pin = {k: v for k, v in pin.items() if k != "require_parameters"}
    baseline_dirs = [Path(p) for p in payload.get("baseline_log_dirs", [payload.get("baseline_log_dir")])]
    base = baseline_request(baseline_dirs[0])
    full_inputs = {}
    for path in (p for d in baseline_dirs for p in d.glob("*.eval")):
        header = read_eval_log(path, header_only=True)
        if header.status == "success":
            for sample in read_eval_log(path, resolve_attachments=True).samples:
                full_inputs[(header.eval.task, str(sample.id))] = sample.input
    checks, fixes, recollect, flags, files = [], [], [], [], []
    counts, providers, stops = Counter(), Counter(), Counter()
    traced, cached = {}, []
    seen = set()
    expected = {(t["name"], str(s["id"])): s for t in payload["tasks"] for s in t["samples"]}
    all_inputs = {(r["task"], r["item"]): r for r in payload["audit"]}
    prompt = (ROOT / "inputs" / condition / "prompt.txt").read_text()
    secrets = [v for k, v in dotenv_values(REPO / ".env").items()
               if v and len(v) >= 12 and any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))]
    secret_pattern = re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA|ENCRYPTED )? ?PRIVATE KEY-----|\b(?:hf_|sk-|rpa_)[A-Za-z0-9_-]{24,}")

    def check(ok, reason, where):
        if not ok:
            checks.append({"reason": reason, "where": where})

    check(sha((ROOT / "inputs" / condition / "job.json").read_bytes()) == summary["job_sha256"], "manifest changed", condition)
    check(sha(prompt) == summary["prompt_sha256"], "prompt changed", domain)
    for path in sorted(Path(log_dir or state["log_dir"]).glob("*.eval")):
        header = read_eval_log(path, header_only=True)
        if header.status != "success":
            counts["logs_" + header.status] += 1
            continue
        before = sha(path.read_bytes())
        log = read_eval_log(path, resolve_attachments=True)
        md, task = log.eval.metadata or {}, log.eval.task
        check(log.eval.model == slug, "judge model", str(path))
        check(log.eval.config.limit is None, "limited run", str(path))
        for key, value in {"protocol": "direct", "domain": domain, "semester": semester,
                           "include_reasoning": True, "max_length": max_length, "truncate": mode, "require_correct": True,
                           "input_manifest_sha256": summary["job_sha256"]}.items():
            check(md.get(key) == value, "metadata " + key, str(path))
        check(log.eval.model_args.get("provider") == payload["config"]["judge"]["model_args"]["provider"], "header provider", str(path))
        for sample in log.samples or []:
            key = (task, str(sample.id));where = {"task": task, "item": str(sample.id), "file": str(path)}
            check(key in expected, "unexpected sample", where)
            check(key not in seen, "duplicate sample", where)
            seen.add(key)
            if key not in expected:
                continue
            check(sample.epoch == 1 and sample.error is None, "sample error/epoch", where)
            check(isinstance(sample.input, str) and sha(sample.input) == all_inputs[key]["truncated_sha256"], "input mismatch", where)
            check(sample.input == expected[key]["input"], "frozen input mismatch", where)
            check((sample.metadata or {}).get("truncated") is all_inputs[key]["truncated"], "truncation flag", where)
            check((sample.metadata or {}).get("chars") == all_inputs[key]["truncated_chars"] <= max_length, "kept characters", where)
            kept, body, full_body = (sample.metadata or {}).get("chars") or 0, sample.input[:-len(suffix)], full_inputs[key][:-len(suffix)]
            text, head = body[len(body) - kept:], body[:len(body) - kept]
            check(sample.input.endswith(suffix) and full_body.startswith(head)
                  and (full_body.startswith(head + text) if mode == "head" else full_body.endswith(text)), "not a cut of the full transcript", where)
            serialized = sample.model_dump_json()
            check(not any(secret in serialized for secret in secrets) and not secret_pattern.search(serialized), "credential content", where)
            successful = [e for e in sample.events if e.event == "model" and not e.error and e.output.choices]
            counts["retried_api_errors"] += sum(e.event == "model" and bool(e.error) for e in sample.events)
            check(bool(successful), "missing successful API trace", where)
            for event in successful:
                output_hash = sha(json.dumps(event.output.model_dump(mode="json", exclude={"time"}), sort_keys=True))
                if event.call is None:
                    check(event.cache == "read", "missing API call", where)
                    if event.cache == "read":
                        cached.append(where | {"input_sha256": sha(sample.input), "output_sha256": output_hash})
                        counts["cache_hits"] += 1
                    continue
                req, resp = event.call.request, event.call.response
                check(req.get("model") == request_model, "request model", where)
                check(req.get("temperature") == 0 and req.get("max_tokens") == base["max_tokens"], "sampling parameters", where)
                check((req.get("extra_body") or {}).get("provider") == pin, "request pin", where)
                check((req.get("extra_body") or {}).get("reasoning") == base["reasoning"], "judge reasoning setting", where)
                messages = req.get("messages", [])
                check([m["content"] for m in messages if m["role"] == "system"] == [prompt], "system prompt", where)
                check([m["content"] for m in messages if m["role"] == "user"] == [sample.input], "API input", where)
                check(resp.get("provider") == base["provider"], "off-pin/missing provider", where)
                check(bool(resp.get("id")), "missing generation id", where)
                providers[str(resp.get("provider"))] += 1
                traced.setdefault((sha(sample.input), output_hash), where)
                counts["recorded_cost_usd"] += float((resp.get("usage") or {}).get("cost") or 0)
            ratings = sample.scores or {}
            check(set(ratings) == {"direct_judge_scorer"}, "scorer", where)
            score = ratings.get("direct_judge_scorer")
            stored = float(score.answer) if score and score.answer is not None else None
            answer, _ = parse_direct_judge_output(sample.output.completion)
            parsed = float(answer) if answer is not None else None
            check(score is not None and (score.value == stored if stored is not None else math.isnan(score.value)), "score consistency", where)
            check(stored is None or stored == parsed, "changed existing rating", where)
            counts["stored_ratings"] += stored is not None
            counts["parsed_ratings"] += parsed is not None
            stop = sample.output.stop_reason
            stops[str(stop)] += 1
            if parsed != stored:
                fixes.append(where | {"old": stored, "new": parsed, "completion": sample.output.completion[-500:]})
            if parsed is None or stop == "max_tokens":
                recollect.append(where | {"reason": "output cap" if stop == "max_tokens" else "missing final rating",
                                          "completion": sample.output.completion[-500:]})
            source_text = sample.input.split("Transcript 1:\n---\n\n", 1)[-1].rsplit("\n\n---\n\nHow capable", 1)[0]
            if re.search(r"\b(?:I am|I'm|developed by|created by)\b[^.\n]{0,80}\b(?:OpenAI|Anthropic|DeepSeek|Qwen|GLM|Gemini|Llama|Claude)\b", source_text, re.I):
                flags.append(where | {"reason": "possible subject self-identification"})
            if any(marker in source_text for marker in ("[Image]", "[Audio]", "[Video]", "[Document]")):
                flags.append(where | {"reason": "nontext placeholder in subject response"})
            lines = [line for line in sample.output.completion.splitlines() if line.strip()]
            if len(lines) > 50 and len(set(lines)) / len(lines) < .3:
                flags.append(where | {"reason": "possible degenerate judgment"})
            counts["samples"] += 1
        check(sha(path.read_bytes()) == before, "log changed during audit", str(path))
        files.append({"path": str(path), "sha256": before, "task": task, "samples": len(log.samples or [])})
    base_files, reused_provenance, reused_cache = {}, [], []
    archives = {}

    def archive_index():
        """(input sha, output sha) -> archive file, over every log of the baseline campaigns' log and
        recovery trees, for API calls that a later resumed baseline log only holds as cache reads."""
        if archives:
            return archives
        roots = {next(p for p in d.parents if p.name == "logs").parent for d in baseline_dirs}
        for path in sorted(p for root in roots for sub in ("logs", "recovery") for p in (root / sub).rglob("*.eval")):
            log = read_eval_log(path, resolve_attachments=True)
            for archived in log.samples or []:
                for e in archived.events:
                    if e.event == "model" and e.call is not None and not e.error and e.output.choices:
                        archives.setdefault((sha(archived.input), sha(json.dumps(e.output.model_dump(mode="json", exclude={"time"}), sort_keys=True))), path)
        archives.setdefault(None, None)
        return archives
    for path in (p for d in baseline_dirs for p in d.glob("*.eval")):
        header = read_eval_log(path, header_only=True)
        if header.status != "success":
            continue
        for summary_row in read_eval_log_sample_summaries(str(path)):
            # A targeted retry log sits beside the main task log: the latest created log wins, as in from_logs.
            key = (header.eval.task, str(summary_row.id))
            if key not in base_files or base_files[key][0] < header.eval.created:
                base_files[key] = (header.eval.created, path)
    for row in payload["reused_ratings"]:
        key = (f"{row['bench']}/{row['model']}", row["item"])
        path = base_files[key][1]
        header = read_eval_log(path, header_only=True)
        check(header.eval.model == slug, "reused judge model", key)
        sample = read_eval_log_sample(path, row["item"], 1, resolve_attachments=True)
        check(sha(sample.input) == all_inputs[key]["truncated_sha256"] == all_inputs[key]["full_sha256"], "reused input mismatch", key)
        answer, _ = parse_direct_judge_output(sample.output.completion)
        check(answer is not None and float(answer) == row["score"], "reused rating mismatch", key)
        check(sample.output.stop_reason != "max_tokens" and sample.error is None, "reused incomplete output", key)
        events = [e for e in sample.events if e.event == "model" and not e.error and e.output.choices]
        check(bool(events), "reused API trace missing", key)
        for event in events:
            if event.call is None:
                # The baseline resumed from its generation cache after a context-screen correction;
                # the original call lives in that campaign's archived earlier attempt.
                output_hash = sha(json.dumps(event.output.model_dump(mode="json", exclude={"time"}), sort_keys=True))
                origin = archive_index().get((sha(sample.input), output_hash))
                check(event.cache == "read" and origin is not None, "reused API provenance missing", key)
                if origin is not None:
                    reused_cache.append({"task": key[0], "item": key[1], "origin": str(origin)})
                    counts["reused_cache_hits"] += 1
                continue
            req, resp = event.call.request, event.call.response
            check(req.get("model") == request_model and req.get("temperature") == 0
                  and req.get("max_tokens") == base["max_tokens"]
                  and (req.get("extra_body") or {}).get("reasoning") == base["reasoning"], "reused model/settings", key)
            check((req.get("extra_body") or {}).get("provider") == baseline_pin, "reused provider pin", key)
            check(resp.get("provider") == base["provider"] and bool(resp.get("id")), "reused served provider", key)
            messages = req.get("messages", [])
            check([m["content"] for m in messages if m["role"] == "system"] == [prompt], "reused prompt", key)
            check([m["content"] for m in messages if m["role"] == "user"] == [sample.input], "reused API input", key)
        reused_provenance.append({"task": key[0], "item": key[1], "file": str(path), "sha256": sha(path.read_bytes())})
    for row in cached:
        origin = traced.get((row["input_sha256"], row["output_sha256"]))
        check(origin is not None, "unverifiable cache origin", row)
        row["origin"] = origin
    declaration = ROOT / f"declared-failures-{tag(condition)}.json"
    declared = json.loads(declaration.read_text()) if declaration.exists() else []
    declared_keys = {(row["task"], row["item"]) for row in declared}
    check(declared_keys <= {(row["task"], row["item"]) for row in recollect}, "declared failure is not a failed rating", domain)
    for row in declared:
        check(len(row["retry_logs"]) >= 2, "insufficient recorded retries", row)
        for path in row["retry_logs"]:
            header = read_eval_log(path, header_only=True)
            sample = read_eval_log_sample(path, row["item"], 1, resolve_attachments=True)
            check(header.eval.task == row["task"] and header.eval.model == slug, "failed retry identity", row)
            check(sample.output.stop_reason == "max_tokens" and not sample.output.completion, "failed retry outcome", row)
            check(sha(sample.input) == all_inputs[(row["task"], row["item"])]["truncated_sha256"], "failed retry input", row)
    recollect = [row for row in recollect if (row["task"], row["item"]) not in declared_keys]
    report = {"condition": condition, "domain": domain, "judge": judge, "semester": semester, "truncate": mode, "max_length": max_length,
              "baseline_request": base, "audited_at": datetime.now(timezone.utc).isoformat(),
              "job_sha256": summary["job_sha256"], "prompt_sha256": summary["prompt_sha256"], "counts": dict(counts),
              "providers": dict(providers), "stop_reasons": dict(stops), "files": files,
              "missing_new_keys": sorted(set(expected) - seen), "reused_verified": len(payload["reused_ratings"]),
              "blocking": checks, "score_repairs": fixes, "recollect": recollect, "review_flags": flags,
              "cache_provenance": cached, "reused_provenance": reused_provenance,
              "reused_cache_provenance": reused_cache, "declared_failures": declared}
    (ROOT / "audit").mkdir(exist_ok=True)
    (ROOT / "audit" / f"{tag(condition)}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"condition": condition, "counts": dict(counts), "providers": dict(providers),
                      "missing": len(report["missing_new_keys"]), "blocking": len(checks),
                      "score_repairs": len(fixes), "recollect": len(recollect), "review_flags": flags}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domains", nargs="+", help="conditions: <domain>/<judge>/<semester>/<head|tail>-<max_length>")
    parser.add_argument("--partial", action="store_true", help="Allow unfinished workers; never a publication gate")
    parser.add_argument("--log-dir", type=Path, help="Audit consolidated recovery copies for one condition")
    args = parser.parse_args()
    if args.log_dir and len(args.domains) != 1:
        parser.error("--log-dir requires exactly one domain")
    reports = [audit(domain, args.log_dir) for domain in args.domains]
    raise SystemExit(any(report["blocking"] or (not args.partial and any(report[key] for key in
                         ("missing_new_keys", "recollect", "score_repairs", "review_flags"))) for report in reports))
