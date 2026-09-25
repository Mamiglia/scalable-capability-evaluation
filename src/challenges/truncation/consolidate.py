"""Consolidate valid retries into derived copies, preserving all original collection logs."""
import argparse
import json
import shutil
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log
from challenges.truncation.prepare_publication import sha, content, rebuild, stamp_edit
from challenges.truncation.recollect import unusable
from challenges.truncation.run import root, tag


def main(domain):
    ROOT = root(domain)
    state = json.loads((ROOT / "results" / domain / "status.json").read_text())
    assert state["status"] != "running", "Primary collection must finish before consolidation"
    source = Path(state["log_dir"])
    destination = ROOT / "effective" / domain
    assert not destination.exists(), "Inspect the existing consolidated copy rather than overwrite it"
    replacements = {}
    for path in sorted((ROOT / "recollections" / domain).rglob("*.eval")):
        log = read_eval_log(path, resolve_attachments=True)
        if log.status == "success":
            for sample in log.samples:
                if not unusable(sample):
                    key = (log.eval.task, str(sample.id))
                    replacements.setdefault(key, (sample, path))
    destination.mkdir(parents=True)
    changed, used = [], set()
    for path in source.glob("*.eval"):
        if read_eval_log(path, header_only=True).status != "success":
            continue  # failed task attempt superseded by eval_set's retry; preserved in place, not consolidated
        original_hash = sha(path.read_bytes())
        log = read_eval_log(path, resolve_attachments=True)
        replaced = []
        for index, sample in enumerate(log.samples):
            key = (log.eval.task, str(sample.id))
            if key not in replacements:
                continue
            assert unusable(sample), "Do not replace a usable judgment"
            replacement, origin = replacements[key]
            assert replacement.input == sample.input and replacement.epoch == sample.epoch
            replacement = replacement.model_copy(deep=True)
            replacement.metadata = dict(replacement.metadata or {}) | {"recollection_source": origin.name,
                "superseded_archive": path.name, "superseded_archive_sha256": original_hash}
            log.samples[index] = replacement
            replaced.append({"item": str(sample.id), "retry_file": str(origin), "retry_sha256": sha(origin.read_bytes())})
            used.add(key)
        target = destination / path.name
        if replaced:
            log = rebuild(log)
            stamp_edit(log, tool="consolidate.py", reason="Replace unusable judgments with valid same-config recollections; subject inputs unchanged.",
                source=path.name, source_sha256=original_hash, samples_changed=len(replaced),
                replacement_judgments=replaced, subject_transcripts_modified=False)
            fingerprint = content(log.samples)
            write_eval_log(log, target)
            assert content(read_eval_log(target, resolve_attachments=True).samples) == fingerprint
            changed.append({"original": str(path), "original_sha256": original_hash,
                            "effective": str(target), "replacements": replaced})
        else:
            shutil.copy2(path, target)
        assert sha(path.read_bytes()) == original_hash
    assert used == set(replacements), "A recovery did not match an original sample"
    report = {"domain": domain, "log_dir": str(destination), "recovered_judgments": len(used), "changed_logs": changed}
    (ROOT / f"consolidation-{tag(domain)}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"domain": domain, "recovered_judgments": len(used), "log_dir": str(destination)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", help="<domain>/<judge>/<semester>/<head|tail|ends>-<max_length>")
    main(parser.parse_args().domain)
