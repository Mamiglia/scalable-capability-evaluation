import importlib.util
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).parent
PREPARE_PATH = HERE / "prepare_style_standardised.py"
sys.path.insert(0, str(HERE))


def load_prepare():
    if not PREPARE_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("prepare_style_standardised", PREPARE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def row(model: str, item: str, rewritten: str = "text") -> dict:
    return {"model": model, "item_id": item, "rewritten": rewritten}


def test_merge_requires_exact_manifest_keys_and_writes_manifest_order(tmp_path):
    prepare = load_prepare()
    assert prepare is not None, "preparation script is missing"
    manifest = tmp_path / "manifest.jsonl"
    parts = [tmp_path / f"part-{index}.jsonl" for index in range(3)]
    merged = tmp_path / "merged.jsonl"
    write_jsonl(manifest, [row("m", "i2"), row("m", "i1"), row("n", "i3")])
    write_jsonl(parts[0], [row("n", "i3", "three")])
    write_jsonl(parts[1], [row("m", "i1", "one")])
    write_jsonl(parts[2], [row("m", "i2", "two")])

    rows = prepare.merge_parts(manifest, parts, merged, expected_rows=3)

    assert [(r["model"], r["item_id"], r["rewritten"]) for r in rows] == [
        ("m", "i2", "two"), ("m", "i1", "one"), ("n", "i3", "three")
    ]
    assert [json.loads(line) for line in merged.read_text().splitlines()] == rows


@pytest.mark.parametrize(
    "part_rows, message",
    [
        ([row("m", "i1"), row("m", "i1"), row("m", "i2")], "duplicate"),
        ([row("m", "i1")], "missing"),
        ([row("m", "i1"), row("m", "i2"), row("m", "extra")], "extra"),
    ],
)
def test_merge_rejects_duplicates_missing_and_extras(tmp_path, part_rows, message):
    prepare = load_prepare()
    assert prepare is not None, "preparation script is missing"
    manifest = tmp_path / "manifest.jsonl"
    parts = [tmp_path / f"part-{index}.jsonl" for index in range(3)]
    write_jsonl(manifest, [row("m", "i1"), row("m", "i2")])
    write_jsonl(parts[0], part_rows)
    write_jsonl(parts[1], [])
    write_jsonl(parts[2], [])

    with pytest.raises(ValueError, match=message):
        prepare.merge_parts(manifest, parts, tmp_path / "merged.jsonl", expected_rows=2)


def test_qc_excludes_only_structural_failures_and_reports_residual_target_hits():
    prepare = load_prepare()
    assert prepare is not None, "preparation script is missing"
    originals = {
        ("m", "ok"): {
            "transcript": "Me is the methyl group; X is likely under A.\nAnswer: A\nAfterword"
        },
        ("m", "bad"): {"transcript": "First answer.\nANSWER: B\nThen correction.\nAnswer: C"},
    }
    rows = [
        row("m", "ok", "Me is retained; X is likely under A.\nAnswer: A\nAfterword"),
        row("m", "bad", "First answer.\nAnswer: C\nThen correction.\nANSWER: B"),
    ]
    lexicon = {
        "personal_pronouns": ["I", "me", "we", "our", "us"],
        "hedging": ["likely"],
        "backtracking": ["wait"],
        "uncertainty": ["unclear"],
    }

    report = prepare.qc_rows(rows, originals, lexicon)

    assert report["failures"] == [{
        "model": "m", "item_id": "bad", "reason": "answer_lines_changed"
    }]
    assert report["residual_targeted"]["personal_pronouns"]["count"] == 1
    assert report["residual_targeted"]["personal_pronouns"]["contexts"][0]["match"] == "Me"
    assert report["residual_targeted"]["hedging"]["count"] == 1
    assert report["features"][0]["rewritten"]["hedging"] > 0
    assert report["features"][0]["rewritten"]["uncertainty"] == 0


def test_qc_compares_answer_lines_before_role_prefix_normalisation():
    prepare = load_prepare()
    assert prepare is not None, "preparation script is missing"
    originals = {("m", "i"): {"transcript": "[Assistant]: Answer: A"}}
    lexicon = {category: [] for category in
               ("personal_pronouns", "hedging", "backtracking")}

    report = prepare.qc_rows([row("m", "i", "Answer: A")], originals, lexicon)

    assert report["failures"] == [{
        "model": "m", "item_id": "i", "reason": "answer_lines_changed"
    }]


def test_structural_qc_preserves_nonblank_content_side_of_final_answer_line():
    prepare = load_prepare()
    assert prepare is not None, "preparation script is missing"
    at_end = "Reasoning.\nAnswer: A"
    with_trailing = "Reasoning.\nAnswer: A\n\nPostscript."

    assert prepare.structural_failure(with_trailing, "Reasoning changed.\nPostscript.\nAnswer: A") \
        == "final_answer_trailing_content_changed"
    assert prepare.structural_failure(at_end, "Reasoning changed.\nAnswer: A\nPostscript.") \
        == "final_answer_trailing_content_changed"


def test_answer_parser_accepts_answer_label_case_insensitively():
    from build_rewrite_evals import rewrite_answer_letter

    assert rewrite_answer_letter("reasoning\nAnswer: b") == "B"
    assert rewrite_answer_letter("reasoning\nFinal Answer: c") == "C"
