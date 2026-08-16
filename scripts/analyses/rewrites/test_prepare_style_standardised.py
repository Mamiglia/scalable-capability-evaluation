import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml


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


def test_answer_labels_exclude_explanatory_prose():
    prepare = load_prepare()
    assert prepare is not None, "preparation script is missing"

    assert prepare._answer_lines(
        "Here's why B is the correct answer:\nANSWER: B"
    ) == ["ANSWER: B"]
    assert prepare._answer_lines(
        "[Assistant]: The following steps determine the answer:\nANSWER: D"
    ) == ["ANSWER: D"]
    assert prepare._answer_lines("### **Correct Answer:**\nB\nAnswer: B") == [
        "### **Correct Answer:**", "Answer: B"
    ]


def test_answer_parser_accepts_answer_label_case_insensitively():
    from build_rewrite_evals import rewrite_answer_letter

    assert rewrite_answer_letter("reasoning\nAnswer: b") == "B"
    assert rewrite_answer_letter("reasoning\nFinal Answer: c") == "C"


def test_main_returns_nonzero_without_building_when_structural_qc_fails(
        tmp_path, monkeypatch):
    prepare = load_prepare()
    originals = tmp_path / "originals.jsonl"
    write_jsonl(originals, [{
        "model": "m", "item_id": "i", "transcript": "Reasoning.\nAnswer: A"
    }])
    rows = [row("m", "i", "Reasoning.\nAnswer: B")]
    built = []

    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "MANIFEST", tmp_path / "manifest.jsonl")
    monkeypatch.setattr(prepare, "ORIGINALS", originals)
    monkeypatch.setattr(prepare, "MERGED", tmp_path / "merged.jsonl")
    monkeypatch.setattr(prepare, "QC_OUT", tmp_path / "qc.json")
    monkeypatch.setattr(prepare, "EXCLUDED_OUT", tmp_path / "excluded.json")
    monkeypatch.setattr(prepare, "FEATURES_OUT", tmp_path / "features.jsonl")
    monkeypatch.setattr(prepare, "EVAL_DIR", tmp_path / "production-evals")
    monkeypatch.setattr(prepare, "merge_parts", lambda *_args: rows)
    monkeypatch.setattr(prepare, "load_lexicon", lambda *_args: {
        category: [] for category in prepare.TARGETED_CATEGORIES
    })
    monkeypatch.setattr(prepare, "_build_evals", lambda *_args: built.append(True) or [])

    assert prepare.main() == 1
    assert built == []
    assert not prepare.EVAL_DIR.exists()
    assert json.loads(prepare.QC_OUT.read_text())["n_structural_failures"] == 1
    assert json.loads(prepare.EXCLUDED_OUT.read_text())["n_excluded"] == 1


def test_main_discards_staged_evals_when_real_scorer_fails(tmp_path, monkeypatch):
    prepare = load_prepare()
    originals = tmp_path / "originals.jsonl"
    write_jsonl(originals, [{
        "model": "m", "item_id": "i", "transcript": "Reasoning.\nAnswer: A"
    }])
    rows = [row("m", "i", "Reworded.\nAnswer: A")]

    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "MANIFEST", tmp_path / "manifest.jsonl")
    monkeypatch.setattr(prepare, "ORIGINALS", originals)
    monkeypatch.setattr(prepare, "MERGED", tmp_path / "merged.jsonl")
    monkeypatch.setattr(prepare, "QC_OUT", tmp_path / "qc.json")
    monkeypatch.setattr(prepare, "EXCLUDED_OUT", tmp_path / "excluded.json")
    monkeypatch.setattr(prepare, "FEATURES_OUT", tmp_path / "features.jsonl")
    monkeypatch.setattr(prepare, "EVAL_DIR", tmp_path / "production-evals")
    monkeypatch.setattr(prepare, "merge_parts", lambda *_args: rows)
    monkeypatch.setattr(prepare, "load_lexicon", lambda *_args: {
        category: [] for category in prepare.TARGETED_CATEGORIES
    })

    def scorer_failure(_rows, out_dir):
        target = out_dir if isinstance(out_dir, Path) else prepare.EVAL_DIR
        target.mkdir(parents=True, exist_ok=True)
        (target / "m.eval").write_text("reduced scorer output")
        return [{"model": "m", "item_id": "i", "reason": "real scorer failure"}]

    monkeypatch.setattr(prepare, "_build_evals", scorer_failure)

    assert prepare.main() == 1
    assert not prepare.EVAL_DIR.exists()
    assert json.loads(prepare.QC_OUT.read_text())["n_scorer_failures"] == 1
    assert json.loads(prepare.EXCLUDED_OUT.read_text())["n_excluded"] == 1


def test_main_discards_staged_evals_when_campaign_coverage_fails(tmp_path, monkeypatch):
    prepare = load_prepare()
    originals = tmp_path / "originals.jsonl"
    write_jsonl(originals, [{
        "model": "m", "item_id": "i", "transcript": "Reasoning.\nAnswer: A"
    }])
    rows = [row("m", "i", "Reworded.\nAnswer: A")]

    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "MANIFEST", tmp_path / "manifest.jsonl")
    monkeypatch.setattr(prepare, "ORIGINALS", originals)
    monkeypatch.setattr(prepare, "MERGED", tmp_path / "merged.jsonl")
    monkeypatch.setattr(prepare, "QC_OUT", tmp_path / "qc.json")
    monkeypatch.setattr(prepare, "EXCLUDED_OUT", tmp_path / "excluded.json")
    monkeypatch.setattr(prepare, "FEATURES_OUT", tmp_path / "features.jsonl")
    monkeypatch.setattr(prepare, "EVAL_DIR", tmp_path / "production-evals")
    monkeypatch.setattr(prepare, "merge_parts", lambda *_args: rows)
    monkeypatch.setattr(prepare, "load_lexicon", lambda *_args: {
        category: [] for category in prepare.TARGETED_CATEGORIES
    })

    def clean_scorer(_rows, out_dir):
        (out_dir / "m.eval").write_text("complete scorer output")
        return []

    monkeypatch.setattr(prepare, "_build_evals", clean_scorer)
    monkeypatch.setattr(
        prepare, "_validate_campaign",
        lambda *_args: (_ for _ in ()).throw(ValueError("missing campaign sides")),
        raising=False,
    )

    assert prepare.main() == 1
    assert not prepare.EVAL_DIR.exists()
    assert json.loads(prepare.QC_OUT.read_text())["coverage_failure"] \
        == "missing campaign sides"
    assert json.loads(prepare.EXCLUDED_OUT.read_text())["n_excluded"] == 0


def test_campaign_validation_rejects_one_missing_manifest_key(tmp_path, monkeypatch):
    prepare = load_prepare()
    config_path = (HERE.parents[2] / "scripts" / "pairwise-newjudges" / "configs"
                   / "manipulations" / "style_standardised_gpt.yaml")
    cfg = yaml.safe_load(config_path.read_text())
    manifest_path = HERE.parents[2] / "judge_results" / "rewrites" / "paraphrase_gpt_v2.jsonl"
    manifest_keys = {
        (cell["model"], cell["item_id"])
        for cell in map(json.loads, manifest_path.read_text().splitlines())
    }
    for model in cfg["transcripts"]:
        (tmp_path / f"{model}.eval").touch()
    missing = manifest_keys - {next(iter(manifest_keys))}

    monkeypatch.setattr(prepare, "CAMPAIGN_CONFIG", config_path, raising=False)
    monkeypatch.setattr(prepare, "_eval_store_keys", lambda *_args: missing, raising=False)
    validate = getattr(prepare, "_validate_campaign", lambda *_args: None)

    with pytest.raises(ValueError, match="manifest"):
        validate(tmp_path, manifest_keys)


def test_eval_store_keys_reads_model_and_item_ids_from_the_staged_stores(
        tmp_path, monkeypatch):
    import inspect_ai.log

    prepare = load_prepare()
    (tmp_path / "model-a.eval").touch()
    (tmp_path / "model-b.eval").touch()
    samples = {
        "model-a.eval": [SimpleNamespace(id="item-1"), SimpleNamespace(id="item-2")],
        "model-b.eval": [SimpleNamespace(id="item-1")],
    }
    monkeypatch.setattr(
        inspect_ai.log, "read_eval_log",
        lambda path: SimpleNamespace(samples=samples[Path(path).name]),
    )
    read_keys = getattr(prepare, "_eval_store_keys", lambda *_args: set())

    assert read_keys(tmp_path) == {
        ("model-a", "item-1"), ("model-a", "item-2"), ("model-b", "item-1")
    }


def test_main_promotes_exact_complete_campaign_after_validation(tmp_path, monkeypatch):
    prepare = load_prepare()
    config_path = (HERE.parents[2] / "scripts" / "pairwise-newjudges" / "configs"
                   / "manipulations" / "style_standardised_gpt.yaml")
    cfg = yaml.safe_load(config_path.read_text())
    manifest_path = HERE.parents[2] / "judge_results" / "rewrites" / "paraphrase_gpt_v2.jsonl"
    rows = list(map(json.loads, manifest_path.read_text().splitlines()))
    manifest_keys = {(cell["model"], cell["item_id"]) for cell in rows}

    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "MANIFEST", tmp_path / "manifest.jsonl")
    monkeypatch.setattr(
        prepare, "ORIGINALS",
        HERE.parents[2] / "judge_results" / "rewrites" / "originals.jsonl",
    )
    monkeypatch.setattr(prepare, "MERGED", tmp_path / "merged.jsonl")
    monkeypatch.setattr(prepare, "QC_OUT", tmp_path / "qc.json")
    monkeypatch.setattr(prepare, "EXCLUDED_OUT", tmp_path / "excluded.json")
    monkeypatch.setattr(prepare, "FEATURES_OUT", tmp_path / "features.jsonl")
    monkeypatch.setattr(prepare, "EVAL_DIR", tmp_path / "production-evals")
    monkeypatch.setattr(prepare, "CAMPAIGN_CONFIG", config_path)
    monkeypatch.setattr(prepare, "merge_parts", lambda *_args: rows)
    monkeypatch.setattr(prepare, "load_lexicon", lambda *_args: {})
    monkeypatch.setattr(prepare, "qc_rows", lambda *_args: {
        "failures": [], "features": [],
        "residual_targeted": {category: {"count": 0, "contexts": []}
                              for category in prepare.TARGETED_CATEGORIES},
    })

    def build_complete_stores(_rows, out_dir):
        for model in cfg["transcripts"]:
            (out_dir / f"{model}.eval").touch()
        return []

    monkeypatch.setattr(prepare, "_build_evals", build_complete_stores)
    monkeypatch.setattr(prepare, "_eval_store_keys", lambda *_args: manifest_keys)

    assert prepare.main() == 0
    assert {path.stem for path in prepare.EVAL_DIR.glob("*.eval")} \
        == set(cfg["transcripts"])
    assert json.loads(prepare.QC_OUT.read_text())["coverage_failure"] is None
    assert json.loads(prepare.EXCLUDED_OUT.read_text())["n_excluded"] == 0
