import json

from features_derivation import derivation_of, symbol_chain, equation_lines


def parts(text):
    return json.dumps([{"role": "assistant", "subparts": [{"kind": "content", "content": text}]}])


def test_derivation_counts_relations_chains_and_answer_derived():
    text = ("Each loop gives 1/(4\\pi)^2, so for L loops the factor is 1/(4\\pi)^{2L}.\n"
            "The note shows 1/(4\\pi)^6, hence 2L = 6, so L = 3.\n"
            "By Stewart's theorem, AK^2 = 40 - 324/25 = 676/25, so AK = 26/5.\n\nANSWER: 3")
    f = derivation_of(parts(text), "How many loops? The factor is 1/(4pi)^6.", None)
    assert f["stated_relations"] == 5
    assert f["chained_equalities"] == 1                 # AK^2 = ... = 676/25
    assert f["named_methods"] == 1
    assert f["consequence_markers"] >= 3               # so, hence, so, so
    assert f["answer_derived"] == 1                    # "L = 3"
    assert "3" not in {"6"} and f["intermediate_numbers"] >= 3   # 40, 324, 25, 676, 26, 5 ...


def test_answer_derived_uses_option_text_for_letter_answers():
    text = "E = \\gamma m c^2 = 20.132 GeV.\n\nANSWER: C"
    options = json.dumps(["18.62 GeV", "23.069 GeV", "20.132 GeV", "21.419 GeV"])
    assert derivation_of(parts(text), "What is the energy?", options)["answer_derived"] == 1
    assert derivation_of(parts("It is C.\n\nANSWER: C"), "What is the energy?", options)["answer_derived"] == 0


def test_symbol_chain_links_later_equations_to_defined_symbols():
    units = equation_lines("Let $a = 2b$. Then $a^2 + b^2 = 36$ so $5b^2 = 36$.")
    n_symbols, links = symbol_chain(units)
    assert n_symbols >= 1 and links == 2
