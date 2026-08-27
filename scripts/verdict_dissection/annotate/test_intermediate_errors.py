from intermediate_errors import parse


def test_parse_extracts_error_list_and_tolerates_prose():
    text = 'Sure.\n{"errors": [{"quote": "1/(4π^2)^3", "kind": "notation", "why": "should be ((4π)^2)^3"}]}'
    errors = parse(text)
    assert len(errors) == 1 and errors[0]["kind"] == "notation"
    assert parse('{"errors": []}') == []
    assert parse("no json here") is None


def test_codex_parse_repairs_latex_escapes():
    from intermediate_errors_codex import parse
    reply = '{"annotations":[{"id":1,"errors":[{"quote":"\\(15^2 = 225\\)","kind":"arithmetic","why":"\\frac{1}{2}"}]},{"id":2,"errors":[]}]}'
    out = parse(reply, 2)
    assert out[1][0]["quote"] == "\\(15^2 = 225\\)" and out[2] == []


def test_codex_parse_treats_omitted_numbers_as_clean():
    from intermediate_errors_codex import parse
    out = parse('{"annotations":[{"id":2,"errors":[{"quote":"x","kind":"other","why":"y"}]}]}', 3)
    assert out == {1: [], 2: [{"quote": "x", "kind": "other", "why": "y"}], 3: []}
