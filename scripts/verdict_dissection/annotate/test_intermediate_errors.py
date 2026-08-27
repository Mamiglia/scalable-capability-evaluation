from intermediate_errors import parse


def test_parse_extracts_error_list_and_tolerates_prose():
    text = 'Sure.\n{"errors": [{"quote": "1/(4π^2)^3", "kind": "notation", "why": "should be ((4π)^2)^3"}]}'
    errors = parse(text)
    assert len(errors) == 1 and errors[0]["kind"] == "notation"
    assert parse('{"errors": []}') == []
    assert parse("no json here") is None
