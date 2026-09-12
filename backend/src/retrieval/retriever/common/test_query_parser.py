from src.retrieval.retriever.common.query_parser import (
    explicit_query_clauses,
    parse_explicit_query,
    rebuild_explicit_query,
)


def test_parse_mixed_explicit_query() -> None:
    events, connectors = parse_explicit_query("A AND B THEN C AND D")
    assert events == [["A", "B"], ["C", "D"]]
    assert connectors == ["AND", "THEN", "AND"]


def test_parser_does_not_split_punctuation_or_lowercase_words() -> None:
    query = "A, B; then C and D."
    assert parse_explicit_query(query)[0] == [[query]]


def test_batch_translation_rebuild_preserves_connectors() -> None:
    query = "một AND hai THEN ba"
    assert explicit_query_clauses(query) == ["một", "hai", "ba"]
    assert rebuild_explicit_query(query, ["one", "two", "three"]) == "one AND two THEN three"
