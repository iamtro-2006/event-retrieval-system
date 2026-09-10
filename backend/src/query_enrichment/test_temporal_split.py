from __future__ import annotations

from src.query_enrichment.llm_query_engine import (
    LLMQueryEngine,
    _parse_temporal_events,
)
from src.retrieval.retriever.common.orchestrator import split_temporal_events


def test_temporal_parser_sorts_explicit_steps() -> None:
    raw = (
        '{"timeline":"multiple","event_count":3,"events":['
        '{"step":3,"event":"The man sits down"},'
        '{"step":1,"event":"The man opens the door"},'
        '{"step":2,"event":"The man enters the room"}]}'
    )
    assert _parse_temporal_events(raw) == [
        "The man opens the door",
        "The man enters the room",
        "The man sits down",
    ]


def test_temporal_parser_rejects_ambiguous_step_numbers() -> None:
    assert _parse_temporal_events(
        '{"timeline":"multiple","event_count":2,"events":'
        '[{"step":1,"event":"First"},{"step":1,"event":"Second"}]}'
    ) is None
    assert _parse_temporal_events(
        '{"timeline":"multiple","event_count":2,"events":'
        '[{"step":1,"event":"First"},{"step":3,"event":"Third"}]}'
    ) is None


def test_temporal_parser_rejects_old_or_inconsistent_contracts() -> None:
    assert _parse_temporal_events('["First", "Second"]') is None
    assert _parse_temporal_events(
        '{"timeline":"single","event_count":2,"events":'
        '[{"step":1,"event":"First"},{"step":2,"event":"Second"}]}'
    ) is None
    assert _parse_temporal_events(
        '{"timeline":"multiple","event_count":1,"events":'
        '[{"step":1,"event":"Only"}]}'
    ) is None


def test_temporal_parser_rejects_one_event_for_explicit_sequence() -> None:
    raw = (
        '{"timeline":"single","event_count":1,"events":'
        '[{"step":1,"event":"The man opens the door and sits down"}]}'
    )
    assert _parse_temporal_events(raw, "The man opens the door, then sits down") is None


def test_temporal_parser_accepts_a_genuine_single_event() -> None:
    raw = (
        '{"timeline":"single","event_count":1,"events":'
        '[{"step":1,"event":"A man runs through a park."}]}'
    )
    assert _parse_temporal_events(raw, "A man runs through a park.") == [
        "A man runs through a park."
    ]


def test_temporal_fallback_splits_explicit_transitions_without_punctuation() -> None:
    assert split_temporal_events(
        "The man opens the door then enters the room and then sits down"
    ) == [
        "The man opens the door",
        "enters the room",
        "sits down",
    ]


def test_temporal_split_uses_versioned_cache_contract() -> None:
    engine = object.__new__(LLMQueryEngine)
    engine._model = "test-model"
    calls = []

    def fake_chat(system_prompt, user_text, cache_key):
        calls.append((system_prompt, user_text, cache_key))
        return (
            '{"timeline":"multiple","event_count":2,"events":'
            '[{"step":2,"event":"B"},{"step":1,"event":"A"}]}'
        )

    engine._chat = fake_chat
    assert engine.split_temporal_events("A then B") == ["A", "B"]
    assert calls[0][2][:2] == ("temporal", 3)
    assert "Never reorder events using commonsense assumptions" in calls[0][0]


def test_temporal_split_repairs_suspicious_single_event_answer() -> None:
    engine = object.__new__(LLMQueryEngine)
    engine._model = "test-model"
    responses = iter(
        [
            '{"timeline":"single","event_count":1,"events":'
            '[{"step":1,"event":"The man enters and sits"}]}',
            '{"timeline":"multiple","event_count":2,"events":'
            '[{"step":1,"event":"The man enters"},'
            '{"step":2,"event":"The man sits"}]}',
        ]
    )
    calls = []

    def fake_chat(system_prompt, user_text, cache_key):
        calls.append(cache_key)
        return next(responses)

    engine._chat = fake_chat
    assert engine.split_temporal_events("The man enters and sits") == [
        "The man enters",
        "The man sits",
    ]
    assert calls[0][0] == "temporal"
    assert calls[1][0] == "temporal-repair"
