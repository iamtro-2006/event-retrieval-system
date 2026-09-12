from __future__ import annotations

import re


_EXPLICIT_CONNECTOR_RE = re.compile(r"\b(AND|THEN)\b")


def parse_explicit_query(query: str) -> tuple[list[list[str]], list[str]]:
    """Parse case-sensitive AND/THEN query syntax.

    AND adds a semantic alternative to the current event. THEN starts the
    next temporal event. Punctuation and lower-case words are never split.
    """
    text = str(query or "").strip()
    if not text:
        return [], []
    pieces = _EXPLICIT_CONNECTOR_RE.split(text)
    events: list[list[str]] = []
    connectors: list[str] = []
    current: list[str] = []
    if pieces[0].strip():
        current.append(pieces[0].strip())
    for connector, raw_clause in zip(pieces[1::2], pieces[2::2]):
        clause = raw_clause.strip()
        if not clause:
            continue
        if connector == "THEN":
            if current:
                events.append(current)
            current = [clause]
        else:
            current.append(clause)
        connectors.append(connector)
    if current:
        events.append(current)
    return events, connectors


def explicit_query_clauses(query: str) -> list[str]:
    events, _ = parse_explicit_query(query)
    return [clause for event in events for clause in event]


def rebuild_explicit_query(query: str, translated_clauses: list[str]) -> str:
    _, connectors = parse_explicit_query(query)
    clauses = [str(value or "").strip() for value in translated_clauses]
    if not clauses:
        return str(query or "")
    rebuilt = clauses[0]
    for connector, clause in zip(connectors, clauses[1:]):
        rebuilt += f" {connector} {clause}"
    return rebuilt


def split_query(query: str) -> list[str]:
    query = query.strip()

    if not query:
        return []

    parts = re.split(
        r"\b(?:and then|then|after that|before that|after|before|and)\b|[,]",
        query,
        flags=re.IGNORECASE,
    )

    parts = [p.strip() for p in parts if p.strip()]

    if query not in parts:
        parts.insert(0, query)

    return list(dict.fromkeys(parts))
