from __future__ import annotations

import re


def split_query(query: str) -> list[str]:
    query = query.strip()

    if not query:
        return []

    parts = re.split(
        r"\b(?:and then|then|after that|before that|after|before|and)\b|[,;]",
        query,
        flags=re.IGNORECASE,
    )

    parts = [p.strip() for p in parts if p.strip()]

    if query not in parts:
        parts.insert(0, query)

    return list(dict.fromkeys(parts))