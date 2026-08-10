from __future__ import annotations

from collections import defaultdict

from godseye.domain import TimelineEvent


def build_timeline(events: list[TimelineEvent]) -> dict[str, list[TimelineEvent]]:
    grouped: dict[str, list[TimelineEvent]] = defaultdict(list)
    for event in events:
        grouped[event.person_id].append(event)

    return {
        person_id: sorted(items, key=lambda item: item.entry_time_s)
        for person_id, items in grouped.items()
    }


def summarize_timeline(events: list[TimelineEvent]) -> str:
    if not events:
        return "No timeline events."

    ordered = sorted(events, key=lambda item: item.entry_time_s)
    person_id = ordered[0].person_id
    path = " -> ".join(event.camera_id for event in ordered)
    total_time = max(event.exit_time_s for event in ordered) - min(
        event.entry_time_s for event in ordered
    )

    return f"Person {person_id}: {path}. Total time: {total_time:.1f}s."

