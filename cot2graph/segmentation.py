"""Rule-based segmentation of a CoT into reasoning steps (Section 3.3)."""

from __future__ import annotations

import re


def build_marker_pattern(markers: list[str]) -> re.Pattern:
    # Longest markers first, so "Let me verify" wins over "Verify".
    escaped = [re.escape(m) for m in sorted(markers, key=len, reverse=True)]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)


def segment_cot(text: str, pattern: re.Pattern, min_chars: int = 100) -> dict[str, str]:
    """Split ``text`` at transition markers and merge fragments shorter than ``min_chars``."""
    if not text:
        return {}

    nodes, buffer = [], ""
    for part in pattern.split(text):
        if pattern.fullmatch(part.strip()):
            if buffer:
                nodes.append(buffer.strip())
            buffer = part.strip() + " "
        else:
            buffer += part.strip() + " "
    if buffer:
        nodes.append(buffer.strip())

    merged = []
    for i, node in enumerate(nodes):
        node = node.strip()
        content_len = len(pattern.sub("", node).strip())
        if content_len < min_chars and i + 1 < len(nodes):
            nodes[i + 1] = node + " " + nodes[i + 1]
        else:
            merged.append(node)
    return {str(i): node for i, node in enumerate(merged)}
