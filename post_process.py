"""
Post-processing for Urdu OCR pipeline.

Takes YOLOv8 box coordinates and per-crop OCR strings, then:
- Sorts boxes in Urdu reading order (top-to-bottom, right-to-left)
- Groups boxes into lines by vertical proximity
- Concatenates text per line in RTL order
- Cleans output (duplicate chars, non-Urdu ASCII noise; preserves Urdu punctuation)

No retraining, no external NLP. Python only.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# RTL and line grouping
# ---------------------------------------------------------------------------

# Vertical distance (in same units as box y coords) below which two boxes
# are considered on the same line. Tune for your layout/image scale.
DEFAULT_VERTICAL_THRESHOLD = 20


def _box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """(x1, y1, x2, y2) -> (center_x, center_y)."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _sort_boxes_reading_order_rtl(
    boxes: List[Tuple[float, float, float, float]],
    texts: List[str],
    vertical_threshold: float,
) -> List[Tuple[Tuple[float, float, float, float], str]]:
    """
    Sort boxes in Urdu reading order: top-to-bottom, then right-to-left within each line.

    RTL: Urdu is read from right to left, so the rightmost box on a line comes first.
    We group by line (similar y), then within each line sort by x descending (larger x = right).
    Primary sort by y (top to bottom); secondary sort by -center_x (right to left).
    """
    if not boxes or len(boxes) != len(texts):
        return list(zip(boxes, texts))

    # Pair and get centers
    paired = [(b, t) for b, t in zip(boxes, texts)]
    with_y = [(_box_center(b)[1], _box_center(b)[0], b, t) for b, t in paired]

    # Sort by y first (top to bottom)
    with_y.sort(key=lambda x: x[0])

    # Group into lines: consecutive boxes whose y-centers are within vertical_threshold
    lines: List[List[Tuple[Tuple[float, float, float, float], str]]] = []
    for _, _, b, t in with_y:
        cy = _box_center(b)[1]
        if not lines:
            lines.append([(b, t)])
            continue
        last_line = lines[-1]
        # Compare to representative y of current line (e.g. first box's y)
        ref_y = _box_center(last_line[0][0])[1]
        if abs(cy - ref_y) <= vertical_threshold:
            last_line.append((b, t))
        else:
            lines.append([(b, t)])

    # Within each line: sort right-to-left (descending x). Urdu is RTL: rightmost
    # box on a line is read first, so we order by -center_x (largest x first).
    out: List[Tuple[Tuple[float, float, float, float], str]] = []
    for line in lines:
        line.sort(key=lambda x: -_box_center(x[0])[0])
        out.extend(line)
    return out


def _clean_urdu_text(line: str) -> str:
    """
    Clean one line of OCR output for Urdu.

    - Remove consecutive duplicate characters (e.g. ااا -> ا).
    - Remove non-Urdu ASCII noise (control chars, Latin, etc.); keep spaces.
    - Preserve Urdu/Arabic script and common punctuation (، ؛ ؟ . etc.).
    """
    if not line:
        return line
    # Remove duplicate consecutive characters
    dedup = []
    for c in line:
        if dedup and c == dedup[-1]:
            continue
        dedup.append(c)
    s = "".join(dedup)
    # Keep: Arabic block (U+0600–U+06FF), Arabic Supplement (U+0750–U+077F),
    # Arabic Extended-A (U+08A0–U+08FF), space, ZWNJ, ZWJ, and common punctuation
    # Strip non-Urdu ASCII (Latin, digits if you want to drop them, etc.)
    kept = []
    for c in s:
        code = ord(c)
        if c.isspace() or c in "\u200c\u200d":  # ZWNJ, ZWJ
            kept.append(c)
        elif 0x0600 <= code <= 0x06FF:  # Arabic
            kept.append(c)
        elif 0x0750 <= code <= 0x077F:  # Arabic Supplement
            kept.append(c)
        elif 0x08A0 <= code <= 0x08FF:  # Arabic Extended-A
            kept.append(c)
        elif c in "،؛؟.۔!\"'()-–—":  # Common punctuation (Urdu/Arabic + neutral)
            kept.append(c)
        # Skip ASCII and other noise (Latin, control, etc.)
    cleaned = "".join(kept)
    # Normalize multiple spaces to one
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def post_process(
    boxes: Sequence[Union[Tuple[float, float, float, float], Sequence[float]]],
    texts: Sequence[str],
    *,
    vertical_threshold: float = DEFAULT_VERTICAL_THRESHOLD,
) -> str:
    """
    Post-process detected boxes and OCR strings into clean, multiline Urdu text in RTL order.

    Steps:
    1. Sort boxes in reading order: top-to-bottom (y), then right-to-left (x) within each line.
    2. Group boxes into lines using vertical_threshold (same line if y-distance <= threshold).
    3. Within each line, concatenate recognized text in RTL order (rightmost box first).
    4. Clean each line: remove duplicate consecutive chars and non-Urdu ASCII; keep Urdu punctuation.
    5. Return a single string with one line per visual line, suitable for RTL display.

    Args:
        boxes: List of (x1, y1, x2, y2) in image coordinates (e.g. from YOLOv8 xyxy).
        texts: List of recognized Urdu strings, one per box; same order as boxes.
        vertical_threshold: Max y-distance to consider two boxes on the same line (same units as y).

    Returns:
        Single multiline string: each line is cleaned and in RTL order; lines separated by newline.
    """
    if not boxes or not texts:
        return ""
    if len(boxes) != len(texts):
        texts = list(texts) + [""] * (len(boxes) - len(texts))
        texts = texts[: len(boxes)]

    # Normalize to list of 4-tuples (x1, y1, x2, y2)
    box_tuples = []
    for b in boxes:
        t = tuple(b)[:4] if hasattr(b, "__iter__") and not isinstance(b, str) else (0.0, 0.0, 0.0, 0.0)
        t = t + (0.0,) * (4 - len(t))
        box_tuples.append((float(t[0]), float(t[1]), float(t[2]), float(t[3])))
    text_list = list(texts)[: len(box_tuples)]

    # Sort in Urdu reading order (top-to-bottom, then right-to-left per line)
    sorted_pairs = _sort_boxes_reading_order_rtl(
        box_tuples, text_list, vertical_threshold
    )

    # Group again by line (same grouping as in sort) and concatenate text per line
    lines_text: List[str] = []
    current_line: List[str] = []
    current_y: float | None = None

    for box, text in sorted_pairs:
        cy = _box_center(box)[1]
        if current_y is None:
            current_y = cy
            current_line = [text]
            continue
        if abs(cy - current_y) <= vertical_threshold:
            current_line.append(text)
        else:
            # New line: emit previous line (already in RTL order: rightmost first)
            line_str = " ".join(current_line)
            lines_text.append(_clean_urdu_text(line_str))
            current_line = [text]
            current_y = cy

    if current_line:
        line_str = " ".join(current_line)
        lines_text.append(_clean_urdu_text(line_str))

    return "\n".join(lines_text)
