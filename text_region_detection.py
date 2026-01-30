"""
OCR Text Region Detection Module (Step 2 of pipeline).

Detects text blocks (paragraphs / columns) from a preprocessed binary image
using classical OpenCV: morphology, contours, and size-based filtering.
No ML, no Detectron2. Windows-compatible. Multi-column safe.

Pipeline: binary image -> dilation -> contours -> filter -> sort -> boxes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Default parameters (tunable for different document layouts)
# ---------------------------------------------------------------------------
# Rectangular kernel: merges nearby text horizontally and vertically into blocks.
# Larger values = fewer, larger blocks; smaller = more, finer blocks.
DILATION_KERNEL_WIDTH = 40   # Merge text along horizontal direction (words/lines)
DILATION_KERNEL_HEIGHT = 15  # Merge text along vertical direction (lines in a block)

# Size-based filtering: drop tiny noise and full-page artefacts.
MIN_BLOCK_WIDTH = 20        # Ignore contours narrower than this (noise, vertical lines)
MIN_BLOCK_HEIGHT = 15       # Ignore contours shorter than this (dots, noise)
MAX_BLOCK_WIDTH_RATIO = 0.95   # Ignore contours wider than this fraction of image width
MAX_BLOCK_HEIGHT_RATIO = 0.95  # Ignore contours taller than this fraction of image height

# Visualization
BOX_COLOR = (0, 255, 0)     # BGR green for bounding boxes
TEXT_COLOR = (0, 255, 0)     # BGR green for labels
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1


def detect_text_blocks(
    binary_image: np.ndarray,
    *,
    kernel_width: int = DILATION_KERNEL_WIDTH,
    kernel_height: int = DILATION_KERNEL_HEIGHT,
    min_width: int = MIN_BLOCK_WIDTH,
    min_height: int = MIN_BLOCK_HEIGHT,
    max_width_ratio: float = MAX_BLOCK_WIDTH_RATIO,
    max_height_ratio: float = MAX_BLOCK_HEIGHT_RATIO,
) -> list[tuple[int, int, int, int]]:
    """
    Detect text regions (paragraphs / blocks) from a binary document image.

    Uses morphological dilation to merge nearby text (characters, words, lines)
    into coherent blocks, then finds contours and filters by size. Resulting
    boxes are sorted top-to-bottom, left-to-right for multi-column and single-
    column documents.

    Args:
        binary_image: Single-channel binary image (e.g. from preprocess.py).
                      Typically white (255) text on black (0) background.
        kernel_width: Width of rectangular dilation kernel (horizontal merge).
        kernel_height: Height of rectangular dilation kernel (vertical merge).
        min_width: Minimum bounding box width; smaller contours are discarded.
        min_height: Minimum bounding box height; smaller contours are discarded.
        max_width_ratio: Maximum width as fraction of image width (filter full-page).
        max_height_ratio: Maximum height as fraction of image height (filter full-page).

    Returns:
        List of bounding boxes as (x, y, w, h), in reading order (top-to-bottom,
        left-to-right).
    """
    if binary_image is None or binary_image.size == 0:
        return []

    # Ensure we have a single-channel image (grayscale/binary)
    if len(binary_image.shape) == 3:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    # Dilation: merge nearby white (text) regions into blocks.
    # Rectangular kernel: horizontal merge (words/lines) and vertical merge (lines).
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(1, kernel_width), max(1, kernel_height)),
    )
    dilated = cv2.dilate(binary_image, kernel)

    # Contour detection: find boundaries of white regions (text blocks).
    # RETR_EXTERNAL: only outer contours (one per connected component).
    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    img_h, img_w = binary_image.shape[:2]
    max_w = int(max_width_ratio * img_w)
    max_h = int(max_height_ratio * img_h)

    boxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Size-based filtering: drop noise and full-page artefacts
        if w < min_width or h < min_height:
            continue
        if w > max_w or h > max_h:
            continue
        boxes.append((x, y, w, h))

    # Sort top-to-bottom, then left-to-right (reading order; multi-column safe).
    # Primary key: y (row). Secondary key: x (column).
    boxes.sort(key=lambda b: (b[1], b[0]))

    return boxes


def visualize_blocks(
    image: np.ndarray,
    boxes: Sequence[tuple[int, int, int, int]],
    *,
    color: tuple[int, int, int] = BOX_COLOR,
    label_color: tuple[int, int, int] = TEXT_COLOR,
    thickness: int = 2,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """
    Draw bounding boxes and block indices on the image for inspection.

    Args:
        image: Image to draw on (BGR or grayscale; converted to BGR for drawing).
        boxes: List of (x, y, w, h) bounding boxes.
        color: BGR color for rectangle outlines.
        label_color: BGR color for index text.
        thickness: Line thickness for rectangles.
        output_path: If set, save the visualization to this path.

    Returns:
        Copy of the image with boxes and labels drawn (BGR).
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    for i, (x, y, w, h) in enumerate(boxes):
        # Draw rectangle around the text block
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
        # Label with block index (top-left corner, slightly above box)
        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        # Background for label readability
        cv2.rectangle(
            vis,
            (x, y - th - 4),
            (x + tw + 4, y),
            color,
            -1,
        )
        cv2.putText(
            vis,
            label,
            (x + 2, y - 2),
            FONT,
            FONT_SCALE,
            label_color,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), vis)

    return vis


def _resize_for_display(image: np.ndarray, max_side: int = 1200) -> np.ndarray:
    """Resize image so it fits on screen while preserving aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    return cv2.resize(
        image,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


# ---------------------------------------------------------------------------
# Main: load preprocessed image, detect blocks, visualize
# ---------------------------------------------------------------------------
def _run_demo(
    image_path: str | Path,
    output_path: str | Path | None = None,
    display: bool = True,
) -> None:
    """
    Load a preprocessed (binary) image, run text block detection, and
    visualize the detected regions (optionally display and save).
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    # Load image (preprocessed binary from Step 1, or grayscale/BGR)
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {path}")

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Ensure binary 0/255: preprocess output is 0 and 255; normalize if 0/1
    if img.max() <= 1:
        img = (img * 255).astype(np.uint8)

    # Detect text blocks
    boxes = detect_text_blocks(img)
    print(f"Detected {len(boxes)} text block(s).")

    # Visualize
    vis = visualize_blocks(img, boxes, output_path=output_path)
    if output_path:
        print(f"Saved visualization: {output_path}")

    if display:
        vis_resized = _resize_for_display(vis)
        cv2.imshow("Text block detection", vis_resized)
        print("Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect text regions (blocks) in a preprocessed binary image.",
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to preprocessed binary image (e.g. from preprocess.py).",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path to save visualization image (default: <input_stem>_blocks.png).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not show OpenCV window; only save visualization if -o is set.",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = args.image_path.parent / f"{args.image_path.stem}_blocks.png"

    _run_demo(args.image_path, output_path=output, display=not args.no_display)
