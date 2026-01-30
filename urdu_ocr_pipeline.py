"""
OCR Step 5 & 6: End-to-end printed Urdu OCR pipeline with structured output.

Chains YOLOv8 text detection (Step 3) and UTRNet-Large recognition (Step 4):
  Input image → YOLOv8 detection → crop regions → UTRNet recognition → Urdu text output.

Step 6 adds: structured JSON result (FYP/demo/evaluation-ready), lightweight logging.
No training, no datasets. Pretrained models only. Windows-compatible.
Structured output mirrors the authors’ HuggingFace demo: one result per image with
regions in reading order, so outputs are traceable and reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from text_detection_yolov8 import detect_and_crop_text
from urdu_recognition_utrnet import recognize_urdu


# ---------------------------------------------------------------------------
# Structured output (Step 6)
# ---------------------------------------------------------------------------
# Default path for JSON result: evaluation- and demo-ready, no extra deps.
DEFAULT_JSON_OUTPUT = Path(__file__).resolve().parent / "outputs" / "urdu_ocr_result.json"


def _build_ocr_result(image_name: str, lines: List[str]) -> dict:
    """
    Build the structured OCR result dict for JSON export.

    Structured output is used so results are:
    - Readable and parseable for FYP submission and reports
    - Traceable (image name, timestamp, region index = reading order)
    - Reproducible (same pipeline → same structure; future metrics can use this)
    Mirrors authors’ demo: one document → list of regions in reading order.
    """
    return {
        "image": image_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "regions": [
            {"id": i, "text": text}
            for i, text in enumerate(lines)
        ],
    }


def save_ocr_result_json(
    image_name: str,
    lines: List[str],
    output_path: str | Path = DEFAULT_JSON_OUTPUT,
) -> Path:
    """
    Save OCR result to a structured JSON file.

    Creates parent directory (e.g. outputs/) if needed. No external logging
    framework; caller may print the path for completion message.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = _build_ocr_result(image_name, lines)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output_path


def run_urdu_ocr(
    image_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    conf_threshold: float = 0.25,
    verbose: bool = True,
) -> List[str]:
    """
    Run full printed Urdu OCR on an image: detect text regions, crop, then recognize.

    Reuses Step 3 (YOLOv8 detect_and_crop_text) and Step 4 (recognize_urdu).
    Detections are already sorted top-to-bottom, left-to-right; crops are
    recognized in that order so the returned list preserves reading order.

    Args:
        image_path: Path to input image (e.g. scanned Urdu document).
        output_dir: Directory for cropped regions. If None, uses <image_stem>_crops
                    next to the input image.
        conf_threshold: YOLOv8 detection confidence threshold (0–1).
        verbose: If True, print detection count and recognition progress (lightweight logging).

    Returns:
        List of recognized Urdu text lines, one per detected region, in reading order.
    """
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    # Default crops directory next to the image
    if output_dir is None:
        output_dir = path.parent / f"{path.stem}_crops"
    output_dir = Path(output_dir)

    # Step 3: YOLOv8 detection → crop and save (boxes already sorted top-to-bottom, left-to-right)
    crop_paths = detect_and_crop_text(path, output_dir, conf_threshold=conf_threshold)
    if not crop_paths:
        if verbose:
            print("Detection count: 0")
        return []

    if verbose:
        print(f"Detection count: {len(crop_paths)}")

    # Step 4: Recognize each crop in order (order = reading order)
    n = len(crop_paths)
    lines: List[str] = []
    for i, crop_path in enumerate(crop_paths):
        if verbose:
            print(f"Recognizing region {i + 1}/{n} ...")
        try:
            text = recognize_urdu(crop_path)
        except Exception as e:
            text = f"[Error: {e}]"
        lines.append(text)

    return lines


# ---------------------------------------------------------------------------
# Main: accept image path, run pipeline, save JSON, print Urdu text
# ---------------------------------------------------------------------------
def _run_demo(
    image_path: str | Path,
    output_dir: str | Path | None = None,
    output_json: str | Path = DEFAULT_JSON_OUTPUT,
) -> None:
    """Run end-to-end Urdu OCR, save structured JSON, and print recognized text."""
    path = Path(image_path)
    if not path.is_file():
        print(f"Error: Image not found: {path}", file=sys.stderr)
        print("Usage: python urdu_ocr_pipeline.py <image_path> [output_dir] [--output-json PATH]", file=sys.stderr)
        return

    print(f"Input: {path.name}")
    print("Running: YOLOv8 detection → crop → UTRNet-Large recognition\n")
    lines = run_urdu_ocr(path, output_dir=output_dir, verbose=True)
    print()
    print(f"Recognized {len(lines)} line(s):\n")
    for i, text in enumerate(lines):
        print(text)
    print()

    # Step 6: save structured result for FYP/demo/evaluation
    json_path = save_ocr_result_json(path.name, lines, output_path=output_json)
    print(f"Result saved to: {json_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end printed Urdu OCR: YOLOv8 + UTRNet-Large, with structured JSON output.",
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to input image (scanned Urdu document).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Directory for cropped regions (default: <image_stem>_crops).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help=f"Path for structured JSON result (default: {DEFAULT_JSON_OUTPUT}).",
    )
    args = parser.parse_args()
    _run_demo(args.image_path, args.output_dir, args.output_json)
