"""
English OCR pipeline: handwritten text recognition using TrOCR (HuggingFace).

- Pretrained TrOCR handwritten English model (microsoft/trocr-base-handwritten).
- Processor and model loaded once and reused for all images.
- Reuses preprocess.py for image preprocessing (grayscale, denoise, deskew).
- Accepts single image path or directory of line images (png/jpg).
- Output: plain English text only. No training, no datasets. CPU-only.
- Fully independent from Urdu OCR; does not modify any Urdu pipeline files.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from PIL import Image

from preprocess import preprocess_image
from text_detection_yolov8 import detect_and_crop_text_with_boxes

# ---------------------------------------------------------------------------
# TrOCR model cache (load once, reuse)
# ---------------------------------------------------------------------------
TROCR_MODEL_ID = "microsoft/trocr-base-handwritten"
_processor = None
_model = None


def _get_trocr():
    """Load TrOCR processor and model once; return cached (processor, model)."""
    global _processor, _model
    if _processor is None or _model is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()  # silence "slow processor", "weights not initialized", "TRAIN this model"
        try:
            _processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID, use_fast=True)
        except TypeError:
            _processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
        _model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_ID)
        _model.eval()
        hf_logging.set_verbosity_warning()  # restore so other code can see warnings if needed
    return _processor, _model


def recognize_english(image_path: str | Path, *, skip_preprocess: bool = False) -> str:
    """
    Recognize handwritten English text in a single image (or crop) using TrOCR.

    By default: preprocess (grayscale, denoise, deskew) then TrOCR.
    If skip_preprocess=True (e.g. for a small crop), only runs TrOCR.

    Args:
        image_path: Path to input image (full page, line, or crop).
        skip_preprocess: If True, do not run preprocess; use image as-is (for region crops).

    Returns:
        Recognized English text string.
    """
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    if skip_preprocess:
        pil_image = Image.open(path).convert("RGB")
    else:
        try:
            _, gray_deskewed = preprocess_image(path)
        except Exception as e:
            raise ValueError(f"Preprocessing failed for {path}: {e}") from e
        pil_image = Image.fromarray(gray_deskewed).convert("RGB")

    processor, model = _get_trocr()
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    # Allow full paragraphs/pages: default max_length=20 was truncating to a few words
    generated_ids = model.generate(pixel_values, max_new_tokens=512)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


def _recognize_english_fullpage_by_strips(image_path: str | Path, num_strips: int = 10) -> str:
    """
    Fallback when region detection fails: split the preprocessed page into horizontal
    strips and run TrOCR on each strip. Avoids feeding the whole page at once (which
    causes TrOCR to hallucinate short phrases like '4 hours').
    """
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        _, gray_deskewed = preprocess_image(path)
    except Exception:
        return recognize_english(path)
    h, w = gray_deskewed.shape[:2]
    if h < 100:
        return recognize_english(path)
    strip_height = max(h // num_strips, 80)
    texts: List[str] = []
    with tempfile.TemporaryDirectory(prefix="ocr_strips_") as tmp:
        tmp_path = Path(tmp)
        for i in range(num_strips):
            y1 = i * strip_height
            y2 = min(y1 + strip_height, h)
            if y2 <= y1:
                break
            strip = gray_deskewed[y1:y2, :]
            if strip.size == 0:
                continue
            pil_strip = Image.fromarray(strip).convert("RGB")
            strip_file = tmp_path / f"strip_{i:02d}.png"
            pil_strip.save(str(strip_file))
            try:
                t = recognize_english(strip_file, skip_preprocess=True)
                if t and t.strip():
                    texts.append(t.strip())
            except Exception:
                pass
    return "\n".join(texts) if texts else recognize_english(path)


def recognize_english_page(image_path: str | Path, crop_dir: Path | None = None) -> str:
    """
    Recognize full-page English: use shared detector (YOLOv8 UrduDoc or OpenCV fallback),
    split regions into LEFT and RIGHT columns by position, run TrOCR per crop, output columns separately.

    Uses detect_and_crop_text_with_boxes from text_detection_yolov8; does not change Urdu pipeline.
    """
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    if crop_dir is None:
        base = path.resolve().parent.parent / "outputs" / "english_crops"
        crop_dir = base / path.stem
    crop_dir = Path(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)

    try:
        boxes_xywh, paths = detect_and_crop_text_with_boxes(path, crop_dir)
    except Exception:
        return _recognize_english_fullpage_by_strips(path)

    if not boxes_xywh or not paths or len(boxes_xywh) != len(paths):
        return _recognize_english_fullpage_by_strips(path)

    img_w_approx = max(x + w for (x, y, w, h) in boxes_xywh)
    mid_x = img_w_approx / 2.0
    left_pairs: List[Tuple[Tuple[int, int, int, int], Path]] = []
    right_pairs: List[Tuple[Tuple[int, int, int, int], Path]] = []
    for box, p in zip(boxes_xywh, paths):
        x, y, w, h = box
        center_x = x + w / 2.0
        if center_x < mid_x:
            left_pairs.append((box, p))
        else:
            right_pairs.append((box, p))
    left_pairs.sort(key=lambda b: b[0][1])
    right_pairs.sort(key=lambda b: b[0][1])

    def run_ocr(paths_in_order: List[Path]) -> List[str]:
        texts: List[str] = []
        for p in paths_in_order:
            try:
                t = recognize_english(p, skip_preprocess=True)
                if t:
                    texts.append(t)
            except Exception:
                pass
        return texts

    left_texts = run_ocr([p for _, p in left_pairs])
    right_texts = run_ocr([p for _, p in right_pairs])

    parts: List[str] = []
    if left_texts:
        parts.append("--- Left column ---")
        parts.extend(left_texts)
    if right_texts:
        if parts:
            parts.append("")
        parts.append("--- Right column ---")
        parts.extend(right_texts)
    if not parts:
        return _recognize_english_fullpage_by_strips(path)
    return "\n".join(parts)


def recognize_all(input_path: str | Path, *, full_page: bool = True) -> List[dict]:
    """
    Recognize English text in a single image or all images in a directory.

    If full_page=True (default): detect text regions first, run TrOCR per region, combine.
    If full_page=False: run TrOCR on the whole image (faster for single-line images).

    Returns:
        List of {"image": "<filename>", "text": "<recognized string>"} in order.
    """
    path = Path(input_path).resolve()
    if path.is_file():
        name = path.name
        try:
            text = recognize_english_page(path) if full_page else recognize_english(path)
        except Exception as e:
            text = f"[Error: {e}]"
        return [{"image": name, "text": text}]

    if not path.is_dir():
        return []

    exts = {".png", ".jpg", ".jpeg"}
    files = sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: p.name.lower(),
    )
    results = []
    for p in files:
        try:
            text = recognize_english_page(p) if full_page else recognize_english(p)
        except Exception as e:
            text = f"[Error: {e}]"
        results.append({"image": p.name, "text": text})
    return results


# ---------------------------------------------------------------------------
# Optional: JSON output (same structure as urdu_ocr_pipeline)
# ---------------------------------------------------------------------------
DEFAULT_JSON_OUTPUT = Path(__file__).resolve().parent / "outputs" / "english_ocr_result.json"


def save_ocr_result_json(
    image_name: str,
    lines: List[str],
    output_path: str | Path = DEFAULT_JSON_OUTPUT,
) -> Path:
    """
    Save OCR result to structured JSON (same schema as urdu_ocr_pipeline).

    Structure: {"image": name, "timestamp": iso, "regions": [{"id": i, "text": t}, ...]}.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "image": image_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "regions": [{"id": i, "text": t} for i, t in enumerate(lines)],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output_path


# ---------------------------------------------------------------------------
# Main: quick testing on one image (or directory)
# ---------------------------------------------------------------------------
def _main() -> None:
    parser = argparse.ArgumentParser(
        description="English OCR (handwritten): TrOCR + preprocess. Single image or directory.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a single image or to a directory of images (png/jpg).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="If set, save result to this JSON file (same structure as Urdu pipeline).",
    )
    parser.add_argument(
        "--whole-image",
        action="store_true",
        help="Run TrOCR on the whole image only (no region detection). Use for single-line images.",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"Error: Not found: {args.input_path}", file=sys.stderr)
        sys.exit(1)

    results = recognize_all(args.input_path, full_page=not args.whole_image)
    if not results:
        print("No images processed.", file=sys.stderr)
        sys.exit(1)

    for r in results:
        print(f"[{r['image']}]")
        print(r["text"])
        print()

    if args.output_json is not None:
        # One JSON per image or one JSON with first image name and all texts as regions
        if len(results) == 1:
            save_ocr_result_json(
                results[0]["image"],
                [results[0]["text"]],
                output_path=args.output_json,
            )
        else:
            # Multiple images: save as one JSON with regions = one per image
            name = args.input_path.name if args.input_path.is_dir() else results[0]["image"]
            lines = [r["text"] for r in results]
            save_ocr_result_json(name, lines, output_path=args.output_json)
        print(f"Result saved to: {args.output_json}")


if __name__ == "__main__":
    _main()
