"""
OCR Text Detection Module (Step 3) – Printed Urdu.

Detects text regions in scanned Urdu documents using pretrained YOLOv8.
No training, no dataset, no ContourNet/Detectron2. Windows-compatible.

Pipeline: input image -> YOLOv8 inference -> bounding boxes -> crop & save.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Defaults (pretrained model only)
# ---------------------------------------------------------------------------
# General-purpose YOLOv8 nano; first run downloads from Ultralytics.
# For printed Urdu text–specific detection, pass a custom .pt (e.g. authors’ checkpoint) via model_path.
_SCRIPT_DIR = Path(__file__).resolve().parent
URDU_DOC_MODEL_PATH = _SCRIPT_DIR / "urdu-text-detection" / "yolov8m_UrduDoc.pt"
# Use UrduDoc YOLOv8 (trained on UrduDoc for text lines) if present; else COCO nano (then OpenCV fallback).
DEFAULT_MODEL_PATH = str(URDU_DOC_MODEL_PATH) if URDU_DOC_MODEL_PATH.is_file() else "yolov8n.pt"
DEFAULT_CONF_THRESHOLD = 0.2        # Same as urdu-text-detection/detect.py
URDU_DOC_IMGSZ = 1280               # Same as detect.py for text line detection


def _opencv_fallback_crop(
    image: np.ndarray,
    image_path: Path,
    output_dir: Path,
) -> List[Path]:
    """When YOLOv8 returns 0 boxes (e.g. document with no COCO objects), use OpenCV text-region detection."""
    try:
        from preprocess import preprocess_image
        from text_region_detection import detect_text_blocks
    except ImportError as e:
        print(f"OpenCV fallback skipped (missing module): {e}", flush=True)
        return []
    try:
        binary_image, _ = preprocess_image(str(image_path))
    except Exception as e:
        print(f"OpenCV fallback skipped (preprocess failed): {e}", flush=True)
        return []
    # detect_text_blocks expects white (255) text on black (0); preprocess gives black-on-white
    binary_image = cv2.bitwise_not(binary_image)
    # Use larger dilation so we get line-level regions (~10–50) instead of word-level (100+)
    h, w = binary_image.shape[:2]
    kernel_w = max(60, min(180, w // 18))
    kernel_h = max(22, min(60, h // 45))
    boxes = detect_text_blocks(
        binary_image,
        kernel_width=kernel_w,
        kernel_height=kernel_h,
    )
    if not boxes:
        print("OpenCV fallback found no text regions; try a clearer or higher-resolution image.", flush=True)
        return []
    saved_paths: List[Path] = []
    for i, (x, y, w, h) in enumerate(boxes):
        x1, y1 = max(0, x), max(0, y)
        x2 = min(image.shape[1], x + w)
        y2 = min(image.shape[0], y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        crop_path = output_dir / f"{i}.png"
        cv2.imwrite(str(crop_path), crop)
        saved_paths.append(crop_path)
    return saved_paths


def _opencv_fallback_crop_with_boxes(
    image: np.ndarray,
    image_path: Path,
    output_dir: Path,
) -> Tuple[List[Tuple[int, int, int, int]], List[Path]]:
    """Same as _opencv_fallback_crop but returns (boxes_xywh, paths) for column-aware use. Does not affect Urdu pipeline."""
    try:
        from preprocess import preprocess_image
        from text_region_detection import detect_text_blocks
    except ImportError:
        return [], []
    try:
        binary_image, _ = preprocess_image(str(image_path))
    except Exception:
        return [], []
    binary_image = cv2.bitwise_not(binary_image)
    h, w = binary_image.shape[:2]
    kernel_w = max(60, min(180, w // 18))
    kernel_h = max(22, min(60, h // 45))
    boxes = detect_text_blocks(binary_image, kernel_width=kernel_w, kernel_height=kernel_h)
    if not boxes:
        return [], []
    saved_paths: List[Path] = []
    boxes_out: List[Tuple[int, int, int, int]] = []
    for i, (x, y, w, h) in enumerate(boxes):
        x1, y1 = max(0, x), max(0, y)
        x2 = min(image.shape[1], x + w)
        y2 = min(image.shape[0], y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        crop_path = output_dir / f"{i}.png"
        cv2.imwrite(str(crop_path), crop)
        saved_paths.append(crop_path)
        boxes_out.append((x, y, w, h))
    return boxes_out, saved_paths


def _sort_boxes_top_to_bottom_left_to_right(
    xyxy: np.ndarray,
    conf: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Sort boxes in reading order: top-to-bottom, then left-to-right.
    xyxy shape: (N, 4) as (x1, y1, x2, y2).
    """
    if xyxy is None or len(xyxy) == 0:
        return xyxy, conf
    # Use center y then center x for stable multi-column order
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
    order = np.lexsort((cx, cy))
    xyxy_sorted = xyxy[order]
    conf_sorted = conf[order] if conf is not None else None
    return xyxy_sorted, conf_sorted


def detect_and_crop_text(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    verbose: bool = False,
) -> List[Path]:
    """
    Load image, run YOLOv8 text detection, crop detected regions, and save crops.

    Uses pretrained YOLOv8 only (no training). Boxes are filtered by confidence,
    sorted top-to-bottom and left-to-right, then cropped from the original image
    and saved as 0.png, 1.png, ... in output_dir.

    Args:
        image_path: Path to input image (e.g. scanned Urdu document).
        output_dir: Directory where cropped text region images will be saved.
        model_path: Path or name of YOLOv8 model (e.g. yolov8n.pt); downloaded if missing.
        conf_threshold: Minimum confidence to keep a detection (0–1).
        verbose: If True, pass verbose=True to model.predict().

    Returns:
        List of paths to saved crop images (output_dir/0.png, 1.png, ...).
    """
    path = Path(image_path)
    out_dir = Path(output_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    # Load image with OpenCV (used for cropping; YOLOv8 can accept path or array)
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLOv8 model (UrduDoc .pt or yolov8n.pt)
    model_path_str = str(model_path)
    model = YOLO(model_path_str)

    # Run inference: use imgsz=1280 for UrduDoc (matches urdu-text-detection/detect.py)
    predict_kw: dict = {"source": str(path), "conf": conf_threshold, "verbose": verbose}
    if "UrduDoc" in model_path_str or Path(model_path_str).name == "yolov8m_UrduDoc.pt":
        predict_kw["imgsz"] = URDU_DOC_IMGSZ
    results = model.predict(**predict_kw)
    if not results:
        saved = _opencv_fallback_crop(image, path, out_dir)
        if saved:
            print("YOLOv8 returned no results; used OpenCV text-region fallback.", flush=True)
        return saved

    # Extract boxes: results[0].boxes.xyxy (x1,y1,x2,y2), results[0].boxes.conf
    r = results[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        saved = _opencv_fallback_crop(image, path, out_dir)
        if saved:
            print("YOLOv8 found no detections (COCO has no 'text' class); used OpenCV text-region fallback.", flush=True)
        return saved

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    # Apply confidence filter (model.predict already uses conf=conf_threshold; keep for clarity)
    mask = conf >= conf_threshold
    xyxy = xyxy[mask]
    conf = conf[mask]

    # After filtering, still no boxes → try OpenCV fallback
    if len(xyxy) == 0:
        saved = _opencv_fallback_crop(image, path, out_dir)
        if saved:
            print("YOLOv8 found no detections (COCO has no 'text' class); used OpenCV text-region fallback.", flush=True)
        return saved

    # Sort top-to-bottom, left-to-right before cropping
    xyxy, conf = _sort_boxes_top_to_bottom_left_to_right(xyxy, conf)

    # Crop each region from the original image and save
    saved_paths: List[Path] = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(image.shape[1], x2)), int(min(image.shape[0], y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        crop_path = out_dir / f"{i}.png"
        cv2.imwrite(str(crop_path), crop)
        saved_paths.append(crop_path)

    return saved_paths


def detect_and_crop_text_with_boxes(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    verbose: bool = False,
) -> Tuple[List[Tuple[int, int, int, int]], List[Path]]:
    """
    Same as detect_and_crop_text but also returns bounding boxes (x, y, w, h) for each crop.

    Used by English OCR for column-aware output. Urdu pipeline uses detect_and_crop_text only.
    Returns: (boxes_xywh, paths) where boxes[i] corresponds to paths[i].
    """
    path = Path(image_path).resolve()
    out_dir = Path(output_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path_str = str(model_path)
    model = YOLO(model_path_str)
    predict_kw: dict = {"source": str(path), "conf": conf_threshold, "verbose": verbose}
    if "UrduDoc" in model_path_str or Path(model_path_str).name == "yolov8m_UrduDoc.pt":
        predict_kw["imgsz"] = URDU_DOC_IMGSZ
    results = model.predict(**predict_kw)

    if not results:
        return _opencv_fallback_crop_with_boxes(image, path, out_dir)
    r = results[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return _opencv_fallback_crop_with_boxes(image, path, out_dir)

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    mask = conf >= conf_threshold
    xyxy = xyxy[mask]
    conf = conf[mask]
    if len(xyxy) == 0:
        return _opencv_fallback_crop_with_boxes(image, path, out_dir)

    xyxy, conf = _sort_boxes_top_to_bottom_left_to_right(xyxy, conf)
    boxes_xywh: List[Tuple[int, int, int, int]] = []
    saved_paths: List[Path] = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(image.shape[1], x2)), int(min(image.shape[0], y2))
        if x2 <= x1 or y2 <= y1:
            continue
        w, h = x2 - x1, y2 - y1
        boxes_xywh.append((x1, y1, w, h))
        crop = image[y1:y2, x1:x2]
        crop_path = out_dir / f"{i}.png"
        cv2.imwrite(str(crop_path), crop)
        saved_paths.append(crop_path)
    return boxes_xywh, saved_paths


# ---------------------------------------------------------------------------
# Main: run on sample Urdu document, save crops, print count
# ---------------------------------------------------------------------------
def _run_demo(
    image_path: str | Path,
    output_dir: str | Path | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> None:
    """Run detection on a sample image and save cropped text regions."""
    path = Path(image_path)
    if not path.is_file():
        print(f"Error: Image not found: {path}", flush=True)
        print("Usage: python text_detection_yolov8.py <image_path> [output_dir]", flush=True)
        return

    out = Path(output_dir) if output_dir else path.parent / f"{path.stem}_crops"
    print(f"Input: {path}")
    print(f"Output dir: {out}")
    print(f"Model: {model_path}")
    print(f"Confidence threshold: {conf_threshold}")

    saved = detect_and_crop_text(path, out, model_path=model_path, conf_threshold=conf_threshold)
    print(f"Detected and saved {len(saved)} text region(s).")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect text regions in printed Urdu documents using pretrained YOLOv8; crop and save.",
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to input image (e.g. scanned Urdu document).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Directory for cropped regions (default: <input_stem>_crops).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="YOLOv8 .pt model (default: urdu-text-detection/yolov8m_UrduDoc.pt if present, else yolov8n.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help=f"Confidence threshold (default: {DEFAULT_CONF_THRESHOLD}).",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = args.image_path.parent / f"{args.image_path.stem}_crops"
    model_path = args.model if args.model is not None else DEFAULT_MODEL_PATH

    _run_demo(args.image_path, out_dir, model_path=model_path, conf_threshold=args.conf)
