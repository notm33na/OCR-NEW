"""
Production FastAPI service for bilingual OCR (Urdu: YOLOv8 + UTRNet, English: TrOCR).

- POST /ocr: image (multipart) + language (urdu | english | auto) -> JSON with text and regions.
- GET /health: readiness check.
- Models loaded once at startup; CPU-only safe; cross-platform paths; UTF-8 safe.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bilingual_ocr_api")

# ---------------------------------------------------------------------------
# UTF-8 and cross-platform paths (before any OCR imports that may print)
# ---------------------------------------------------------------------------
def _configure_utf8() -> None:
    """Force UTF-8 on Windows to avoid UnicodeEncodeError in logs or child processes."""
    import sys
    if getattr(sys, "platform", "") == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


_configure_utf8()

# Project root (cross-platform; no hardcoded C:\)
PROJECT_ROOT = Path(__file__).resolve().parent
# Upload directory: use system temp so we don't write to repo; works on Windows and Linux
UPLOAD_DIR = Path(tempfile.gettempdir()) / "ocr_uploads"


# ---------------------------------------------------------------------------
# Lazy imports for OCR (load only when needed; models loaded in lifespan)
# ---------------------------------------------------------------------------
def _import_ocr_modules():
    """Import pipeline modules; called after app is created so paths are set."""
    from text_detection_yolov8 import (
        detect_and_crop_text,
        detect_and_crop_text_with_boxes,
        preload_model,
    )
    from urdu_recognition_utrnet import get_urdu_inprocess_state, recognize_urdu
    from english_ocr_pipeline import recognize_english, recognize_english_page
    from script_detection import detect_script, detect_script_page
    from post_process import _clean_urdu_text
    from bilingual_ocr import _run_detection_and_crops
    return {
        "detect_and_crop_text": detect_and_crop_text,
        "detect_and_crop_text_with_boxes": detect_and_crop_text_with_boxes,
        "preload_model": preload_model,
        "get_urdu_inprocess_state": get_urdu_inprocess_state,
        "recognize_urdu": recognize_urdu,
        "recognize_english": recognize_english,
        "recognize_english_page": recognize_english_page,
        "detect_script": detect_script,
        "detect_script_page": detect_script_page,
        "_clean_urdu_text": _clean_urdu_text,
        "_run_detection_and_crops": _run_detection_and_crops,
    }


# ---------------------------------------------------------------------------
# Lifespan: load models once at startup
# ---------------------------------------------------------------------------
# Thread pool for CPU-bound OCR (don't block event loop)
_executor = ThreadPoolExecutor(max_workers=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create upload dir; load YOLOv8, UTRNet, and TrOCR once so first request is fast."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app.state.ocr = _import_ocr_modules()

    # Preload YOLOv8 model (shared for Urdu and auto detection)
    try:
        app.state.yolo_model = app.state.ocr["preload_model"]()
        logger.info("YOLOv8 model loaded at startup")
    except Exception as e:
        logger.warning("YOLOv8 model failed to load at startup: %s", e)
        app.state.yolo_model = None

    # Preload UTRNet (shared for Urdu and auto)
    app.state.urdu_state = app.state.ocr["get_urdu_inprocess_state"]()
    logger.info("UTRNet model loaded at startup")

    # Preload TrOCR so first English request doesn't lag
    try:
        from english_ocr_pipeline import _get_trocr
        _get_trocr()
        logger.info("TrOCR model loaded at startup")
    except Exception:
        pass

    logger.info("All models loaded — API ready")
    yield
    app.state.yolo_model = None
    app.state.urdu_state = None
    app.state.ocr = None


app = FastAPI(
    title="Bilingual OCR API",
    description="Urdu (YOLOv8 + UTRNet) and English (TrOCR) OCR; script detection for auto.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Response models (UTF-8 safe JSON only)
# ---------------------------------------------------------------------------
def _ocr_region(id: int, language: str, text: str, bbox: list[int] | None = None) -> dict[str, Any]:
    region: dict[str, Any] = {"id": id, "language": language, "text": text}
    if bbox is not None:
        region["bbox"] = bbox  # [x, y, w, h]
    return region


def _ocr_response(
    language_detected: str,
    text: str,
    regions: list[dict[str, Any]],
    *,
    text_regions: list[list[int]] | None = None,
    confidence: float = 1.0,
    detection_method: str = "yolov8",
) -> dict[str, Any]:
    resp: dict[str, Any] = {
        "language_detected": language_detected,
        "text": text,
        "regions": regions,
        "confidence": confidence,
        "detection_method": detection_method,
    }
    if text_regions is not None:
        resp["text_regions"] = text_regions
    return resp


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict[str, Any]:
    """Readiness check for Railway and load balancers."""
    return {
        "status": "ok",
        "yolo_loaded": getattr(app.state, "yolo_model", None) is not None,
        "urdu_loaded": getattr(app.state, "urdu_state", None) is not None,
    }


# ---------------------------------------------------------------------------
# POST /ocr — main OCR endpoint
# ---------------------------------------------------------------------------
def _run_ocr_sync(
    path: Path,
    language: str,
    ocr_mod: dict,
    urdu_state: Any,
    yolo_model: Any,
) -> dict[str, Any]:
    """Run OCR synchronously (called from thread pool). Returns JSON-serializable dict."""
    detect_and_crop_text_with_boxes = ocr_mod["detect_and_crop_text_with_boxes"]
    detect_and_crop_text = ocr_mod["detect_and_crop_text"]
    recognize_urdu = ocr_mod["recognize_urdu"]
    recognize_english = ocr_mod["recognize_english"]
    recognize_english_page = ocr_mod["recognize_english_page"]
    detect_script = ocr_mod["detect_script"]
    detect_script_page = ocr_mod["detect_script_page"]
    _clean_urdu_text = ocr_mod["_clean_urdu_text"]
    _run_detection_and_crops = ocr_mod["_run_detection_and_crops"]

    regions: list[dict[str, Any]] = []
    full_text_parts: list[str] = []
    text_regions: list[list[int]] = []
    detection_method = "yolov8"

    if language == "urdu":
        crop_dir = path.parent / f"{path.stem}_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)

        # --- YOLOv8 detection with boxes ---
        logger.info("Running YOLOv8 Urdu text detection on %s", path.name)
        try:
            boxes_xywh, crop_paths = detect_and_crop_text_with_boxes(
                path, crop_dir, verbose=False, model=yolo_model,
            )
        except Exception as e:
            logger.warning("YOLOv8 detection raised exception: %s — trying without cached model", e)
            try:
                boxes_xywh, crop_paths = detect_and_crop_text_with_boxes(
                    path, crop_dir, verbose=False,
                )
            except Exception as e2:
                logger.error("YOLOv8 detection failed completely: %s", e2)
                boxes_xywh, crop_paths = [], []

        logger.info("YOLOv8 detected %d text region(s)", len(crop_paths))

        # --- Fallback: if 0 regions, run full-image recognition ---
        if not crop_paths:
            logger.warning(
                "0 regions detected for %s — using full-image fallback", path.name,
            )
            detection_method = "full_image_fallback"
            try:
                text = recognize_urdu(path, _inprocess_state=urdu_state)
                text = _clean_urdu_text(text)
            except Exception as e:
                logger.error("Full-image Urdu recognition failed: %s", e)
                text = f"[Error: {e}]"
            regions.append(_ocr_region(0, "urdu", text))
            full_text_parts.append(text)
        else:
            # --- Normal path: recognize each crop ---
            text_regions = [list(b) for b in boxes_xywh]
            for i, crop_path in enumerate(crop_paths):
                bbox = list(boxes_xywh[i]) if i < len(boxes_xywh) else None
                try:
                    text = recognize_urdu(crop_path, _inprocess_state=urdu_state)
                    text = _clean_urdu_text(text)
                except Exception as e:
                    logger.error("Urdu recognition error on crop %d: %s", i, e)
                    text = f"[Error: {e}]"
                regions.append(_ocr_region(i, "urdu", text, bbox=bbox))
                full_text_parts.append(text)
            logger.info("Urdu recognition completed: %d region(s)", len(regions))

        language_detected = "urdu"

    elif language == "english":
        logger.info("Running English OCR (TrOCR) on %s", path.name)
        detection_method = "trocr_page"
        try:
            text = recognize_english_page(path)
        except Exception as e:
            logger.error("English recognition failed: %s", e)
            text = f"[Error: {e}]"
        regions = [_ocr_region(0, "english", text)]
        full_text_parts = [text]
        language_detected = "english"

    else:
        # --- Auto: detect script per region ---
        crop_dir = path.parent / f"{path.stem}_auto_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Running YOLOv8 auto-detection on %s", path.name)
        try:
            boxes_xywh, crop_paths = detect_and_crop_text_with_boxes(
                path, crop_dir, verbose=False, model=yolo_model,
            )
        except Exception as e:
            logger.warning("YOLOv8 auto-detection failed: %s — retrying without cached model", e)
            try:
                boxes_xywh, crop_paths = detect_and_crop_text_with_boxes(
                    path, crop_dir, verbose=False,
                )
            except Exception as e2:
                logger.error("Auto-detection failed completely: %s", e2)
                boxes_xywh, crop_paths = [], []

        logger.info("Auto-detection found %d text region(s)", len(crop_paths))
        page_script_urdu = detect_script_page(path)

        if not crop_paths:
            # Fallback: full-image recognition with script detection
            logger.warning(
                "0 regions detected (auto) for %s — using full-image fallback", path.name,
            )
            detection_method = "full_image_fallback"
            script_result = detect_script(path)
            script = script_result.get("script", "english")
            try:
                if script == "urdu":
                    text = recognize_urdu(path, _inprocess_state=urdu_state)
                    text = _clean_urdu_text(text)
                else:
                    text = recognize_english(path, skip_preprocess=True)
            except Exception as e:
                logger.error("Full-image fallback recognition failed: %s", e)
                text = f"[Error: {e}]"
                script = "unknown"
            regions.append(_ocr_region(0, script, text))
            full_text_parts.append(text)
        else:
            text_regions = [list(b) for b in boxes_xywh]
            for i, crop_path in enumerate(crop_paths):
                script = (
                    "urdu"
                    if page_script_urdu == "urdu"
                    else detect_script(crop_path).get("script", "english")
                )
                bbox = list(boxes_xywh[i]) if i < len(boxes_xywh) else None
                try:
                    if script == "urdu":
                        text = recognize_urdu(crop_path, _inprocess_state=urdu_state)
                        text = _clean_urdu_text(text)
                    else:
                        text = recognize_english(crop_path, skip_preprocess=True)
                except Exception as e:
                    logger.error("Recognition error on crop %d: %s", i, e)
                    text = f"[Error: {e}]"
                    script = "unknown"
                regions.append(_ocr_region(i, script, text, bbox=bbox))
                full_text_parts.append(text)
            logger.info("Auto recognition completed: %d region(s)", len(regions))

        language_detected = (
            "urdu" if any(r["language"] == "urdu" for r in regions) else "english"
        )

    full_text = "\n".join(full_text_parts)
    return _ocr_response(
        language_detected,
        full_text,
        regions,
        text_regions=text_regions or None,
        confidence=1.0,
        detection_method=detection_method,
    )


@app.post("/ocr")
async def ocr(
    image: UploadFile = File(...),
    language: str = Form("auto"),
) -> dict[str, Any]:
    """
    Run OCR on uploaded image.

    - language: "urdu" | "english" | "auto" (script detection per region).
    - Returns JSON: language_detected, text (combined), regions (id, language, text, bbox),
      text_regions (list of [x,y,w,h]), confidence, detection_method.
    """
    if language not in ("urdu", "english", "auto"):
        raise HTTPException(400, detail="language must be urdu, english, or auto")

    # 1. Save uploaded image to temp uploads dir (cross-platform)
    suffix = Path(image.filename or "image").suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png"):
        suffix = ".jpg"
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        contents = await image.read()
        path.write_bytes(contents)
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to save upload: {e}") from e

    try:
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            _run_ocr_sync,
            path,
            language,
            app.state.ocr,
            app.state.urdu_state,
            app.state.yolo_model,
        )
        return result
    except Exception as e:
        logger.exception("Unhandled error processing %s", image.filename)
        raise HTTPException(500, detail=str(e)) from e
    finally:
        # Clean up: remove uploaded file and any crop dir we created
        try:
            if path.exists():
                path.unlink()
            for d in (
                path.parent / f"{path.stem}_crops",
                path.parent / f"{path.stem}_auto_crops",
            ):
                if d.exists() and d.is_dir():
                    for f in d.iterdir():
                        try:
                            f.unlink()
                        except Exception:
                            pass
                    try:
                        d.rmdir()
                    except Exception:
                        pass
        except Exception:
            pass
