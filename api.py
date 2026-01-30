"""
Bilingual OCR API - FastAPI backend for single-image OCR.

Production-ready for Railway deployment. Uses bilingual_ocr.process_image() for OCR.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

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
# Import bilingual_ocr and set OCR readiness
# ---------------------------------------------------------------------------
OCR_READY = False
bilingual_ocr = None

try:
    import bilingual_ocr as _bilingual_ocr
    bilingual_ocr = _bilingual_ocr
    OCR_READY = True
    logger.info("bilingual_ocr module loaded successfully; OCR is ready")
except Exception as e:
    logger.warning("bilingual_ocr module not available: %s", e)

# ---------------------------------------------------------------------------
# App and paths
# ---------------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parent
TEMP_UPLOADS = APP_ROOT / "temp_uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}

app = FastAPI(
    title="Bilingual OCR API",
    version="1.0.0",
    description="OCR API for Urdu and English images (YOLOv8 + UTRNet, TrOCR).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    """Create temp_uploads directory at startup (Railway / production)."""
    _ensure_temp_uploads()
    logger.info("temp_uploads directory ready")


def _ensure_temp_uploads() -> Path:
    TEMP_UPLOADS.mkdir(parents=True, exist_ok=True)
    return TEMP_UPLOADS


def _is_allowed_file(filename: str | None, content_type: str | None) -> bool:
    if not filename:
        return False
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return False
    if content_type and content_type.lower() not in ALLOWED_CONTENT_TYPES:
        # Be lenient: some clients send wrong content-type
        return True
    return True


# ---------------------------------------------------------------------------
# GET / - Root
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint: welcome, version, status, and available endpoints."""
    status = "ready" if OCR_READY else "initializing"
    return {
        "message": "Welcome to the Bilingual OCR API",
        "version": "1.0.0",
        "status": status,
        "endpoints": [
            {"path": "/", "method": "GET", "description": "This info"},
            {"path": "/health", "method": "GET", "description": "Health and OCR readiness"},
            {"path": "/ocr", "method": "POST", "description": "Upload image and run OCR (form: file, lang)"},
        ],
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check: status and OCR readiness."""
    return {
        "status": "healthy",
        "ocr_ready": OCR_READY,
    }


# ---------------------------------------------------------------------------
# POST /ocr
# ---------------------------------------------------------------------------
@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    lang: str = Form("auto"),
):
    """
    Run OCR on uploaded image.

    - file: image file (jpg/png).
    - lang: "auto" (script detection), "urdu", or "english".
    Returns: success, filename, language, extracted text, details.
    """
    if not OCR_READY or bilingual_ocr is None:
        logger.error("OCR requested but module not ready")
        raise HTTPException(status_code=503, detail="OCR service not ready")

    if lang not in ("auto", "urdu", "english"):
        lang = "auto"

    filename = file.filename or "image"
    if not _is_allowed_file(filename, file.content_type):
        logger.warning("Rejected file: filename=%s content_type=%s", filename, file.content_type)
        raise HTTPException(status_code=400, detail="Invalid file type. Use jpg or png.")

    suffix = Path(filename).suffix.lower() or ".jpg"
    if suffix not in ALLOWED_EXTENSIONS:
        suffix = ".jpg"
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    temp_path = _ensure_temp_uploads() / safe_name

    try:
        contents = await file.read()
        temp_path.write_bytes(contents)
    except Exception as e:
        logger.exception("Failed to save upload %s", filename)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}") from e

    try:
        logger.info("Processing image: filename=%s lang=%s", filename, lang)
        result = bilingual_ocr.process_image(temp_path, lang=lang)

        if not result.get("success"):
            error_msg = result.get("error", "OCR failed")
            logger.warning("OCR failed for %s: %s", filename, error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        return {
            "success": True,
            "filename": filename,
            "language": result.get("language", "unknown"),
            "extracted_text": result.get("text", ""),
            "details": result.get("details", []),
            "confidence": result.get("confidence", 1.0),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Processing error for %s: %s", filename, e)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        # Always clean up temporary file
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception as e:
            logger.warning("Cleanup failed for %s: %s", temp_path, e)


# ---------------------------------------------------------------------------
# Local runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    _ensure_temp_uploads()
    uvicorn.run(app, host="0.0.0.0", port=8000)
