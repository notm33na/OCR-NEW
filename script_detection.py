"""
Script detection for OCR pipelines: Urdu vs English.

Uses lightweight pytesseract OCR and Unicode range checks only.
No ML models, no training. Explainable and FYP-safe.
Reusable by future pipelines (bilingual routing, etc.).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Unicode ranges (explainable classification)
# ---------------------------------------------------------------------------
# Arabic block: used for Urdu and Arabic script (FYP-safe heuristic).
ARABIC_START = 0x0600
ARABIC_END = 0x06FF


def _has_arabic_script(text: str) -> bool:
    """
    Return True if any character in text falls in the Arabic Unicode block (U+0600–U+06FF).

    Urdu uses the Arabic script, so presence of this range indicates Urdu (or Arabic).
    No ML; purely explainable Unicode check.
    """
    if not text:
        return False
    for char in text:
        if ARABIC_START <= ord(char) <= ARABIC_END:
            return True
    return False


def _tesseract_string(image_arg, *, lang: str) -> str:
    """Run Tesseract and return text; empty on any error."""
    try:
        import pytesseract
        if isinstance(image_arg, (str, Path)):
            return pytesseract.image_to_string(str(image_arg), lang=lang) or ""
        from PIL import Image
        import numpy as np
        arr = image_arg
        if len(arr.shape) == 3:
            arr = arr[:, :, ::-1]
        pil = Image.fromarray(arr)
        return pytesseract.image_to_string(pil, lang=lang) or ""
    except Exception:
        return ""


def _has_latin_letters(text: str) -> bool:
    """True if text has at least one Latin letter (A-Z, a-z)."""
    for c in text:
        if "a" <= c <= "z" or "A" <= c <= "Z":
            return True
    return False


def detect_script_page(image: str | Path) -> Literal["urdu"] | None:
    """
    Page-level script check: only treat page as "all Urdu" when page has Arabic and no clear English.

    Run Tesseract with lang="eng" first on the full page. If eng output has Latin letters → None
    (page is English or mixed; use per-crop detection). If eng is empty or has no Latin, run lang="ara";
    if ara has Arabic → "urdu" (treat all crops on this page as Urdu). Else → None.
    This keeps all-English pages (e.g. QSL card) from being forced to Urdu when ara returns noise.
    """
    if isinstance(image, (str, Path)) and not Path(image).is_file():
        return None
    text_eng = _tesseract_string(image, lang="eng")
    if text_eng.strip() and _has_latin_letters(text_eng):
        return None
    text_ara = _tesseract_string(image, lang="ara")
    return "urdu" if _has_arabic_script(text_ara) else None


def detect_script(image: str | Path | "np.ndarray") -> dict:
    """
    Detect script (Urdu vs English): English only when eng output clearly has Latin text.

    1. Run Tesseract with lang="eng" first.
    2. If output has Arabic Unicode → "urdu".
    3. If output has Latin letters (A–Z, a–z) and no Arabic → "english".
    4. If output is empty or has no Latin letters (e.g. numbers only, or Urdu crop) → try lang="ara";
       if ara has Arabic → "urdu", else "english".
    So: all-English pages get Latin from eng → English; Urdu crops get little/no Latin from eng, we try ara → Urdu.
    """
    if isinstance(image, (str, Path)) and not Path(image).is_file():
        return {"script": "english", "confidence": "heuristic"}

    text_eng = _tesseract_string(image, lang="eng")
    if _has_arabic_script(text_eng):
        return {"script": "urdu", "confidence": "heuristic"}
    if text_eng.strip() and _has_latin_letters(text_eng):
        return {"script": "english", "confidence": "heuristic"}

    text_ara = _tesseract_string(image, lang="ara")
    script: Literal["urdu", "english"] = "urdu" if _has_arabic_script(text_ara) else "english"
    return {"script": script, "confidence": "heuristic"}


# ---------------------------------------------------------------------------
# Main: test detection on sample images
# ---------------------------------------------------------------------------
def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect script (Urdu vs English) from images using pytesseract + Unicode checks.",
    )
    parser.add_argument(
        "images",
        type=Path,
        nargs="*",
        help="Paths to sample images to test. If none, print usage.",
    )
    args = parser.parse_args()

    if not args.images:
        print("Usage: python script_detection.py <image1> [image2 ...]", file=sys.stderr)
        print("Example: python script_detection.py doc.png", file=sys.stderr)
        sys.exit(0)

    for path in args.images:
        if not path.is_file():
            print(f"Skip (not found): {path}", file=sys.stderr)
            continue
        result = detect_script(path)
        print(f"{path.name}: script={result['script']}, confidence={result['confidence']}")


if __name__ == "__main__":
    _main()
