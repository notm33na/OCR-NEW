# OCR System Limitations

This document describes known limitations of the bilingual OCR system. No ground-truth accuracy or dataset-based evaluation is included.

---

## Handwritten Urdu

- **Printed Urdu only.**  
  The pipeline uses **UTRNet-Large**, which is trained for **printed** Urdu text. Handwritten Urdu is **not** supported; quality on handwritten Urdu will be poor or unusable.
- **Line / region images.**  
  UTRNet expects cropped line/region images (from YOLOv8 or OpenCV detection). Full-page handwritten Urdu with complex layout is out of scope.

---

## No Detectron2

- **Layout detection** does **not** use Detectron2.  
  Detectron2 is not officially supported on Windows and is not installed. Layout and region grouping use:
  - **YOLOv8** (pretrained) for text detection, or  
  - **OpenCV** morphology + contours (`text_region_detection.py`) as fallback.  
- No learned layout model; no Detectron2-based table/figure detection.

---

## No Training or Datasets

- **Pretrained models only.**  
  No training code, no fine-tuning, no custom datasets. The system uses:
  - YOLOv8 (general or provided checkpoint),
  - UTRNet-Large (authors’ checkpoint),
  - TrOCR handwritten (HuggingFace).
- **No accuracy metrics** that require ground truth (e.g. character/word accuracy).  
  Evaluation is limited to coverage, script distribution, runtime, and failure counts (`evaluation.py`).

---

## Heuristic Script Detection

- **Script detection** (`script_detection.py`) is **heuristic**, not learned.  
  It uses:
  - Lightweight pytesseract OCR,  
  - Presence of **Arabic Unicode range (U+0600–U+06FF)** → `urdu`, else `english`.  
- **Limitations:**
  - Mixed script within one crop is treated as one language (first script “wins”).
  - No word-level or sub-region script detection.
  - Tesseract quality and language settings affect the heuristic; no confidence score beyond `"heuristic"`.
  - Arabic-only documents are classified as Urdu (script, not language).

---

## Other Constraints

- **Windows, CPU-only by default.**  
  CUDA is optional; the codebase is written to run on Windows without WSL/Ubuntu. No GPU-specific logic is required.
- **No cloud services.**  
  All processing is local (models, Tesseract, UTRNet subprocess).
- **No GUI.**  
  CLI and file-based input/output only.
- **UTRNet subprocess.**  
  Urdu recognition runs the UTRNet repo’s `read.py` via subprocess; Python 3.9 or lower is recommended for that repo (see SETUP_NOTES).

These limitations are intentional for the current FYP scope and can be extended in future work (see FUTURE_WORK.md).
