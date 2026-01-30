# OCR System Architecture

## System Diagram

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                     INPUT LAYER                          │
                    │  (PDF / single image / folder of images)                │
                    └─────────────────────────┬───────────────────────────────┘
                                              │
                    ┌─────────────────────────▼───────────────────────────────┐
                    │  bilingual_ocr.py / urdu_ocr.py / english_ocr_pipeline   │
                    │  (orchestration, PDF→images via pypdfium2)              │
                    └─────────────────────────┬───────────────────────────────┘
                                              │
                    ┌─────────────────────────▼───────────────────────────────┐
                    │  preprocess.py                                           │
                    │  load → grayscale → denoise → adaptive threshold → deskew│
                    └─────────────────────────┬───────────────────────────────┘
                                              │
                    ┌─────────────────────────▼───────────────────────────────┐
                    │  TEXT DETECTION                                         │
                    │  YOLOv8 (text_detection_yolov8) [preferred]             │
                    │  OpenCV (text_region_detection) [fallback]              │
                    │  → bounding boxes / crops in reading order              │
                    └─────────────────────────┬───────────────────────────────┘
                                              │
                    ┌─────────────────────────▼───────────────────────────────┐
                    │  script_detection.py (bilingual only)                     │
                    │  pytesseract + Unicode (U+0600–U+06FF) → urdu | english  │
                    └─────────────────────────┬───────────────────────────────┘
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
              │                                                                  │
    ┌─────────▼─────────┐                                            ┌─────────▼─────────┐
    │  Urdu path        │                                            │  English path     │
    │  urdu_recognition_│                                            │  english_ocr_    │
    │  utrnet (UTRNet-  │                                            │  pipeline        │
    │  Large, read.py)  │                                            │  (TrOCR hand-    │
    │  → post_process   │                                            │   written)       │
    │  _clean_urdu_text │                                            │                  │
    └─────────┬─────────┘                                            └─────────┬─────────┘
              │                                                                  │
              └───────────────────────────────┬─────────────────────────────────┘
                                              │
                    ┌─────────────────────────▼───────────────────────────────┐
                    │  OUTPUT                                                 │
                    │  UTF-8 text file, optional JSON (page, regions)         │
                    │  evaluation.py (metrics from JSON, no ground truth)     │
                    └────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Input**  
   PDF (pypdfium2 → page images), single image, or folder of jpg/png.

2. **Preprocessing**  
   `preprocess.py`: load → grayscale → median denoise → adaptive threshold → deskew.  
   Output: binary + deskewed grayscale (used by OpenCV detector or TrOCR/UTRNet as needed).

3. **Text detection**  
   - **Primary:** YOLOv8 (`text_detection_yolov8.detect_and_crop_text`) on original image → crops `0.png`, `1.png`, … in reading order.  
   - **Fallback:** OpenCV `text_region_detection.detect_text_blocks` on binary → crop by boxes.

4. **Script detection (bilingual only)**  
   `script_detection.detect_script(crop)`: pytesseract + Arabic Unicode range → `urdu` or `english`.

5. **Recognition**  
   - **Urdu:** UTRNet-Large via subprocess (`read.py`, HRNet + DBiLSTM + CTC).  
   - **English:** TrOCR handwritten (HuggingFace), with optional preprocessing.

6. **Post-processing**  
   Urdu text only: `post_process._clean_urdu_text` (dedupe, remove non-Urdu ASCII).

7. **Output**  
   One UTF-8 text file; optional JSON `{page, regions: [{language, text}]}`.  
   Evaluation: `evaluation.py` reads that JSON and reports coverage, script distribution, runtime, failures (no ground truth).

## Module Roles

| Module | Role |
|--------|------|
| `bilingual_ocr.py` | Unified orchestrator: input → detect → script → route → recognize → output. |
| `urdu_ocr.py` | Urdu-only: PDF/folder → YOLOv8 + UTRNet → one text file. |
| `urdu_ocr_pipeline.py` | Single-image Urdu: YOLOv8 + UTRNet, JSON output. |
| `english_ocr_pipeline.py` | English-only: preprocess + TrOCR (single image or folder). |
| `preprocess.py` | OpenCV preprocessing (no ML). |
| `text_detection_yolov8.py` | YOLOv8 detection + crop (pretrained). |
| `text_region_detection.py` | OpenCV morphology + contours (fallback). |
| `script_detection.py` | Heuristic script: pytesseract + Unicode. |
| `urdu_recognition_utrnet.py` | UTRNet-Large via read.py. |
| `post_process.py` | Urdu RTL/clean (boxes + text or text-only). |
| `evaluation.py` | Metrics from JSON (coverage, script, runtime, failures). |

## Dependencies (no Detectron2)

- **Detection:** ultralytics (YOLOv8), OpenCV.  
- **Recognition:** transformers (TrOCR), UTRNet repo (subprocess).  
- **PDF:** pypdfium2.  
- **Script:** pytesseract + Tesseract binary.  
- **Preprocessing:** OpenCV, NumPy.  
- No Detectron2; no training; pretrained models only.
