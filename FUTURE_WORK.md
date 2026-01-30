# Future Work

Suggested extensions beyond the current bilingual OCR system. Not implemented; no new dependencies or model code added in the current codebase.

---

## Word-Level Mixed-Language OCR

- **Current:** Script detection and routing are at **region/crop** level (one language per crop).
- **Future:**  
  - Word-level or sub-region script detection.  
  - Mixed Urdu/English in the same line or block, with per-word or per-phrase routing to UTRNet vs TrOCR.  
  - May require a small classifier or stronger heuristic on top of detection + recognition.

---

## Fine-Tuned Urdu Handwriting Model

- **Current:** UTRNet-Large is for **printed** Urdu only; handwritten Urdu is unsupported (see LIMITATIONS.md).
- **Future:**  
  - Fine-tune UTRNet (or a similar encoder–decoder) on handwritten Urdu data.  
  - Or integrate a dedicated handwritten Urdu model (e.g. from literature or a new dataset).  
  - Would need handwritten Urdu datasets and training/evaluation pipelines.

---

## Learned Layout Detection

- **Current:**  
  - Text regions from **YOLOv8** (pretrained) or **OpenCV** morphology + contours.  
  - No Detectron2; no learned layout/reading-order model.
- **Future:**  
  - Integrate a **layout detection** model (e.g. Detectron2 on Linux/WSL, or a Windows-friendly alternative) for:  
    - Blocks, columns, tables, figures.  
    - Reading order and hierarchy.  
  - Use layout to order and group regions before script detection and recognition.  
  - Optionally use **LayoutParser** with a backend that runs on Windows (e.g. EfficientDet, PaddleDetection) instead of Detectron2.

---

## Other Directions

- **Accuracy evaluation:** Ground-truth transcriptions and metrics (CER/WER, etc.) for Urdu and English.  
- **Confidence scores:** Per-region or per-word confidence from models or heuristics.  
- **RTL ordering:** Full RTL line grouping using box coordinates (e.g. in `post_process`) when boxes are available in the pipeline.  
- **End-to-end training:** Joint or multi-task models for detection + script + recognition (would require datasets and training infrastructure).

These items are listed for examiners and future developers; the current deliverable remains pretrained-only, with no training or new ML models.
