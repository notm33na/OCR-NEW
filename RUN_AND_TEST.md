# Run and Test Guide — Bilingual OCR (Windows)

First-time execution and testing plan. **Do not proceed if Step A reports BROKEN.**

---

## Prerequisites

- Virtual environment **activated**: `.\ocr_env\Scripts\Activate.ps1`
- Working directory: project root (e.g. `c:\Meena\Tabeeb\OCR`)
- **Sample inputs:** Create `sample_inputs/` and add:
  - At least one **printed Urdu** image (e.g. `urdu_printed_1.jpg`) — scanned or photo of printed Urdu text
  - At least one **handwritten English** image (e.g. `english_handwritten_1.jpg`) — handwritten English line or paragraph  
  If you don’t have these yet, use any `.jpg`/`.png` you have and adjust paths below.

---

## STEP A — Pre-flight checks (NO OCR YET)

### 1. Run verification

```powershell
python verify_bilingual_ocr.py
```

### 2. How to interpret the final status

| Status   | Meaning |
|----------|--------|
| **READY** | All critical checks passed. Safe to run OCR. |
| **WARNING** | Some non-critical issues (e.g. Python >3.9 for UTRNet, CUDA missing but GPU present). You can still test; watch for UTRNet/TrOCR errors. |
| **BROKEN** | At least one critical failure (e.g. PyTorch import, Tesseract missing, UTRNet weights/repo missing, YOLOv8/TrOCR import or sanity failed). **Do not proceed** until these are fixed. |

### 3. If WARNING appears

- **Python >3.9:** UTRNet repo recommends 3.7–3.9. If UTRNet fails later, consider a separate venv with Python 3.9.
- **CUDA not available:** Normal for CPU-only; OCR will run on CPU.
- **Tesseract not in PATH:** Script detection in bilingual mode may misroute; install Tesseract and add to PATH (see SETUP_NOTES.md).

**WARNING does not block testing** unless you see “BROKEN” as well.

### 4. If BROKEN

- Fix the issues listed in the report (e.g. install Visual C++ Redistributable for PyTorch DLL, install Tesseract, clone UTRNet and add `saved_models/UTRNet-Large/best_norm_ED.pth`).
- Re-run `python verify_bilingual_ocr.py` until status is **READY** or **WARNING** (no BROKEN).

**Do NOT proceed to Step B if status is BROKEN.**

---

## STEP B — Unit sanity tests (isolated modules)

Use your own image paths if you don’t have `sample_inputs/`. Replace `sample_inputs/urdu_printed_1.jpg` etc. with a real file.

### 1. Preprocessing

```powershell
python preprocess.py sample_inputs/urdu_printed_1.jpg
```

- **Confirm:** A window shows original, grayscale, thresholded, deskewed. Closing the window saves the image.
- **Confirm:** In the same folder as the input, you see `<stem>_preprocessed.png` and `<stem>_gray.png`.
- **Deskew:** Text should look more horizontal, not rotated wrongly. If the page was already straight, rotation should be minimal.

**If it fails:** Check that the path exists and is a valid image (PATH/image issue).

---

### 2. YOLOv8 detection

```powershell
python text_detection_yolov8.py sample_inputs/urdu_printed_1.jpg outputs/test_crops/
```

- **Confirm:** Folder `outputs/test_crops/` contains `0.png`, `1.png`, … (one per detected region).
- **Confirm:** Filenames are numeric and ordered (reading order: top-to-bottom, left-to-right).
- First run may download `yolov8n.pt` once.

**If it fails:** PATH (wrong image path), or model loading (PyTorch/ultralytics). Fix PATH first; then check verify step for PyTorch.

---

### 3. UTRNet recognition

```powershell
python urdu_recognition_utrnet.py outputs/test_crops/
```

- **Confirm:** Urdu text is printed for each crop (e.g. `0.png: …`, `1.png: …`). Imperfect output is acceptable.
- **Confirm:** No crash; script exits normally.

**If it fails:**  
- **Subprocess/UTRNet:** UTRNet repo must be at `UTRNet-High-Resolution-Urdu-Text-Recognition` and `saved_models/UTRNet-Large/best_norm_ED.pth` must exist.  
- **Python version:** If you see encoding or import errors, try Python 3.9 in a separate venv for this step only.

---

### 4. TrOCR English test

```powershell
python english_ocr_pipeline.py sample_inputs/english_handwritten_1.jpg
```

- **Confirm:** English text is printed for the image (handwritten or printed).
- **Confirm:** No crash. First run may download TrOCR model once.

**If it fails:** Model loading (transformers/torch) or image path. Check verify step for transformers/torch.

---

### Minimal fix hints (Step B)

| Symptom           | Likely cause     | Minimal fix |
|-------------------|------------------|-------------|
| File not found    | PATH             | Use correct path; create `sample_inputs/` and add images. |
| PyTorch DLL error | Runtime          | Install Visual C++ Redistributable (see SETUP_NOTES.md). |
| Tesseract not found | Environment    | Install Tesseract, add to PATH (script_detection / bilingual). |
| UTRNet fails      | Model/repo PATH  | Clone UTRNet repo, place `best_norm_ED.pth` in `saved_models/UTRNet-Large/`. |
| No crops created  | YOLOv8 / image   | Try another image with clear text; check confidence (e.g. `--conf 0.2` in text_detection_yolov8). |

---

## STEP C — End-to-end printed Urdu test

```powershell
python urdu_ocr.py --input sample_inputs/urdu_printed_1.jpg --output outputs/urdu_result.txt
```

**Validate:**

- File **outputs/urdu_result.txt** exists.
- File contains Urdu text (readable even if imperfect).
- No crash; script prints “Output written to: …”.
- Logs are reasonable (e.g. detection count, “Recognized N line(s)”).

**If it fails:** Same as Step B.2 and B.3 (detection + UTRNet). Fix those first.

---

## STEP D — End-to-end bilingual test (core test)

```powershell
python bilingual_ocr.py --input sample_inputs/ --output outputs/bilingual_result.txt --lang auto --save-json --debug-images
```

- If you have only one image, run on that file instead of folder:  
  `--input sample_inputs/urdu_printed_1.jpg`
- `sample_inputs/` must contain at least one jpg/png so the folder run does something.

**Validate:**

1. **outputs/bilingual_result.txt** exists and contains text with `[urdu]` / `[english]` (or `[unknown]`) prefixes per region.
2. **outputs/bilingual_result.json** exists (list of `{page, regions: [{language, text}]}`).
3. **outputs/debug/page_1/** (and page_2, … if multiple pages) contains:
   - `preprocessed_binary.png`
   - `preprocessed_gray.png`
   - `crops/0.png`, `1.png`, …

**Routing:**

- Regions with Urdu script (Arabic Unicode) → UTRNet; log shows “Routing: Urdu X, English Y” with X ≥ 1 for Urdu.
- Regions without → TrOCR (English). Y ≥ 1 for English if you have English content.

**How to visually verify using debug images:**

- **preprocessed_gray.png:** Deskewed page; text should be roughly horizontal and readable.
- **preprocessed_binary.png:** Black-and-white text; no major loss of text regions.
- **crops/0.png, 1.png, …:** Each crop should be a single text block/line. Order = reading order (top-to-bottom, left-to-right).
- Compare `bilingual_result.txt`: line order should match crop order; Urdu lines should correspond to crops that look like Urdu script.

**If it fails:** Check Step A and B first. Typical causes: missing Tesseract (script detection), UTRNet/TrOCR path or model, or empty input folder.

---

## STEP E — Evaluation test

```powershell
python evaluation.py --input outputs/bilingual_result.json --output outputs/evaluation.json
```

**Validate:**

- **outputs/evaluation.json** exists.
- **Console:** OCR coverage (total_regions, regions_with_text, empty_regions) matches what you expect from the run (e.g. total_regions ≥ 1 if you had detections).
- **Script distribution:** count_urdu / count_english match the mix of content you used.
- **Runtime:** total_time_seconds and avg_time_per_page_seconds are present; if you didn’t pass `--total-time`, they may be null — that’s OK.
- **Failure metrics:** regions_failed_ocr and pages_skipped are consistent with your run (e.g. no skip if everything ran).

**If it fails:** Input must be the JSON produced by bilingual_ocr (list of `{page, regions}`). Check that `--input` points to that file.

---

## STEP F — Final go / no-go checklist

Use this after a full run (A → E) to summarize status.

### WORKING

- [ ] **Pre-flight:** `verify_bilingual_ocr.py` finishes with READY or WARNING (no BROKEN).
- [ ] **Preprocess:** `preprocess.py` runs, saves preprocessed image, deskew looks correct.
- [ ] **YOLOv8:** `text_detection_yolov8.py` produces crops in `outputs/test_crops/` in order.
- [ ] **UTRNet:** `urdu_recognition_utrnet.py` prints Urdu text for those crops.
- [ ] **TrOCR:** `english_ocr_pipeline.py` prints English text for a handwritten/English image.
- [ ] **Urdu E2E:** `urdu_ocr.py` produces `outputs/urdu_result.txt` with Urdu content.
- [ ] **Bilingual E2E:** `bilingual_ocr.py` produces `bilingual_result.txt`, `bilingual_result.json`, and `outputs/debug/page_*/` with preprocessed + crops.
- [ ] **Evaluation:** `evaluation.py` produces `evaluation.json` and console report; numbers match the run.

### ACCEPTABLE FOR FYP

- [ ] Script detection is heuristic (pytesseract + Unicode); occasional misroute (e.g. Urdu ↔ English) is acceptable.
- [ ] OCR quality is imperfect (e.g. some wrong characters); no ground-truth accuracy required.
- [ ] Runtime metrics in evaluation may be null if `--total-time` was not passed.
- [ ] WARNING from verify (e.g. Python >3.9, no CUDA) is acceptable as long as no BROKEN.

### KNOWN LIMITATION (but OK)

- [ ] **Printed Urdu only;** handwritten Urdu is out of scope.
- [ ] **No Detectron2;** layout uses YOLOv8 + OpenCV only.
- [ ] **No training/datasets;** pretrained models only.
- [ ] **Tesseract optional but recommended** for bilingual script detection; if missing, use `--lang urdu` or `--lang english` to force one language.

---

## One-command summary (after sample_inputs exist)

Run in order; stop if any step fails:

```powershell
.\ocr_env\Scripts\Activate.ps1
cd c:\Meena\Tabeeb\OCR

python verify_bilingual_ocr.py
# → Must not be BROKEN

python preprocess.py sample_inputs/urdu_printed_1.jpg
python text_detection_yolov8.py sample_inputs/urdu_printed_1.jpg outputs/test_crops/
python urdu_recognition_utrnet.py outputs/test_crops/
python english_ocr_pipeline.py sample_inputs/english_handwritten_1.jpg

python urdu_ocr.py --input sample_inputs/urdu_printed_1.jpg --output outputs/urdu_result.txt
python bilingual_ocr.py --input sample_inputs/ --output outputs/bilingual_result.txt --lang auto --save-json --debug-images
python evaluation.py --input outputs/bilingual_result.json --output outputs/evaluation.json
```

Then complete **STEP F** checklist above for a clear go/no-go and FYP acceptance.
