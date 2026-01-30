# Complete Steps to Run the Bilingual OCR Code

**Windows · PowerShell · Project root: `c:\Meena\Tabeeb\OCR`**

---

## 1. Open terminal and go to project

```powershell
cd c:\Meena\Tabeeb\OCR
```

---

## 2. Activate the virtual environment

```powershell
.\ocr_env\Scripts\Activate.ps1
```

You should see `(ocr_env)` in the prompt.  
If you get an execution policy error, run once (as Administrator if needed):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then run the activate command again.

---

## 3. (Optional) Create sample inputs folder

```powershell
mkdir sample_inputs -ErrorAction SilentlyContinue
```

Put at least one image in `sample_inputs/`:

- **Printed Urdu:** e.g. `urdu_printed_1.jpg` (scanned or photo of printed Urdu text)
- **English:** e.g. `english_handwritten_1.jpg` (handwritten or printed English)

If you use different names, replace the paths in the commands below.

---

## 4. Pre-flight check (must not be BROKEN)

```powershell
python verify_bilingual_ocr.py
```

- **READY** or **WARNING** → continue to step 5.
- **BROKEN** → stop. Fix the issues (e.g. Visual C++ Redistributable for PyTorch, Tesseract in PATH, UTRNet repo and weights). Then run this again until you get READY or WARNING.

---

## 5. Run preprocessing (single image)

```powershell
python preprocess.py sample_inputs/urdu_printed_1.jpg
```

- A window shows original, grayscale, thresholded, deskewed. Close the window to finish.
- You should see `urdu_printed_1_preprocessed.png` and `urdu_printed_1_preprocessed_gray.png` in `sample_inputs/`.

---

## 6. Run YOLOv8 text detection (crop regions)

```powershell
python text_detection_yolov8.py sample_inputs/urdu_printed_1.jpg outputs/test_crops/
```

- You should see `outputs/test_crops/0.png`, `1.png`, … (first run may download `yolov8n.pt`).

---

## 7. Run UTRNet recognition on crops (Urdu)

```powershell
python urdu_recognition_utrnet.py outputs/test_crops/
```

- You should see Urdu text printed for each crop (e.g. `0.png: …`, `1.png: …`).

---

## 8. Run TrOCR on one image (English)

```powershell
python english_ocr_pipeline.py sample_inputs/english_handwritten_1.jpg
```

- You should see English text printed (first run may download TrOCR model).

---

## 9. End-to-end Urdu OCR (one image → one text file)

```powershell
python urdu_ocr.py --input sample_inputs/urdu_printed_1.jpg --output outputs/urdu_result.txt
```

- You should see `outputs/urdu_result.txt` with Urdu text and a message like “Output written to: …”.

---

## 10. End-to-end bilingual OCR (folder → text + JSON + debug)

```powershell
python bilingual_ocr.py --input sample_inputs/ --output outputs/bilingual_result.txt --lang auto --save-json --debug-images
```

- If you have only one image, use that file instead of the folder:

```powershell
python bilingual_ocr.py --input sample_inputs/urdu_printed_1.jpg --output outputs/bilingual_result.txt --lang auto --save-json --debug-images
```

You should see:

- `outputs/bilingual_result.txt` — combined text with `[urdu]` / `[english]` per region
- `outputs/bilingual_result.json` — structured `{page, regions: [{language, text}]}`
- `outputs/debug/page_1/preprocessed_binary.png`, `preprocessed_gray.png`, `crops/0.png`, …

---

## 11. Run evaluation on bilingual JSON

```powershell
python evaluation.py --input outputs/bilingual_result.json --output outputs/evaluation.json
```

- You should see a short report in the console and `outputs/evaluation.json` created.

---

## 12. Other useful commands

**Urdu-only on a PDF:**

```powershell
python urdu_ocr.py --input path\to\document.pdf --output outputs/urdu_pdf_result.txt
```

**Bilingual on a PDF:**

```powershell
python bilingual_ocr.py --input path\to\document.pdf --output outputs/bilingual_pdf.txt --lang auto --save-json
```

**Force Urdu only (no script detection):**

```powershell
python bilingual_ocr.py --input sample_inputs/ --output outputs/urdu_only.txt --lang urdu
```

**Force English only:**

```powershell
python bilingual_ocr.py --input sample_inputs/ --output outputs/english_only.txt --lang english
```

**Quiet mode (less console output):**

```powershell
python bilingual_ocr.py --input sample_inputs/ --output outputs/result.txt -q
```

---

## Quick copy-paste sequence (after activation)

After steps 1–2 and with images in `sample_inputs/`:

```powershell
python verify_bilingual_ocr.py
python preprocess.py sample_inputs/urdu_printed_1.jpg
python text_detection_yolov8.py sample_inputs/urdu_printed_1.jpg outputs/test_crops/
python urdu_recognition_utrnet.py outputs/test_crops/
python english_ocr_pipeline.py sample_inputs/english_handwritten_1.jpg
python urdu_ocr.py --input sample_inputs/urdu_printed_1.jpg --output outputs/urdu_result.txt
python bilingual_ocr.py --input sample_inputs/ --output outputs/bilingual_result.txt --lang auto --save-json --debug-images
python evaluation.py --input outputs/bilingual_result.json --output outputs/evaluation.json
```

Stop at any step that errors; fix the cause (path, PyTorch, Tesseract, UTRNet) then re-run from that step.
