# Bilingual OCR Verification Script

## Purpose

`verify_bilingual_ocr.py` checks that the Windows environment is correctly set up for:

- **Printed Urdu OCR:** YOLOv8 (text detection) + UTRNet-Large (recognition)
- **Handwritten English OCR:** Microsoft TrOCR
- **Preprocessing:** OpenCV
- **Optional:** Tesseract (pytesseract)

No dataset training; local execution only; CPU or CUDA if available.

## How to run

From the project root, with your virtual environment activated:

```powershell
cd "c:\Meena\Tabeeb\OCR"
.\ocr_env\Scripts\Activate.ps1
python verify_bilingual_ocr.py
```

## What it checks (summary)

| # | Check | What is verified |
|---|--------|-------------------|
| 1 | Python | Version 3.8+ (TrOCR, YOLOv8); warns if >3.9 (UTRNet recommends ≤3.9) |
| 2 | Environment | venv / conda / system; name and path |
| 3 | PyTorch | Import, version, CUDA availability and version |
| 4 | Core deps | transformers, ultralytics, cv2, PIL, numpy, matplotlib, pytesseract |
| 5 | Tesseract | Binary in PATH, `tesseract --version` |
| 6 | UTRNet weights | `saved_models/UTRNet-Large/best_norm_ED.pth` exists |
| 7 | YOLOv8 sanity | Load yolov8n.pt, run on dummy image (no training) |
| 8 | TrOCR sanity | Load microsoft/trocr-base-handwritten, run on dummy image |
| 9 | UTRNet import | Repo present; model.py, utils.py, read.py importable |
| 10 | Final report | READY / WARNING / BROKEN and fix commands |

## Expected output when READY

- **1.** Python 3.8+; UTRNet warning only if Python > 3.9  
- **2.** Environment type (venv/conda) and path  
- **3.** torch version; CUDA available or not; no critical errors  
- **4.** All core deps: `[PASS]`  
- **5.** Tesseract found and `tesseract --version` runs  
- **6.** `[PASS] Found: ...\saved_models\UTRNet-Large\best_norm_ED.pth`  
- **7.** `[PASS] YOLOv8 loaded and inference on dummy image succeeded.`  
- **8.** `[PASS] TrOCR loaded and text generation on dummy image succeeded...`  
- **9.** `[PASS] UTRNet repository valid; model, utils, read imported successfully.`  
- **10.** Status: **READY** – “Environment is correctly configured for bilingual OCR.”

## When BROKEN

The script prints:

- What is wrong  
- Why it matters  
- Exact fix (Windows): commands and steps (e.g. install Tesseract, add to PATH, clone UTRNet, download weights).

## Dependencies to install first (if missing)

```powershell
# PyTorch (CPU or CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or with CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core
pip install transformers ultralytics opencv-python pillow numpy matplotlib pytesseract
```

## UTRNet setup (no training)

1. Clone repo into project root:
   ```powershell
   git clone https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition.git
   ```
2. Download UTRNet-Large from [Google Drive](https://drive.google.com/file/d/1xXG7vsSePBw4vtapIEdPWEZ-qrbR9Q9K/view?usp=sharing).
3. Create folder `saved_models\UTRNet-Large\` and place the file as `best_norm_ED.pth`.

## Tesseract (Windows)

- Installer: https://github.com/UB-Mannheim/tesseract/wiki  
- Add the folder containing `tesseract.exe` to system PATH.  
- Verify: `tesseract --version`

## Console output expectations

- **First run with TrOCR:** “Loading microsoft/trocr-base-handwritten; first run may download from Hugging Face...” — normal.
- **First run with YOLOv8:** “Loading pretrained YOLOv8 nano (yolov8n.pt); first run may download...” — normal.
- Sections 1–10 are printed in order; each has a header and PASS/FAIL/SKIP lines. First run with TrOCR/YOLOv8 may download models (announced). No silent downloads. Final report shows READY/WARNING/BROKEN and Fix (Windows) for each BROKEN item.

## Exit code

- `0`: All critical checks passed (READY or WARNING only).  
- `1`: At least one BROKEN item; fix and re-run.
