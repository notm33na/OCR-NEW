# OCR Environment Setup Notes (Windows, CPU-only)

## 1. Python version

- **Required:** Python 3.9 or newer.
- **Check:** `python --version`
- **If missing:** Install from [python.org](https://www.python.org/downloads/) (Windows installer). During setup, check "Add Python to PATH".

---

## 2. Virtual environment

Create and use the virtual environment:

```powershell
cd "c:\Meena\Tabeeb\OCR"
python -m venv ocr_env
```

---

## 3. Activate the environment (PowerShell)

```powershell
.\ocr_env\Scripts\Activate.ps1
```

If you see an execution policy error, run once (as Administrator if needed):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again. Your prompt should show `(ocr_env)`.

---

## 4. Install dependencies (CPU-only)

Activate `ocr_env`, then:

```powershell
# PyTorch CPU (use this index)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Rest of the stack
pip install transformers opencv-python pillow numpy matplotlib scikit-image pytesseract layoutparser
```

Or use the pinned `requirements.txt` (install PyTorch from the CPU index first, then `pip install -r requirements.txt`).

---

## 5. Detectron2 – deferred (not installed)

- **Layout detection** in LayoutParser can use Detectron2, but **Detectron2 is not officially supported on Windows** (no prebuilt wheels; building from source is complex and often fails).
- **Current setup:** Only base **layoutparser** is installed (data structures, visualization, load/export). No Detectron2.
- **Later options:** Use LayoutParser with other backends, e.g. `layoutparser[effdet]` or `layoutparser[paddledetection]`, or add Detectron2 later if you use WSL or a Linux build.

---

## 6. Tesseract OCR for Windows

- **Purpose:** `pytesseract` is a Python wrapper; the actual engine is Tesseract. You must install Tesseract on Windows and have it on PATH.
- **Installer (official):**  
  https://github.com/UB-Mannheim/tesseract/wiki  
  (Windows installer from UB Mannheim – recommended.)
- **During install:** Optionally add "Tesseract to PATH" or note the install path (e.g. `C:\Program Files\Tesseract-OCR`).
- **Add to PATH (if not done by installer):**
  1. Win + R → `sysdm.cpl` → Advanced → Environment Variables.
  2. Under "System" or "User", select "Path" → Edit → New.
  3. Add the folder that contains `tesseract.exe`, e.g. `C:\Program Files\Tesseract-OCR`.
  4. OK out and **restart the terminal** (or PowerShell).
- **Verify:**

```powershell
tesseract --version
```

You should see the Tesseract version and supported languages.

---

## 7. Verify Python environment

With `ocr_env` activated and from the project root:

```powershell
python verify_env.py
```

You should see `[OK]` for: torch, transformers, cv2 (opencv-python), layoutparser.

---

## Constraints (this setup)

- **Windows only** – no Ubuntu, no WSL, no Docker.
- **CPU only** – no CUDA/GPU.
- **No OCR logic** – environment and dependencies only; TrOCR/UTRNet and pipeline code are separate.

---

## Troubleshooting

### Torch: "DLL initialization routine failed" (c10.dll)

If `python verify_env.py` fails on `torch` with a DLL error:

1. **Install Microsoft Visual C++ Redistributable** (latest for your architecture):  
   https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist  
   Install both "x64" and "x86" if unsure.
2. Restart the terminal and run `python verify_env.py` again.
3. If you use Anaconda, try running the script from a **new** PowerShell window (not inside Conda) so only the venv is active.
