"""
Bilingual OCR environment verification (Windows, local only).
Checks: TrOCR (English handwritten), YOLOv8 (text detection), UTRNet-Large (Urdu),
OpenCV, Tesseract. NO training, NO silent large downloads.

Run: python verify_bilingual_ocr.py
     (from project root, with venv/conda activated)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
UTRNET_WEIGHT_PATH = PROJECT_ROOT / "saved_models" / "UTRNet-Large" / "best_norm_ED.pth"
# Possible UTRNet repo locations (clone from GitHub into project)
UTRNET_REPO_NAMES = [
    "UTRNet-High-Resolution-Urdu-Text-Recognition",
    "UTRNet",
]
UTRNET_DOWNLOAD_URL = (
    "https://drive.google.com/file/d/1xXG7vsSePBw4vtapIEdPWEZ-qrbR9Q9K/view?usp=sharing"
)
UTRNET_REPO_URL = "https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition"

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
class Result:
    READY = "READY"
    WARNING = "WARNING"
    BROKEN = "BROKEN"

issues: list[tuple[str, str, str]] = []  # (severity, what, fix)

def add_issue(severity: str, what: str, fix: str = ""):
    issues.append((severity, what, fix))

def _section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

def _subsection(title: str) -> None:
    print()
    print("- " + title)

# ---------------------------------------------------------------------------
# 1. Python environment check
# ---------------------------------------------------------------------------
def check_python() -> None:
    _section("1. Python Environment Check")
    ver = sys.version_info
    version_str = f"{ver.major}.{ver.minor}.{ver.micro}"
    print(f"Current Python: {version_str} ({sys.executable})")

    # TrOCR: >=3.8
    trocr_ok = ver >= (3, 8)
    print(f"  TrOCR (>=3.8): {'OK' if trocr_ok else 'FAIL'}")

    # UTRNet: repo recommends Python 3.7; user asked <=3.9 recommended
    utrnet_recommended = ver <= (3, 9)
    if not utrnet_recommended:
        print(f"  UTRNet (<=3.9 recommended): WARNING - you have {version_str}")
        add_issue(
            Result.WARNING,
            f"Python {version_str} may be incompatible with UTRNet (repo recommends 3.7, <=3.9 recommended).",
            "Use a separate venv with Python 3.9 if UTRNet fails: py -3.9 -m venv ocr_utrnet",
        )
    else:
        print(f"  UTRNet (<=3.9 recommended): OK")

    # Ultralytics YOLOv8: 3.8+
    yolo_ok = ver >= (3, 8)
    print(f"  Ultralytics YOLOv8 (3.8+): {'OK' if yolo_ok else 'FAIL'}")
    if not yolo_ok:
        add_issue(
            Result.BROKEN,
            f"Python {version_str} is below 3.8; YOLOv8 requires 3.8+.",
            "Install Python 3.8+ from https://www.python.org/downloads/ (Windows).",
        )

# ---------------------------------------------------------------------------
# 2. Virtual / Conda environment detection
# ---------------------------------------------------------------------------
def check_environment() -> None:
    _section("2. Virtual / Conda Environment Detection")
    in_isolated = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    # Prefer executable path: conda Python paths usually contain "conda"
    exe_lower = sys.executable.lower()
    if "conda" in exe_lower and in_isolated:
        print(f"Environment type: conda")
        print(f"Environment name: {os.environ.get('CONDA_DEFAULT_ENV', '(unknown)')}")
    elif in_isolated:
        print(f"Environment type: venv")
        env_name = Path(sys.prefix).name or "(venv)"
        print(f"Environment name: {env_name}")
    else:
        print(f"Environment type: system Python")
        print(f"Environment name: (none)")
        add_issue(
            Result.WARNING,
            "Running on system Python. Isolated venv/conda is recommended.",
            'python -m venv ocr_env  then  .\\ocr_env\\Scripts\\Activate.ps1',
        )
    base = getattr(sys, "base_prefix", getattr(sys, "real_prefix", sys.prefix))
    print(f"Prefix (executable base): {base}")
    print(f"Python executable: {sys.executable}")

# ---------------------------------------------------------------------------
# 3. PyTorch verification
# ---------------------------------------------------------------------------
def check_pytorch() -> None:
    _section("3. PyTorch Verification")
    try:
        import torch
        print(f"torch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Device: {torch.cuda.get_device_name(0)}")
        else:
            # Check if we're on Windows and nvidia-smi exists (GPU might exist but CUDA not used)
            nv_path = shutil.which("nvidia-smi")
            if nv_path:
                add_issue(
                    Result.WARNING,
                    "CUDA is not available in PyTorch but nvidia-smi was found. GPU may be present.",
                    "Install CUDA toolkit and reinstall PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
                )
        # UTRNet was developed on PyTorch 1.9.1; newer may work but warn
        try:
            major, minor = map(int, torch.__version__.split("+")[0].split(".")[:2])
            if (major, minor) > (1, 9):
                add_issue(
                    Result.WARNING,
                    f"UTRNet was developed on PyTorch 1.9.1; current {torch.__version__} may have API differences.",
                    "If UTRNet fails, try a venv with: pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html",
                )
        except (ValueError, IndexError):
            pass
        return
    except Exception as e:
        print(f"FAIL: Could not import torch: {e}")
        add_issue(
            Result.BROKEN,
            f"PyTorch import failed: {e}",
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        )

# ---------------------------------------------------------------------------
# 4. Core dependency verification
# ---------------------------------------------------------------------------
def check_core_deps() -> None:
    _section("4. Core Dependency Verification")
    deps = [
        ("transformers (TrOCR)", "transformers"),
        ("ultralytics (YOLOv8)", "ultralytics"),
        ("cv2 (OpenCV)", "cv2"),
        ("PIL", "PIL"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("pytesseract", "pytesseract"),
    ]
    for label, mod in deps:
        try:
            __import__(mod)
            print(f"  [PASS] {label}")
        except Exception as e:
            print(f"  [FAIL] {label}: {e}")
            fix = "pip install " + {
                "transformers (TrOCR)": "transformers",
                "ultralytics (YOLOv8)": "ultralytics",
                "cv2 (OpenCV)": "opencv-python",
                "PIL": "pillow",
                "numpy": "numpy",
                "matplotlib": "matplotlib",
                "pytesseract": "pytesseract",
            }.get(label, mod)
            add_issue(Result.BROKEN, f"Missing or broken: {label}", fix)

# ---------------------------------------------------------------------------
# 5. External binary: Tesseract
# ---------------------------------------------------------------------------
def check_tesseract() -> None:
    _section("5. External Binary Check (Tesseract)")
    tesseract_exe = shutil.which("tesseract")
    if not tesseract_exe:
        print("  [FAIL] Tesseract not found in PATH.")
        add_issue(
            Result.BROKEN,
            "Tesseract OCR is not installed or not on PATH.",
            "1) Download Windows installer: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "2) Run installer; optionally add Tesseract to PATH during setup.\n"
            "3) If not added: Win+R -> sysdm.cpl -> Advanced -> Environment Variables -> Path -> New -> add folder containing tesseract.exe (e.g. C:\\Program Files\\Tesseract-OCR).\n"
            "4) Restart terminal and run: tesseract --version",
        )
        return
    print(f"  Found: {tesseract_exe}")
    try:
        out = subprocess.run(
            [tesseract_exe, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0,
        )
        if out.returncode == 0:
            print(f"  [PASS] tesseract --version:")
            for line in (out.stdout or out.stderr or "").strip().split("\n")[:3]:
                print(f"    {line}")
        else:
            print(f"  [WARN] tesseract exited with {out.returncode}")
    except Exception as e:
        print(f"  [WARN] Could not run tesseract --version: {e}")

# ---------------------------------------------------------------------------
# 6. Model availability: UTRNet-Large weights
# ---------------------------------------------------------------------------
def check_utrnet_weights() -> None:
    _section("6. Model Availability Check (UTRNet-Large)")
    if UTRNET_WEIGHT_PATH.is_file():
        print(f"  [PASS] Found: {UTRNET_WEIGHT_PATH}")
        return
    print(f"  [FAIL] Not found: {UTRNET_WEIGHT_PATH}")
    add_issue(
        Result.BROKEN,
        "UTRNet-Large pretrained weights (best_norm_ED.pth) are missing.",
        "1) Download UTRNet-Large from Google Drive: " + UTRNET_DOWNLOAD_URL + "\n"
        "2) Create folder: saved_models\\UTRNet-Large\\\n"
        "3) Place the downloaded file as: saved_models\\UTRNet-Large\\best_norm_ED.pth\n"
        "Do NOT train; use the pretrained checkpoint only.",
    )

# ---------------------------------------------------------------------------
# 7. YOLOv8 sanity test (no training)
# ---------------------------------------------------------------------------
def check_yolov8_sanity() -> None:
    _section("7. YOLOv8 Sanity Test (NO TRAINING)")
    try:
        from ultralytics import YOLO
        import numpy as np
    except Exception as e:
        print(f"  [SKIP] Cannot run YOLOv8 test: {e}")
        add_issue(Result.BROKEN, "YOLOv8 sanity test skipped (ultralytics or numpy missing).", "pip install ultralytics numpy")
        return

    print("  Loading pretrained YOLOv8 nano (yolov8n.pt); first run may download...")
    try:
        model = YOLO("yolov8n.pt")
        # Dummy image: 640x640 RGB
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy, verbose=False)
        print("  [PASS] YOLOv8 loaded and inference on dummy image succeeded.")
    except Exception as e:
        print(f"  [FAIL] YOLOv8 sanity test: {e}")
        add_issue(
            Result.BROKEN,
            f"YOLOv8 inference failed: {e}",
            "pip install ultralytics  and ensure PyTorch is installed.",
        )

# ---------------------------------------------------------------------------
# 8. TrOCR sanity test
# ---------------------------------------------------------------------------
def check_trocr_sanity() -> None:
    _section("8. TrOCR Sanity Test")
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from PIL import Image
    except Exception as e:
        print(f"  [SKIP] Cannot run TrOCR test: {e}")
        add_issue(Result.BROKEN, "TrOCR sanity test skipped (transformers or PIL missing).", "pip install transformers pillow")
        return

    print("  Loading microsoft/trocr-base-handwritten; first run may download from Hugging Face...")
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        # Dummy small image (e.g. 32x128)
        dummy_pil = Image.new("RGB", (128, 32), color=(255, 255, 255))
        pixel_values = processor(images=dummy_pil, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"  [PASS] TrOCR loaded and text generation on dummy image succeeded (sample output: {repr(text[:50])}).")
    except Exception as e:
        print(f"  [FAIL] TrOCR sanity test: {e}")
        add_issue(
            Result.BROKEN,
            f"TrOCR inference failed: {e}",
            "pip install transformers torch pillow  and ensure network access for first-time model download.",
        )

# ---------------------------------------------------------------------------
# 9. UTRNet import check (repo structure)
# ---------------------------------------------------------------------------
def check_utrnet_import() -> None:
    _section("9. UTRNet Import Check")
    utrnet_root = None
    for name in UTRNET_REPO_NAMES:
        d = PROJECT_ROOT / name
        if d.is_dir():
            utrnet_root = d
            break
    if not utrnet_root:
        print(f"  [FAIL] UTRNet repository directory not found.")
        print(f"  Looked for: {[str(PROJECT_ROOT / n) for n in UTRNET_REPO_NAMES]}")
        add_issue(
            Result.BROKEN,
            "UTRNet source code not found. Clone the repo into the project root.",
            f"git clone {UTRNET_REPO_URL}.git\n"
            f"  Then re-run this script. Do NOT train; pretrained weights go in saved_models/UTRNet-Large/.",
        )
        return

    print(f"  Found repo at: {utrnet_root}")
    required_files = ["model.py", "utils.py", "read.py"]
    for f in required_files:
        if not (utrnet_root / f).is_file():
            print(f"  [FAIL] Missing: {f}")
            add_issue(Result.BROKEN, f"UTRNet repo missing {f}.", f"Ensure repo is fully cloned: {UTRNET_REPO_URL}")
            return

    # Import by path so we don't pollute global namespace
    import importlib.util
    loaded = []
    for mod_name in ["model", "utils", "read"]:
        path = utrnet_root / f"{mod_name}.py"
        try:
            spec = importlib.util.spec_from_file_location(f"utrnet_{mod_name}", path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                # Optional: add utrnet_root to path for any internal imports (e.g. modules/)
                sys.path.insert(0, str(utrnet_root))
                try:
                    spec.loader.exec_module(mod)
                finally:
                    sys.path.pop(0)
                loaded.append(mod_name)
        except Exception as e:
            print(f"  [FAIL] Import {mod_name}.py: {e}")
            add_issue(
                Result.BROKEN,
                f"UTRNet {mod_name}.py failed to import: {e}",
                "Check Python version (UTRNet recommends 3.7) and PyTorch version; clone full repo including modules/.",
            )
            return

    print(f"  [PASS] UTRNet repository valid; model, utils, read imported successfully.")
    print("  (No training or dataset required.)")

# ---------------------------------------------------------------------------
# 10. Final report
# ---------------------------------------------------------------------------
def print_final_report() -> None:
    _section("10. Final Report")
    broken = [i for i in issues if i[0] == Result.BROKEN]
    warnings = [i for i in issues if i[0] == Result.WARNING]

    if broken:
        print("  Status: BROKEN")
        for _, what, fix in broken:
            print(f"    - {what}")
            if fix:
                print(f"      Fix (Windows): {fix.replace(chr(10), chr(10) + '      ')}")
    if warnings:
        print("  Warnings:")
        for _, what, fix in warnings:
            print(f"    - {what}")
            if fix:
                print(f"      Suggestion: {fix.replace(chr(10), chr(10) + '      ')}")

    if not broken and not warnings:
        print("  READY - Environment is correctly configured for bilingual OCR.")
    elif not broken and warnings:
        print("  WARNING - Environment is usable but some warnings apply.")
    else:
        print("  BROKEN - Fix the issues above before running OCR pipelines.")

    print()
    if broken:
        print("  Summary: Fix all BROKEN items, then re-run this script.")
    elif warnings:
        print("  Summary: Address WARNINGs if you hit compatibility issues (e.g. UTRNet).")
    else:
        print("  Summary: Ready for printed Urdu (YOLOv8 + UTRNet-Large) and handwritten English (TrOCR).")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("Bilingual OCR Environment Verification (Windows, local)")
    print("TrOCR | YOLOv8 | UTRNet-Large | OpenCV | Tesseract")
    check_python()
    check_environment()
    check_pytorch()
    check_core_deps()
    check_tesseract()
    check_utrnet_weights()
    check_yolov8_sanity()
    check_trocr_sanity()
    check_utrnet_import()
    print_final_report()
    broken_count = sum(1 for i in issues if i[0] == Result.BROKEN)
    return 1 if broken_count else 0

if __name__ == "__main__":
    sys.exit(main())
