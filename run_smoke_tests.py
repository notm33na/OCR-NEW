"""
Smoke test runner for the bilingual OCR project (Windows, CPU-only).
Automates STEP A, STEP B, and STEP C from RUN_AND_TEST.md.
Uses subprocess to invoke existing scripts; stops immediately on failure.
"""

import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (pathlib, Windows-compatible)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_INPUTS = PROJECT_ROOT / "sample_inputs"
OUTPUTS = PROJECT_ROOT / "outputs"
TEST_CROPS = OUTPUTS / "test_crops"

# Step A
VERIFY_SCRIPT = PROJECT_ROOT / "verify_bilingual_ocr.py"

# Step B inputs/outputs
URDU_PRINTED_IMAGE = SAMPLE_INPUTS / "urdu_printed_1.jpg"
ENGLISH_HANDWRITTEN_IMAGE = SAMPLE_INPUTS / "english_handwritten_1.jpg"

# Preprocessing outputs (same folder as input)
URDU_PREPROCESSED_PNG = SAMPLE_INPUTS / "urdu_printed_1_preprocessed.png"
URDU_PREPROCESSED_GRAY_PNG = SAMPLE_INPUTS / "urdu_printed_1_preprocessed_gray.png"

# Step C output
URDU_RESULT_TXT = OUTPUTS / "urdu_result.txt"

# Timeouts (seconds)
VERIFY_TIMEOUT = 120
PREPROCESS_TIMEOUT = 90   # GUI may block; files are written before display
DETECTION_TIMEOUT = 120
UTRNET_TIMEOUT = 300
TROCR_TIMEOUT = 120
URDU_OCR_TIMEOUT = 300


def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str, exit_code: int = 1) -> None:
    log(msg)
    sys.exit(exit_code)


def run_cmd(
    cmd: list[str],
    timeout: int,
    step_name: str,
    check_returncode: bool = True,
    capture_stdout: bool = False,
) -> subprocess.CompletedProcess:
    """Run command; on timeout or non-zero, exit with clear message."""
    log(f"  Running: {' '.join(str(c) for c in cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            timeout=timeout,
            capture_output=capture_stdout,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        fail(f"❌ {step_name}: command timed out after {timeout}s")
    except OSError as e:
        fail(f"❌ {step_name}: failed to run command: {e}")
    if check_returncode and result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        fail(f"❌ {step_name}: exit code {result.returncode}\n{err}")
    return result


def step_a_preflight() -> None:
    """STEP A — Pre-flight verification: verify_bilingual_ocr.py."""
    log("")
    log("========== STEP A — Pre-flight verification ==========")
    t0 = time.perf_counter()
    result = run_cmd(
        [sys.executable, str(VERIFY_SCRIPT)],
        timeout=VERIFY_TIMEOUT,
        step_name="STEP A",
        check_returncode=False,
        capture_stdout=True,
    )
    elapsed = time.perf_counter() - t0
    # Show verification output so user sees status
    if result.stdout:
        log(result.stdout)
    if result.stderr:
        log(result.stderr)
    out = (result.stdout or "") + (result.stderr or "")
    if "BROKEN" in out:
        fail("❌ STEP A FAILED: Environment is BROKEN")
    # Print status line (READY or WARNING)
    for line in out.splitlines():
        if "READY" in line or "WARNING" in line or "Status:" in line:
            log(line.strip())
            break
    log(f"  STEP A completed in {elapsed:.1f}s")
    log("")


def step_b_preprocessing() -> None:
    """B.1 — Preprocessing: preprocess.py sample_inputs/urdu_printed_1.jpg"""
    log("  --- B.1 Preprocessing ---")
    if not URDU_PRINTED_IMAGE.exists():
        fail(f"❌ STEP B.1 FAILED: input not found: {URDU_PRINTED_IMAGE}")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "preprocess.py", str(URDU_PRINTED_IMAGE)],
            cwd=str(PROJECT_ROOT),
            timeout=PREPROCESS_TIMEOUT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        # Files are written before GUI; check outputs even on timeout
        if URDU_PREPROCESSED_PNG.exists() and URDU_PREPROCESSED_GRAY_PNG.exists():
            log(f"  (Preprocess timed out waiting for GUI; output files present)")
            return
        fail(f"❌ STEP B.1 FAILED: timed out and output files not created")
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        fail(f"❌ STEP B.1 FAILED: exit code {proc.returncode}\n{proc.stderr or proc.stdout or ''}")
    if not URDU_PREPROCESSED_PNG.exists():
        fail(f"❌ STEP B.1 FAILED: output not created: {URDU_PREPROCESSED_PNG}")
    if not URDU_PREPROCESSED_GRAY_PNG.exists():
        fail(f"❌ STEP B.1 FAILED: output not created: {URDU_PREPROCESSED_GRAY_PNG}")
    log(f"  B.1 completed in {elapsed:.1f}s")


def step_b_yolov8() -> None:
    """B.2 — YOLOv8 detection; at least one .png in outputs/test_crops/."""
    log("  --- B.2 YOLOv8 detection ---")
    TEST_CROPS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    run_cmd(
        [sys.executable, "text_detection_yolov8.py", str(URDU_PRINTED_IMAGE), str(TEST_CROPS)],
        timeout=DETECTION_TIMEOUT,
        step_name="STEP B.2",
    )
    elapsed = time.perf_counter() - t0
    pngs = list(TEST_CROPS.glob("*.png"))
    if not pngs:
        fail(f"❌ STEP B.2 FAILED: no .png in {TEST_CROPS}")
    log(f"  B.2 completed in {elapsed:.1f}s ({len(pngs)} crop(s))")


def step_b_utrnet() -> None:
    """B.3 — UTRNet recognition; must print Urdu text (no subprocess error)."""
    log("  --- B.3 UTRNet recognition ---")
    t0 = time.perf_counter()
    result = run_cmd(
        [sys.executable, "urdu_recognition_utrnet.py", str(TEST_CROPS)],
        timeout=UTRNET_TIMEOUT,
        step_name="STEP B.3",
        capture_stdout=True,
    )
    elapsed = time.perf_counter() - t0
    out = (result.stdout or "").strip()
    # Require some non-empty output (Urdu or placeholder lines like "0.png: ...")
    if not out:
        fail("❌ STEP B.3 FAILED: no Urdu text printed")
    log(f"  B.3 completed in {elapsed:.1f}s")


def step_b_trocr() -> None:
    """B.4 — TrOCR English; must produce output text."""
    log("  --- B.4 TrOCR English ---")
    if not ENGLISH_HANDWRITTEN_IMAGE.exists():
        fail(f"❌ STEP B.4 FAILED: input not found: {ENGLISH_HANDWRITTEN_IMAGE}")
    t0 = time.perf_counter()
    result = run_cmd(
        [sys.executable, "english_ocr_pipeline.py", str(ENGLISH_HANDWRITTEN_IMAGE)],
        timeout=TROCR_TIMEOUT,
        step_name="STEP B.4",
        capture_stdout=True,
    )
    elapsed = time.perf_counter() - t0
    out = (result.stdout or "").strip()
    if not out:
        fail("❌ STEP B.4 FAILED: no output text")
    log(f"  B.4 completed in {elapsed:.1f}s")


def step_c_e2e_urdu() -> None:
    """STEP C — End-to-end printed Urdu OCR; outputs/urdu_result.txt exists and size > 0."""
    log("")
    log("========== STEP C — End-to-end printed Urdu OCR ==========")
    if not URDU_PRINTED_IMAGE.exists():
        fail(f"❌ STEP C FAILED: input not found: {URDU_PRINTED_IMAGE}")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    run_cmd(
        [
            sys.executable,
            "urdu_ocr.py",
            "--input", str(URDU_PRINTED_IMAGE),
            "--output", str(URDU_RESULT_TXT),
        ],
        timeout=URDU_OCR_TIMEOUT,
        step_name="STEP C",
    )
    elapsed = time.perf_counter() - t0
    if not URDU_RESULT_TXT.exists():
        fail(f"❌ STEP C FAILED: output file not created: {URDU_RESULT_TXT}")
    if URDU_RESULT_TXT.stat().st_size == 0:
        fail(f"❌ STEP C FAILED: output file is empty: {URDU_RESULT_TXT}")
    log(f"  STEP C completed in {elapsed:.1f}s")
    log("")


def main() -> int:
    log("Smoke tests — Windows, CPU-only (subprocess)")
    log("Project root: " + str(PROJECT_ROOT))
    try:
        step_a_preflight()
        log("========== STEP B — Unit sanity tests ==========")
        step_b_preprocessing()
        step_b_yolov8()
        step_b_utrnet()
        step_b_trocr()
        step_c_e2e_urdu()
    except SystemExit as e:
        raise
    log("========== FINAL SUMMARY ==========")
    log("✅ SMOKE TEST PASSED")
    log("System is READY for API wrapping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
