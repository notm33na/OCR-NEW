"""
OCR Step 4: Printed Urdu text recognition using UTRNet-Large (pretrained).

Recognizes printed Urdu from cropped text region images. Prefers in-process
UTRNet (load model once, infer all crops); falls back to subprocess read.py.
Fixed config: HRNet + DBiLSTM + CTC. Windows-compatible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict


def _configure_stdout_utf8() -> None:
    """Force UTF-8 on Windows to avoid UnicodeEncodeError when printing Urdu."""
    if sys.platform != "win32":
        return
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Paths (UTRNet repo and pretrained weights)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
UTRNET_REPO_DIR = PROJECT_ROOT / "UTRNet-High-Resolution-Urdu-Text-Recognition"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "saved_models" / "UTRNet-Large" / "best_norm_ED.pth"

# Fixed config matching read.py defaults
FEATURE_EXTRACTION = "HRNet"
SEQUENCE_MODELING = "DBiLSTM"
PREDICTION = "CTC"
IMG_H = 32
IMG_W = 400
SUBPROCESS_TIMEOUT = 120  # seconds per image when using subprocess fallback

class RecognitionResult(TypedDict):
    """Single crop recognition result."""
    image: str
    text: str


def _load_utrnet_inprocess(repo: Path, model_path: Path, device: Any):
    """Import UTRNet modules from repo, load model once, return (model, converter, transform, opt)."""
    repo_str = str(repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from PIL import Image
    import torch
    from model import Model
    from dataset import NormalizePAD
    from utils import CTCLabelConverter, AttnLabelConverter

    glyphs_path = repo / "UrduGlyphs.txt"
    with open(glyphs_path, "r", encoding="utf-8") as f:
        content = "".join(line.strip() for line in f)
    character = content + " "

    opt = SimpleNamespace(
        imgH=IMG_H,
        imgW=IMG_W,
        rgb=False,
        FeatureExtraction=FEATURE_EXTRACTION,
        SequenceModeling=SEQUENCE_MODELING,
        Prediction=PREDICTION,
        num_fiducial=20,
        input_channel=1,
        output_channel=32 if FEATURE_EXTRACTION == "HRNet" else 512,
        hidden_size=256,
        batch_max_length=100,
        character=character,
        saved_model=str(model_path),
        device=device,
    )
    opt.num_class = len(character)
    if opt.rgb:
        opt.input_channel = 3

    converter = CTCLabelConverter(opt.character) if "CTC" in PREDICTION else AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    model = Model(opt)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    model = model.to(device)
    model.eval()
    transform = NormalizePAD((1, opt.imgH, opt.imgW))
    return model, converter, transform, opt, Image


def _recognize_one_inprocess(
    image_path: Path,
    model: Any,
    converter: Any,
    transform: Any,
    opt: Any,
    Image: Any,
    device: Any,
) -> str:
    """Run UTRNet inference on one image (same preprocessing as read.py)."""
    import torch
    if opt.rgb:
        img = Image.open(str(image_path)).convert("RGB")
    else:
        img = Image.open(str(image_path)).convert("L")
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    w, h = img.size
    ratio = w / float(h)
    if math.ceil(opt.imgH * ratio) > opt.imgW:
        resized_w = opt.imgW
    else:
        resized_w = math.ceil(opt.imgH * ratio)
    img = img.resize((resized_w, opt.imgH), Image.Resampling.BICUBIC)
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(img)
    preds_size = torch.IntTensor([preds.size(1)])
    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index.data, preds_size.data)[0]
    return preds_str


def recognize_urdu(
    image_path: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    utrnet_repo: str | Path = UTRNET_REPO_DIR,
    _inprocess_state: dict | None = None,
) -> str:
    """
    Recognize printed Urdu text in a single image using UTRNet-Large (pretrained).

    Uses in-process UTRNet when _inprocess_state is provided (model already loaded);
    otherwise uses subprocess read.py. Returns the recognized text string.
    """
    path = Path(image_path).resolve()
    repo = Path(utrnet_repo).resolve()
    model = Path(model_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if not model.is_file():
        raise FileNotFoundError(f"Model not found: {model}. Place UTRNet-Large best_norm_ED.pth there.")
    if not repo.is_dir() or not (repo / "read.py").is_file():
        raise FileNotFoundError(f"UTRNet repo not found: {repo}.")

    # In-process path (when state is passed from recognize_all)
    if _inprocess_state is not None:
        return _recognize_one_inprocess(
            path,
            _inprocess_state["model"],
            _inprocess_state["converter"],
            _inprocess_state["transform"],
            _inprocess_state["opt"],
            _inprocess_state["Image"],
            _inprocess_state["device"],
        )

    # Subprocess fallback (single image)
    cmd = [
        sys.executable,
        str(repo / "read.py"),
        "--image_path", str(path),
        "--saved_model", str(model),
        "--FeatureExtraction", FEATURE_EXTRACTION,
        "--SequenceModeling", SEQUENCE_MODELING,
        "--Prediction", PREDICTION,
    ]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        cmd,
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=SUBPROCESS_TIMEOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"UTRNet read.py failed (exit {result.returncode}): {err}")
    lines = [ln.strip() for ln in (result.stdout or "").strip().splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("UTRNet read.py produced no output.")
    return lines[-1]


def get_urdu_inprocess_state(
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    utrnet_repo: str | Path = UTRNET_REPO_DIR,
) -> dict | None:
    """
    Load UTRNet once and return state for in-process recognition (or None if unavailable).

    Callers (e.g. bilingual_ocr) can pass this to recognize_urdu(..., _inprocess_state=state)
    to avoid one subprocess per crop.
    """
    repo = Path(utrnet_repo).resolve()
    model = Path(model_path).resolve()
    if not model.is_file() or not (repo / "read.py").is_file():
        return None
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_obj, converter, transform, opt, Image_mod = _load_utrnet_inprocess(repo, model, device)
        return {
            "model": model_obj,
            "converter": converter,
            "transform": transform,
            "opt": opt,
            "Image": Image_mod,
            "device": device,
        }
    except Exception:
        return None


def recognize_all(
    crop_dir: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    utrnet_repo: str | Path = UTRNET_REPO_DIR,
) -> list[RecognitionResult]:
    """
    Run recognition on all cropped images in a directory, in order.

    Loads UTRNet model once (in-process) and runs inference for each crop;
    falls back to subprocess read.py per image if in-process import fails.
    Expects crops named 0.png, 1.png, ... (as produced by text_detection_yolov8.py).

    Args:
        crop_dir: Directory containing crop images (e.g. output_dir from Step 3).
        model_path: Path to UTRNet-Large checkpoint.
        utrnet_repo: Path to UTRNet repo.

    Returns:
        List of {"image": "0.png", "text": "recognized Urdu text"} in order.
    """
    crop_dir = Path(crop_dir)
    repo = Path(utrnet_repo).resolve()
    model = Path(model_path).resolve()
    if not crop_dir.is_dir():
        return []
    if not model.is_file() or not (repo / "read.py").is_file():
        return []

    def sort_key(p: Path) -> tuple[int, str]:
        try:
            return (int(p.stem), p.name)
        except ValueError:
            return (999999, p.name)

    image_files = sorted(
        [p for p in crop_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")],
        key=sort_key,
    )
    if not image_files:
        return []

    results: list[RecognitionResult] = []
    inprocess_state = None

    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_obj, converter, transform, opt, Image_mod = _load_utrnet_inprocess(repo, model, device)
        inprocess_state = {
            "model": model_obj,
            "converter": converter,
            "transform": transform,
            "opt": opt,
            "Image": Image_mod,
            "device": device,
        }
    except Exception as e:
        if image_files:
            print(f"In-process UTRNet unavailable ({e}); using subprocess (slower).", flush=True)

    for p in image_files:
        try:
            if inprocess_state is not None:
                text = recognize_urdu(p, model_path=model_path, utrnet_repo=utrnet_repo, _inprocess_state=inprocess_state)
            else:
                text = recognize_urdu(p, model_path=model_path, utrnet_repo=utrnet_repo)
        except Exception as e:
            text = f"[Error: {e}]"
        results.append({"image": p.name, "text": text})
    return results


# ---------------------------------------------------------------------------
# Main: read crops from directory, run recognition, save/print ASCII-safe
# ---------------------------------------------------------------------------
def _run_demo(
    crop_dir: str | Path,
    *,
    output_txt: Path | None = None,
    output_json: Path | None = None,
) -> bool:
    """Run recognition on all crops in crop_dir; optionally save to UTF-8 .txt and .json. Returns True on success."""
    crop_dir = Path(crop_dir)
    if not crop_dir.is_dir():
        print(f"Error: Directory not found: {crop_dir}", file=sys.stderr)
        print("Usage: python urdu_recognition_utrnet.py <crop_dir> [--output-txt PATH] [--output-json PATH]", file=sys.stderr)
        return False

    print(f"Crop directory: {crop_dir}", flush=True)
    print("Running UTRNet-Large (HRNet + DBiLSTM + CTC)...", flush=True)
    results = recognize_all(crop_dir)
    n = len(results)
    print(f"Processed {n} image(s).", flush=True)

    if output_txt is not None:
        output_txt = Path(output_txt)
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(output_txt, "w", encoding="utf-8") as f:
            for r in results:
                f.write(f"{r['image']}: {r['text']}\n")
        print(f"Results saved to {output_txt}", flush=True)

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_json}", flush=True)

    return True


if __name__ == "__main__":
    _configure_stdout_utf8()
    parser = argparse.ArgumentParser(
        description="Recognize printed Urdu text in cropped images using pretrained UTRNet-Large.",
    )
    parser.add_argument(
        "crop_dir",
        type=Path,
        help="Directory of cropped images (e.g. from text_detection_yolov8.py: 0.png, 1.png, ...).",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save recognition results to a UTF-8 encoded .txt file (one line per image: text).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save recognition results to a .json file (list of {image, text} objects).",
    )
    args = parser.parse_args()
    ok = _run_demo(args.crop_dir, output_txt=args.output_txt, output_json=args.output_json)
    sys.exit(0 if ok else 1)
