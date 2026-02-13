#!/usr/bin/env python3
"""
Download the YOLOv8 UrduDoc text-detection model for local use.
Creates urdu-text-detection/yolov8m_UrduDoc.pt (same as used in Docker at build time).

Run from OCR-NEW root:
    python scripts/download_urdudoc_model.py

Source: https://github.com/abdur75648/urdu-text-detection/releases
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import urllib.request
except ImportError:
    urllib.request = None  # type: ignore

URL = "https://github.com/abdur75648/urdu-text-detection/releases/download/v1.0.0/yolov8m_UrduDoc.pt"


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "urdu-text-detection"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "yolov8m_UrduDoc.pt"

    if out_path.is_file():
        print(f"Model already present: {out_path}")
        return

    print(f"Downloading YOLOv8 UrduDoc model to {out_path} ...")
    try:
        urllib.request.urlretrieve(URL, out_path)  # type: ignore
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
