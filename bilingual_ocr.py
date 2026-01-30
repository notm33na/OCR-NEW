"""
Unified bilingual OCR orchestrator.

Reuses: preprocess, text_detection_yolov8 (YOLOv8) or text_region_detection (OpenCV),
script_detection, urdu_ocr_pipeline / urdu_recognition_utrnet, english_ocr_pipeline,
post_process (Urdu RTL cleanup). No new ML, no training, CPU-only, Windows-compatible.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

import cv2

from preprocess import preprocess_image
from text_detection_yolov8 import detect_and_crop_text
from text_region_detection import detect_text_blocks
from script_detection import detect_script, detect_script_page
from urdu_recognition_utrnet import get_urdu_inprocess_state, recognize_urdu
from english_ocr_pipeline import recognize_english
from post_process import _clean_urdu_text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT = "result.txt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEPARATOR_WIDTH = 60
DEBUG_BASE_DIR = Path(__file__).resolve().parent / "outputs" / "debug"


def _pdf_pages_to_images(pdf_path: Path) -> tuple[list[Path], object]:
    """Convert each PDF page to PNG using pypdfium2. Returns (paths, temp_dir)."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise ImportError("PDF support requires pypdfium2. pip install pypdfium2") from None
    pdf = pdfium.PdfDocument(str(pdf_path))
    n_pages = len(pdf)
    temp_dir = tempfile.TemporaryDirectory(prefix="bilingual_ocr_pdf_")
    temp_path = Path(temp_dir.name)
    paths = []
    for i in range(n_pages):
        page = pdf.get_page(i)
        pil_image = page.render_topil(scale=2.0, rotation=0, colour=(255, 255, 255, 255))
        out = temp_path / f"page_{i + 1:04d}.png"
        pil_image.save(str(out))
        paths.append(out)
    pdf.close()
    return paths, temp_dir


def _collect_images_from_folder(folder: Path) -> list[Path]:
    """Collect jpg/png from folder, sorted by name. Skips preprocessed outputs (e.g. *_preprocessed.png)."""
    if not folder.is_dir():
        return []
    paths = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        and "_preprocessed" not in p.stem and "_preprocessed_gray" not in p.stem
    ]
    paths.sort(key=lambda p: p.name.lower())
    return paths


def _crop_boxes_to_dir(image_path: Path, boxes: list[tuple[int, int, int, int]], crop_dir: Path) -> list[Path]:
    """Crop image using (x,y,w,h) boxes and save to crop_dir. Returns list of paths in order."""
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    crop_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (x, y, w, h) in enumerate(boxes):
        if w <= 0 or h <= 0:
            continue
        crop = image[y : y + h, x : x + w]
        p = crop_dir / f"{i}.png"
        cv2.imwrite(str(p), crop)
        paths.append(p)
    return paths


def _run_detection_and_crops(
    image_path: Path,
    crop_dir: Path,
    use_yolov8: bool = True,
    debug_page_dir: Path | None = None,
) -> list[Path]:
    """
    Run text detection and save crops. YOLOv8 preferred; OpenCV fallback if no crops.
    When debug_page_dir is set, save preprocessed image there before detection.
    Returns list of crop paths in reading order.
    """
    if debug_page_dir is not None:
        debug_page_dir.mkdir(parents=True, exist_ok=True)
        try:
            binary, gray = preprocess_image(image_path)
            cv2.imwrite(str(debug_page_dir / "preprocessed_binary.png"), binary)
            cv2.imwrite(str(debug_page_dir / "preprocessed_gray.png"), gray)
        except Exception:
            pass
    if use_yolov8:
        try:
            paths = detect_and_crop_text(image_path, crop_dir, verbose=False)
            if paths:
                return paths
        except Exception:
            pass
    # OpenCV fallback: preprocess -> detect_text_blocks -> crop
    try:
        binary, _ = preprocess_image(image_path)
        boxes = detect_text_blocks(binary)
        if not boxes:
            return []
        return _crop_boxes_to_dir(image_path, boxes, crop_dir)
    except Exception:
        return []


def _process_one_page(
    image_path: Path,
    lang: str,
    crop_base_dir: Path | None,
    quiet: bool,
    debug_page_dir: Path | None = None,
) -> list[dict[str, str]]:
    """
    Process one page/image: detect regions, script-detect, route to Urdu or English, clean Urdu.
    When debug_page_dir is set, saves preprocessed images and crops under that dir.
    Returns list of {"language": "urdu"|"english", "text": "..."} in reading order.
    """
    if debug_page_dir is not None:
        crop_dir = debug_page_dir / "crops"
    elif crop_base_dir is None:
        crop_dir = image_path.parent / f"_bilingual_crops_{image_path.stem}"
    else:
        crop_dir = Path(crop_base_dir)
    crop_paths = _run_detection_and_crops(
        image_path, crop_dir, use_yolov8=True, debug_page_dir=debug_page_dir
    )
    if not crop_paths:
        if not quiet:
            print(f"  No regions detected: {image_path.name}", file=sys.stderr)
        return []

    urdu_state = get_urdu_inprocess_state() if lang in ("auto", "urdu") else None
    page_script_urdu = detect_script_page(image_path) if lang == "auto" else None
    regions = []
    for i, crop_path in enumerate(crop_paths):
        try:
            if lang == "auto":
                if page_script_urdu == "urdu":
                    script = "urdu"
                else:
                    script = detect_script(crop_path)["script"]
            elif lang == "urdu":
                script = "urdu"
            else:
                script = "english"

            if script == "urdu":
                text = recognize_urdu(
                    crop_path,
                    _inprocess_state=urdu_state,
                )
                text = _clean_urdu_text(text)
            else:
                text = recognize_english(crop_path)
            regions.append({"language": script, "text": text})
        except Exception as e:
            if not quiet:
                print(f"  Skip crop {crop_path.name}: {e}", file=sys.stderr)
            regions.append({"language": "unknown", "text": f"[Error: {e}]"})
    return regions


def _enumerate_input(
    input_path: Path,
    lang: str,
    crop_base_dir: Path | None,
    quiet: bool,
    debug_images: bool = False,
    debug_base_dir: Path | None = None,
) -> Iterator[tuple[str, list[dict[str, str]]]]:
    """
    Yield (label, regions) for each page/image. label e.g. "Page 1 (doc.pdf)" or "1/5: img.jpg".
    When debug_images is True, saves preprocessed + crops under debug_base_dir/page_N/.
    Skips bad pages; does not crash.
    """
    path = Path(input_path).resolve()
    _pdf_temp = None
    if debug_base_dir is None:
        debug_base_dir = DEBUG_BASE_DIR

    def _do_page(page_index: int, label: str, image_path: Path) -> list[dict[str, str]]:
        debug_page_dir = (debug_base_dir / f"page_{page_index}") if debug_images else None
        if not quiet:
            print(f"  Page {page_index} started: {label}")
        try:
            regions = _process_one_page(
                image_path, lang, crop_base_dir, quiet, debug_page_dir=debug_page_dir
            )
        except Exception as e:
            if not quiet:
                print(f"  Warning: Skip {label}: {e}", file=sys.stderr)
            return [{"language": "unknown", "text": f"[Skipped: {e}]"}]
        if not quiet and regions:
            n_urdu = sum(1 for r in regions if r.get("language") == "urdu")
            n_english = sum(1 for r in regions if r.get("language") == "english")
            print(f"  Regions detected: {len(regions)}")
            print(f"  Routing: Urdu {n_urdu}, English {n_english}")
        if not quiet:
            print(f"  Page {page_index} completed.")
        return regions

    if path.is_file():
        if path.suffix.lower() == ".pdf":
            try:
                page_paths, _pdf_temp = _pdf_pages_to_images(path)
            except Exception as e:
                if not quiet:
                    print(f"Error: PDF: {e}", file=sys.stderr)
                return
            for i, page_path in enumerate(page_paths):
                label = f"Page {i + 1} / {len(page_paths)} ({path.name})"
                regions = _do_page(i + 1, label, page_path)
                yield (label, regions)
            return
        # Single image
        label = path.name
        regions = _do_page(1, label, path)
        yield (label, regions)
        return

    if path.is_dir():
        image_paths = _collect_images_from_folder(path)
        if not image_paths:
            if not quiet:
                print(f"Warning: No jpg/png in {path}", file=sys.stderr)
            return
        for i, img_path in enumerate(image_paths):
            label = f"{i + 1} / {len(image_paths)}: {img_path.name}"
            regions = _do_page(i + 1, label, img_path)
            yield (label, regions)
        return

    if not quiet:
        print(f"Error: Not found: {path}", file=sys.stderr)


def _write_text_output(output_path: Path, items: list[tuple[str, list[dict[str, str]]]]) -> None:
    """Write one UTF-8 text file with separators; each region as language: text."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for label, regions in items:
            sep = "=" * SEPARATOR_WIDTH
            f.write(f"{sep}\n {label}\n{sep}\n\n")
            for r in regions:
                f.write(f"[{r['language']}] {r['text']}\n")
            f.write("\n")


def _build_json_output(items: list[tuple[str, list[dict[str, str]]]]) -> list[dict[str, Any]]:
    """Build list of {page: N, regions: [{language, text}]} for JSON save."""
    out = []
    for page_num, (label, regions) in enumerate(items, start=1):
        out.append({"page": page_num, "regions": regions})
    return out


def run(
    input_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    lang: str = "auto",
    save_json: bool = False,
    json_path: Path | None = None,
    debug_images: bool = False,
    debug_base_dir: Path | None = None,
    quiet: bool = False,
) -> Path:
    """
    Run bilingual OCR on input (single image, folder, or PDF). Write UTF-8 text file;
    optionally save JSON with {page, regions: [{language, text}]}.
    When debug_images is True, saves preprocessed + crops under outputs/debug/page_X/.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if lang not in ("auto", "urdu", "english"):
        lang = "auto"

    if not quiet:
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Lang: {lang} (auto=script detection, urdu=Urdu only, english=English only)")
        if debug_images:
            print(f"Debug images: {debug_base_dir or DEBUG_BASE_DIR}")

    items = list(_enumerate_input(
        input_path, lang, None, quiet,
        debug_images=debug_images,
        debug_base_dir=debug_base_dir,
    ))
    if not items:
        if not quiet:
            print("No pages or images processed.", file=sys.stderr)
        return output_path

    _write_text_output(output_path, items)

    if save_json:
        jpath = json_path or output_path.with_suffix(".json")
        jpath.parent.mkdir(parents=True, exist_ok=True)
        data = _build_json_output(items)
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if not quiet:
            print(f"JSON: {jpath}")

    if not quiet:
        print("Done.")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Bilingual OCR: Urdu (UTRNet) + English (TrOCR). Single image, folder, or PDF.",
    )
    parser.add_argument("--input", "-i", type=Path, required=True, metavar="PATH", help="PDF, image file, or folder of images.")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT, metavar="FILE", help=f"Output UTF-8 text file (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--lang", type=str, choices=["auto", "urdu", "english"], default="auto", help="Language: auto (script detection), urdu, or english.")
    parser.add_argument("--save-json", action="store_true", help="Save structured JSON: {page, regions: [{language, text}]}.")
    parser.add_argument("--json", type=Path, default=None, metavar="FILE", help="Path for JSON (default: <output>.json).")
    parser.add_argument("--debug-images", action="store_true", help="Save preprocessed images and crops under outputs/debug/page_X/.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-essential logs; high-level progress only.")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    run(
        args.input,
        args.output,
        lang=args.lang,
        save_json=args.save_json,
        json_path=args.json,
        debug_images=args.debug_images,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    _main()
