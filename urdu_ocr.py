"""
Production Urdu OCR entry point: PDF and batch image support.

- PDF input: convert each page to image (pypdfium2), run existing YOLOv8 + UTRNet pipeline per page.
- Batch input: process all images in a folder (jpg/png) through the same pipeline.
- Output: single UTF-8 text file with clear page/image separators.

Reuses urdu_ocr_pipeline.run_urdu_ocr. No new ML, no GUI, no cloud. Python only.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Iterator

from urdu_ocr_pipeline import run_urdu_ocr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT = "result.txt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEPARATOR_WIDTH = 60


def _pdf_pages_to_images(pdf_path: Path) -> tuple[list[Path], object]:
    """
    Convert each PDF page to a PNG image using pypdfium2 (lightweight, no poppler).

    Renders each page into a temporary directory. Returns (list of image paths, temp_dir).
    Caller must keep temp_dir alive while using the paths; use as context manager.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise ImportError(
            "PDF support requires pypdfium2. Install with: pip install pypdfium2"
        ) from None

    pdf = pdfium.PdfDocument(str(pdf_path))
    n_pages = len(pdf)
    temp_dir = tempfile.TemporaryDirectory(prefix="urdu_ocr_pdf_")
    temp_path = Path(temp_dir.name)
    paths = []
    for i in range(n_pages):
        page = pdf.get_page(i)
        pil_image = page.render_topil(
            scale=2.0,  # 2x for better OCR on small text
            rotation=0,
            colour=(255, 255, 255, 255),
        )
        out = temp_path / f"page_{i + 1:04d}.png"
        pil_image.save(str(out))
        paths.append(out)
    pdf.close()
    return paths, temp_dir


def _collect_images_from_folder(folder: Path) -> list[Path]:
    """Collect jpg/png paths from folder, sorted by name for stable order."""
    if not folder.is_dir():
        return []
    paths = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort(key=lambda p: p.name.lower())
    return paths


def _write_combined_output(
    output_path: Path,
    items: list[tuple[str, list[str]]],
) -> None:
    """
    Write all recognized text to a single UTF-8 file with clear separators.

    items: list of (label, lines) e.g. ("Page 1 (doc.pdf)", ["line1", "line2"])
    Preserves page and image boundaries for traceability.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for label, lines in items:
            sep = "=" * SEPARATOR_WIDTH
            f.write(f"{sep}\n")
            f.write(f" {label}\n")
            f.write(f"{sep}\n\n")
            for line in lines:
                f.write(line + "\n")
            f.write("\n")
    return


def _process_input(
    input_path: Path,
    verbose: bool,
) -> Iterator[tuple[str, list[str]]]:
    """
    Yield (label, lines) for each page/image from input_path.

    input_path can be a PDF file or a directory of images.
    Skips unreadable items and continues; raises only on fatal errors.
    """
    path = Path(input_path).resolve()
    if path.is_file():
        if path.suffix.lower() == ".pdf":
            # PDF: convert each page to image (pypdfium2), then run pipeline per page
            try:
                page_paths, _pdf_temp = _pdf_pages_to_images(path)
            except Exception as e:
                print(f"Error: Could not read PDF: {e}", file=sys.stderr)
                return
            # Keep _pdf_temp alive until all pages are processed (temp dir cleanup on exit)
            for i, page_path in enumerate(page_paths):
                label = f"Page {i + 1} / {len(page_paths)} ({path.name})"
                try:
                    lines = run_urdu_ocr(
                        page_path,
                        output_dir=path.parent / f"_pdf_pages_{path.stem}" / f"page_{i + 1}",
                        verbose=verbose,
                    )
                except Exception as e:
                    print(f"Warning: Skipping {label}: {e}", file=sys.stderr)
                    lines = [f"[Skipped: {e}]"]
                yield (label, lines)
            return
        # Single image
        label = path.name
        try:
            lines = run_urdu_ocr(path, verbose=verbose)
        except Exception as e:
            print(f"Warning: Skipping {path.name}: {e}", file=sys.stderr)
            lines = [f"[Skipped: {e}]"]
        yield (label, lines)
        return

    if path.is_dir():
        # Batch: all images in folder
        image_paths = _collect_images_from_folder(path)
        if not image_paths:
            print(f"Warning: No jpg/png images found in {path}", file=sys.stderr)
            return
        for i, img_path in enumerate(image_paths):
            label = f"{i + 1} / {len(image_paths)}: {img_path.name}"
            try:
                lines = run_urdu_ocr(img_path, verbose=verbose)
            except Exception as e:
                print(f"Warning: Skipping {img_path.name}: {e}", file=sys.stderr)
                lines = [f"[Skipped: {e}]"]
            yield (label, lines)
        return

    print(f"Error: Input not found (file or directory): {path}", file=sys.stderr)


def run(
    input_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    verbose: bool = True,
) -> Path:
    """
    Run Urdu OCR on input (PDF or image folder), write combined text to output_path.

    Skips unreadable images/pages and continues. No crash on single-item failure.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    items = list(_process_input(input_path, verbose=verbose))
    if not items:
        print("No pages or images processed.", file=sys.stderr)
        return output_path
    _write_combined_output(output_path, items)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Urdu OCR: PDF or folder of images → single UTF-8 text file (YOLOv8 + UTRNet).",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        metavar="PATH",
        help="Input: path to a PDF file or to a folder of images (jpg/png).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="FILE",
        help=f"Output UTF-8 text file (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce per-image progress (detection count, recognition steps).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    out = run(
        args.input,
        args.output,
        verbose=not args.quiet,
    )
    print(f"Output written to: {out}")


if __name__ == "__main__":
    _main()
