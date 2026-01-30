"""
Lightweight, explainable evaluation metrics for the bilingual OCR system.

No ground truth, no accuracy %, no datasets. Operates on structured JSON
output from bilingual_ocr.py. Pure Python; no OCR models or reruns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_REPORT_PATH = Path(__file__).resolve().parent / "outputs" / "evaluation.json"


def _is_failed_or_empty(text: str) -> bool:
    """True if region text indicates failure or is effectively empty."""
    if not text or not text.strip():
        return True
    t = text.strip()
    if t.startswith("[Error") or t.startswith("[Skipped"):
        return True
    return False


def evaluate_bilingual_output(
    results_json: list[dict[str, Any]] | str | Path,
    timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Compute evaluation metrics from bilingual_ocr.py structured JSON output.

    Accepts: list of {page, regions: [{language, text}]}, or path to JSON file.
    timings: optional {"total_time": seconds, "pages_skipped": N}. If missing,
    total_time and avg_time_per_page are omitted; pages_skipped derived from data.

    Returns:
        dict with: total_regions, regions_with_text, empty_regions,
        count_urdu, count_english, total_time (if timings), avg_time_per_page (if timings),
        regions_failed_ocr, pages_skipped.
    """
    if timings is None:
        timings = {}

    # Load JSON if path
    if isinstance(results_json, (str, Path)):
        path = Path(results_json)
        if not path.is_file():
            return {"error": f"File not found: {path}"}
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = results_json

    if not isinstance(results, list):
        return {"error": "Expected list of {page, regions}."}

    total_regions = 0
    regions_with_text = 0
    count_urdu = 0
    count_english = 0
    regions_failed_ocr = 0
    pages_skipped = 0

    for item in results:
        if not isinstance(item, dict):
            continue
        regions = item.get("regions", [])
        if not regions:
            pages_skipped += 1
            continue
        page_has_any_success = False
        for r in regions:
            if not isinstance(r, dict):
                continue
            total_regions += 1
            lang = (r.get("language") or "").strip().lower()
            text = (r.get("text") or "").strip()
            if _is_failed_or_empty(text):
                regions_failed_ocr += 1
            else:
                regions_with_text += 1
                page_has_any_success = True
            if lang == "urdu":
                count_urdu += 1
            elif lang == "english":
                count_english += 1
        if not page_has_any_success:
            pages_skipped += 1

    empty_regions = total_regions - regions_with_text
    num_pages = len(results)
    total_time = timings.get("total_time")
    if timings.get("pages_skipped") is not None:
        pages_skipped = int(timings["pages_skipped"])

    metrics = {
        "ocr_coverage": {
            "total_regions": total_regions,
            "regions_with_text": regions_with_text,
            "empty_regions": empty_regions,
        },
        "script_distribution": {
            "count_urdu": count_urdu,
            "count_english": count_english,
        },
        "failure_metrics": {
            "regions_failed_ocr": regions_failed_ocr,
            "pages_skipped": pages_skipped,
        },
    }

    if total_time is not None:
        metrics["runtime_metrics"] = {
            "total_time_seconds": round(total_time, 2),
            "avg_time_per_page_seconds": round(total_time / num_pages, 2) if num_pages else 0,
        }
    else:
        metrics["runtime_metrics"] = {
            "total_time_seconds": None,
            "avg_time_per_page_seconds": None,
        }

    metrics["summary"] = {
        "total_pages": num_pages,
    }
    return metrics


def print_evaluation_report(metrics: dict[str, Any]) -> None:
    """Print a clean, examiner-friendly console summary."""
    if "error" in metrics:
        print(f"Error: {metrics['error']}", file=sys.stderr)
        return

    cov = metrics.get("ocr_coverage", {})
    script = metrics.get("script_distribution", {})
    fail = metrics.get("failure_metrics", {})
    runtime = metrics.get("runtime_metrics", {})
    summary = metrics.get("summary", {})

    print()
    print("=" * 50)
    print("  Bilingual OCR Evaluation Report")
    print("=" * 50)
    print()
    print("  OCR Coverage")
    print("  -----------")
    print(f"    total_regions      : {cov.get('total_regions', 0)}")
    print(f"    regions_with_text  : {cov.get('regions_with_text', 0)}")
    print(f"    empty_regions      : {cov.get('empty_regions', 0)}")
    print()
    print("  Script Distribution")
    print("  ------------------")
    print(f"    Urdu regions       : {script.get('count_urdu', 0)}")
    print(f"    English regions    : {script.get('count_english', 0)}")
    print()
    print("  Runtime Metrics")
    print("  ---------------")
    total_time = runtime.get("total_time_seconds")
    avg_time = runtime.get("avg_time_per_page_seconds")
    if total_time is not None:
        print(f"    total_time (s)      : {total_time}")
        print(f"    avg_time_per_page  : {avg_time} s")
    else:
        print("    (not provided)")
    print()
    print("  Failure Metrics")
    print("  ----------------")
    print(f"    regions_failed_ocr : {fail.get('regions_failed_ocr', 0)}")
    print(f"    pages_skipped      : {fail.get('pages_skipped', 0)}")
    print()
    print("  Summary")
    print("  -------")
    print(f"    total_pages        : {summary.get('total_pages', 0)}")
    print()
    print("=" * 50)


def save_evaluation_report(
    metrics: dict[str, Any],
    output_path: str | Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Save metrics to a JSON file. Creates parent dirs if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate bilingual OCR output (lightweight metrics, no ground truth).",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        metavar="JSON",
        help="Path to structured JSON from bilingual_ocr.py (list of {page, regions}).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        metavar="FILE",
        help=f"Path for evaluation JSON report (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Optional: total runtime in seconds (for runtime metrics).",
    )
    parser.add_argument(
        "--pages-skipped",
        type=int,
        default=None,
        metavar="N",
        help="Optional: number of pages skipped (overrides derived value).",
    )
    args = parser.parse_args()

    timings = {}
    if args.total_time is not None:
        timings["total_time"] = args.total_time
    if args.pages_skipped is not None:
        timings["pages_skipped"] = args.pages_skipped

    metrics = evaluate_bilingual_output(args.input, timings if timings else None)
    print_evaluation_report(metrics)

    if "error" not in metrics:
        path = save_evaluation_report(metrics, args.output)
        print(f"Report saved: {path}")


if __name__ == "__main__":
    _main()
