"""
OCR Image Preprocessing Module (Step 1 of pipeline).

OpenCV-only preprocessing for scanned documents and handwritten notes.
No deep learning. Windows-compatible.

Pipeline: load -> grayscale -> denoise -> adaptive threshold -> deskew.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants (tunable for low-quality scans and handwritten text)
# ---------------------------------------------------------------------------
BLUR_KERNEL_SIZE = 3          # Median blur kernel; odd, 3 or 5 for noise removal
ADAPTIVE_BLOCK_SIZE = 15     # Block size for adaptive threshold (odd, ~11–31)
ADAPTIVE_C = 8               # Constant subtracted from mean in adaptive threshold
MIN_CONTOUR_AREA = 500       # Min contour area to consider for deskew (filter noise)
MAX_CONTOUR_AREA_RATIO = 0.5 # Max contour area as ratio of image (filter full-page)
ANGLE_QUANTILE = 0.5         # Median angle for deskew (0.5 = median)


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load an image from disk. Supports common formats (PNG, JPG, TIFF, etc.).

    Args:
        image_path: Path to the image file.

    Returns:
        BGR image as numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the image could not be loaded (e.g. corrupt or unsupported).
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image (unsupported or corrupt): {path}")
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to grayscale. Single channel improves thresholding and deskew.

    Args:
        image: BGR or grayscale image.

    Returns:
        Grayscale image (uint8).
    """
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray, kernel_size: int = BLUR_KERNEL_SIZE) -> np.ndarray:
    """
    Reduce noise using median blur. Preserves edges better than Gaussian for
    text and handwritten strokes; helps with low-quality scans.

    Args:
        image: Grayscale image.
        kernel_size: Odd kernel size (3 or 5 typical).

    Returns:
        Denoised grayscale image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


def adaptive_threshold(
    image: np.ndarray,
    block_size: int = ADAPTIVE_BLOCK_SIZE,
    c: int = ADAPTIVE_C,
) -> np.ndarray:
    """
    Binarize image with adaptive thresholding for uneven lighting (e.g. scans
    with shadows or non-uniform illumination). Each pixel is compared to a
    local mean.

    Args:
        image: Grayscale image.
        block_size: Size of neighbourhood (must be odd).
        c: Constant subtracted from the mean.

    Returns:
        Binary image (0 or 255); text usually white on black for many OCR APIs.
    """
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )
    return binary


def _get_skew_angle_from_binary(
    binary: np.ndarray,
    min_area: int = MIN_CONTOUR_AREA,
    max_area_ratio: float = MAX_CONTOUR_AREA_RATIO,
    quantile: float = ANGLE_QUANTILE,
) -> float:
    """
    Estimate skew angle (degrees) from binary image using contours.
    Uses minAreaRect of text-like contours; returns median angle so a few
    outliers (noise, graphics) do not dominate.

    Args:
        binary: Binary image (0 and 255).
        min_area: Ignore contours smaller than this.
        max_area_ratio: Ignore contours larger than this fraction of image area.
        quantile: Which quantile of angles to use (0.5 = median).

    Returns:
        Estimated skew angle in degrees (positive = CCW tilt of text lines).
    """
    h, w = binary.shape
    total_area = h * w
    # Find contours (external only to get text block outlines)
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    angles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area_ratio * total_area:
            continue
        rect = cv2.minAreaRect(cnt)
        # rect[2] is angle in degrees in [-90, 0); we use it for skew
        angle = rect[2]
        # Normalize to small tilt: prefer angle in [-45, 45]
        if angle < -45:
            angle += 90
        angles.append(angle)
    if not angles:
        return 0.0
    return float(np.quantile(angles, quantile))


def deskew(
    image: np.ndarray,
    binary: np.ndarray,
    skew_angle: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Rotate image and binary to correct skew. If skew_angle is not provided,
    it is estimated from the binary image.

    Args:
        image: Grayscale image to deskew.
        binary: Binary image used for angle estimation (and to deskew).
        skew_angle: Override estimated angle (degrees); if None, estimate from binary.

    Returns:
        (deskewed_grayscale, deskewed_binary, angle_used)
    """
    if skew_angle is None:
        skew_angle = _get_skew_angle_from_binary(binary)
    # Only rotate if angle is meaningful (avoid jitter on already straight docs)
    if abs(skew_angle) < 0.2:
        return image.copy(), binary.copy(), skew_angle

    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, -skew_angle, 1.0)
    # Expand canvas so rotated image is not cropped
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw / 2) - center[0]
    M[1, 2] += (nh / 2) - center[1]
    gray_deskewed = cv2.warpAffine(
        image, M, (nw, nh),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    binary_deskewed = cv2.warpAffine(
        binary, M, (nw, nh),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return gray_deskewed, binary_deskewed, skew_angle


def preprocess_image(image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline: load -> grayscale -> denoise -> adaptive
    threshold -> deskew. Suitable for scanned documents and handwritten notes.

    Args:
        image_path: Path to the input image.

    Returns:
        (binary_image, deskewed_grayscale_image)
        - binary_image: Final black & white image (deskewed), for OCR input.
        - deskewed_grayscale: Deskewed grayscale, for debugging/visualization.

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If image cannot be loaded.
    """
    # 1. Load
    bgr = load_image(image_path)
    # 2. Grayscale
    gray = to_grayscale(bgr)
    # 3. Denoise (median blur)
    denoised = denoise(gray)
    # 4. Adaptive threshold -> binary
    binary = adaptive_threshold(denoised)
    # 5. Deskew (estimate angle from binary, then rotate both binary and grayscale)
    gray_deskewed, binary_deskewed, _ = deskew(denoised, binary)
    # Return final binary and deskewed grayscale (for debugging)
    return binary_deskewed, gray_deskewed


# ---------------------------------------------------------------------------
# Main: test block with display and save
# ---------------------------------------------------------------------------
def _run_demo(image_path: str | Path, output_path: str | Path | None) -> None:
    """
    Load a sample image, run preprocessing, show intermediate results in
    OpenCV windows, and save the final preprocessed image.
    """
    path = Path(image_path)
    if not path.is_file():
        print(f"Error: Sample image not found: {path}", file=sys.stderr)
        print("Usage: python preprocess.py <path_to_image> [output_path]", file=sys.stderr)
        sys.exit(1)

    # Load and pipeline (we need intermediates for display)
    bgr = load_image(path)
    gray = to_grayscale(bgr)
    denoised = denoise(gray)
    binary = adaptive_threshold(denoised)
    gray_deskewed, binary_deskewed, angle = deskew(denoised, binary)

    # Default output path if not provided
    if output_path is None:
        output_path = path.parent / f"{path.stem}_preprocessed.png"
    out = Path(output_path)

    # Save final binary (main output) and optionally deskewed grayscale
    cv2.imwrite(str(out), binary_deskewed)
    cv2.imwrite(str(out.parent / f"{out.stem}_gray.png"), gray_deskewed)
    print(f"Saved: {out}")
    print(f"Saved (grayscale): {out.parent / (out.stem + '_gray.png')}")
    print(f"Deskew angle (degrees): {angle:.2f}")

    # Display intermediate results (resize if very large so they fit on screen)
    max_display = 800
    def _resize_for_display(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) <= max_display:
            return img
        scale = max_display / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    cv2.imshow("1_original", _resize_for_display(bgr))
    cv2.imshow("2_grayscale", _resize_for_display(gray))
    cv2.imshow("3_thresholded", _resize_for_display(binary))
    cv2.imshow("4_deskewed_binary", _resize_for_display(binary_deskewed))
    cv2.imshow("5_deskewed_grayscale", _resize_for_display(gray_deskewed))
    print("Close any OpenCV window or press a key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a scanned/handwritten image for OCR (display + save).",
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to input image (e.g. scanned document or handwritten note).",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        nargs="?",
        default=None,
        help="Path for saved preprocessed image (default: <input_stem>_preprocessed.png).",
    )
    args = parser.parse_args()
    _run_demo(args.image_path, args.output_path)
