"""Standalone OCR accuracy benchmark.

Runs the real OCR pipeline (CCLN + PaddleOCR + ISO 6346 recovery)
against every labeled image in images/ and reports accuracy.

Independent of the simulator and dashboard — just a direct
measurement of OCR pipeline quality against ground truth.

Usage:
    python scripts/benchmark_ocr.py
    python scripts/benchmark_ocr.py --max-images 300
    python scripts/benchmark_ocr.py --csv-output results/ocr_benchmark.csv
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from src.ocr import RealOCRAdapter


# Same regex as RealImageSource — extracts the ISO 6346 code from
# "0_TEMU6372145_1.jpg" style filenames.
_FILENAME_CODE_RE = re.compile(
    r"^(?:\d+_)?([A-Z]{4}\d{7})(?:_\d+)?\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)


def extract_ground_truth(filename: str) -> str | None:
    """Return the ISO 6346 code embedded in the filename, or None if unlabeled."""
    m = _FILENAME_CODE_RE.match(filename)
    return m.group(1).upper() if m else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        default="images",
        help="Directory containing labeled container images.",
    )
    parser.add_argument(
        "--ccln-weights",
        default="models/ccln.pth",
        help="Path to CCLN model weights.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit benchmark to N images. Default: all images in directory.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional: path to write per-image results as CSV.",
    )
    parser.add_argument(
        "--no-rotations",
        action="store_true",
        help="Skip 90/180/270° rotation search (4x faster but loses vertical codes).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda (for CCLN).",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"ERROR: {images_dir} does not exist.")
        sys.exit(1)

    # Discover all image files.
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.max_images:
        image_paths = image_paths[:args.max_images]

    print(f"Found {len(image_paths)} images in {images_dir}/")

    # Initialize OCR (slow — ~5–15s for first inference).
    print("Initializing CCLN + PaddleOCR (may take a moment)...")
    t0 = time.time()
    ocr = RealOCRAdapter(
        ccln_weights_path=args.ccln_weights,
        device=args.device,
        paddle_gpu=False,
        use_localization=True,
        try_rotations=not args.no_rotations,
    )
    print(f"OCR ready in {time.time() - t0:.1f}s\n")

    # Run benchmark.
    results = []
    n_labeled = 0
    n_unlabeled = 0
    n_correct = 0          # recovered code matches ground truth exactly
    n_iso_valid = 0        # any valid ISO 6346 output (may not match truth)
    n_failed = 0           # OCR produced no valid code at all
    total_latency = 0.0

    print(f"{'#':>4} {'File':<32} {'Truth':<12} {'OCR':<12} {'Result':<14}  {'Latency'}")
    print("-" * 90)

    for i, path in enumerate(image_paths, 1):
        truth = extract_ground_truth(path.name)
        if truth is None:
            n_unlabeled += 1
            continue
        n_labeled += 1

        try:
            pil = Image.open(path).convert("RGB")
            result = ocr.predict(pil)
        except Exception as e:
            print(f"{i:>4} {path.name:<32} {truth:<12} ERROR: {e}")
            results.append({
                "image": path.name,
                "ground_truth": truth,
                "raw_string": "",
                "recovered_code": "",
                "is_valid": False,
                "correct": False,
                "latency_ms": 0,
                "error": str(e),
            })
            n_failed += 1
            continue

        total_latency += result.latency_ms

        recovered = result.recovered_code or ""
        is_valid = result.is_valid
        correct = is_valid and recovered == truth

        if correct:
            n_correct += 1
        if is_valid:
            n_iso_valid += 1
        if not is_valid:
            n_failed += 1

        if correct:
            verdict = "✓ correct"
        elif is_valid:
            verdict = "~ valid≠truth"
        else:
            verdict = "✗ failed"

        print(
            f"{i:>4} {path.name[:30]:<32} {truth:<12} "
            f"{(recovered or '—'):<12} {verdict:<14}  {result.latency_ms:>6.0f}ms"
        )

        results.append({
            "image": path.name,
            "ground_truth": truth,
            "raw_string": result.raw_string,
            "recovered_code": recovered,
            "is_valid": is_valid,
            "correct": correct,
            "latency_ms": result.latency_ms,
            "error": "",
        })

    # Summary.
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total images discovered:    {len(image_paths)}")
    print(f"Unlabeled images skipped:   {n_unlabeled}")
    print(f"Labeled images evaluated:   {n_labeled}")
    print()

    if n_labeled > 0:
        accuracy = n_correct / n_labeled
        valid_rate = n_iso_valid / n_labeled
        valid_but_wrong = n_iso_valid - n_correct

        print(f"True accuracy (correct / labeled):              "
              f"{n_correct}/{n_labeled} = {accuracy:.1%}")
        print(f"ISO 6346 valid rate (passed checksum / labeled): "
              f"{n_iso_valid}/{n_labeled} = {valid_rate:.1%}")
        print(f"Valid but wrong (false confidence):              "
              f"{valid_but_wrong}/{n_labeled} = "
              f"{valid_but_wrong/n_labeled:.1%}")
        print(f"OCR failed (no valid code produced):             "
              f"{n_failed}/{n_labeled} = {n_failed/n_labeled:.1%}")
        print()
        print(f"Mean OCR latency:           {total_latency/n_labeled:.0f}ms per image")

    if args.csv_output:
        out_path = Path(args.csv_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote per-image results to {out_path}")


if __name__ == "__main__":
    main()