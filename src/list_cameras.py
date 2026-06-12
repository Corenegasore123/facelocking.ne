"""
Probe OpenCV camera indices and report which ones return frames.

Usage (from repo root):
    python -m src.list_cameras
    python -m src.list_cameras --max-index 8
"""
from __future__ import annotations

import argparse
import sys

import cv2


def probe_index(index: int, backend: int | None, warmup_reads: int) -> dict:
    if backend is None:
        cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index, backend)

    result = {
        "index": index,
        "opened": False,
        "readable": False,
        "width": None,
        "height": None,
        "backend": backend,
        "error": None,
    }

    if not cap.isOpened():
        cap.release()
        return result

    result["opened"] = True

    try:
        for _ in range(warmup_reads):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                result["readable"] = True
                result["height"], result["width"] = frame.shape[:2]
                break
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        cap.release()

    return result


def backend_label(backend: int | None) -> str:
    if backend is None:
        return "default"
    if backend == cv2.CAP_DSHOW:
        return "CAP_DSHOW"
    if backend == cv2.CAP_MSMF:
        return "CAP_MSMF"
    return str(backend)


def main() -> int:
    parser = argparse.ArgumentParser(description="List working OpenCV camera indices.")
    parser.add_argument("--max-index", type=int, default=6, help="Highest index to probe (inclusive).")
    parser.add_argument("--warmup-reads", type=int, default=5, help="Frame reads per opened device.")
    parser.add_argument(
        "--backend",
        choices=["default", "dshow", "msmf", "all"],
        default="all",
        help="Video backend to use on Windows (all tries default, dshow, then msmf).",
    )
    args = parser.parse_args()

    if args.backend == "default":
        backends: list[int | None] = [None]
    elif args.backend == "dshow":
        backends = [cv2.CAP_DSHOW]
    elif args.backend == "msmf":
        backends = [cv2.CAP_MSMF]
    else:
        backends = [None, cv2.CAP_DSHOW, cv2.CAP_MSMF]

    print("Probing camera indices...")
    print(f"Range: 0..{args.max_index}")
    print()

    working: list[tuple[int, int | None]] = []

    for backend in backends:
        label = backend_label(backend)
        print(f"Backend: {label}")
        print("-" * 56)
        for index in range(0, args.max_index + 1):
            result = probe_index(index, backend, max(1, args.warmup_reads))
            if not result["opened"]:
                print(f"  [{index}] not opened")
                continue
            if result["readable"]:
                w, h = result["width"], result["height"]
                print(f"  [{index}] OK  {w}x{h}")
                working.append((index, backend))
            else:
                err = f" ({result['error']})" if result["error"] else ""
                print(f"  [{index}] opened but no frame{err}")
        print()

    if working:
        print("Working cameras:")
        for index, backend in working:
            print(f"  --camera-index {index}  (backend: {backend_label(backend)})")
        best_index, best_backend = working[0]
        print()
        print("Suggested command:")
        if best_backend == cv2.CAP_DSHOW:
            print(f"  set OPENCV_VIDEOIO_PRIORITY_MSMF=0")
        print(f"  python -m src.enroll   # then update index to {best_index} if needed")
        return 0

    print("No working camera found.")
    print("Checks:")
    print("  - Close apps using the camera (Teams, Zoom, Camera app)")
    print("  - Replug the USB camera")
    print("  - Try: python -m src.list_cameras --max-index 10 --backend dshow")
    return 1


if __name__ == "__main__":
    sys.exit(main())
