"""Create expected project folders (add models manually)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DIRS = [
    "config",
    "src/vision",
    "src/dashboard",
    "firmware/esp32/face_tracker",
    "dashboard",
    "data/db",
    "data/enroll",
    "data/logs",
    "diagrams",
    "models",
    "scripts",
]


def main() -> None:
    for rel in DIRS:
        (REPO_ROOT / rel).mkdir(parents=True, exist_ok=True)

    for name in ("embedder_arcface.onnx", "face_landmarker.task"):
        path = REPO_ROOT / "models" / name
        if not path.exists():
            print(f"  [missing] {path}")

    print(f"Ready: {REPO_ROOT}")


if __name__ == "__main__":
    main()
