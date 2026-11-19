"""Download and verify that Ultralytics can load its official YOLO11l weights."""

import os
from pathlib import Path

from ultralytics import YOLO


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    model = YOLO("yolo11l.pt")
    print(f"Ready: {Path('yolo11l.pt').resolve()} ({len(model.names)} classes)")
