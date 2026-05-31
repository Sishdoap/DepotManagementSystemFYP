"""Run the real-OCR + real-images simulator for the dashboard demo.

Uses your CCLN + PaddleOCR pipeline on actual container photos from images/.
This is the demo configuration; the fast-mode evaluation in scripts/
run_evaluation.py keeps using MockOCRAdapter + SyntheticImageSource.
"""

import random
import sys
import time
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.dashboard_io import clear_frames
from src.db import drop_schema, init_schema, make_engine
from src.image_source import RealImageSource
from src.ocr import RealOCRAdapter
from src.router import UnstrictFIFORouter
from src.simulation import DepotSimulation, SimulationConfig


DB_PATH = "depot_live.db"
IMAGES_DIR = "images"

# Real OCR takes ~4s per truck. We want the demo brisk but not so fast that
# the OCR can't keep up. With factor=0.2 (5x real-time), gates clear quickly
# while OCR has time to actually run.
REALTIME_FACTOR = 0.2


def main():
    db_url = f"sqlite:///{DB_PATH}"
    engine = make_engine(db_url)
    drop_schema(engine)
    init_schema(engine)
    clear_frames()

    image_source = RealImageSource(IMAGES_DIR, rng=random.Random(0))
    print(f"Loaded {image_source.n_images} images from {IMAGES_DIR}/")

    print("Initializing CCLN + PaddleOCR (may take a moment)...")
    t0 = time.time()
    ocr = RealOCRAdapter(
        ccln_weights_path="models/ccln.pth",
        device="cpu",
        paddle_gpu=False,
        use_localization=True,
        try_rotations=True,
    )
    print(f"OCR ready in {time.time() - t0:.1f}s")

    config = SimulationConfig(
        duration_seconds=86400,
        arrival_rate_per_minute=2.0,
        seed=0,
    )

    sim = DepotSimulation(
        config=config,
        router=UnstrictFIFORouter(alpha=0.7),
        ocr=ocr,
        engine=engine,
        image_source=image_source,
        fast_mode=False,
        realtime_factor=REALTIME_FACTOR,
        share_frames=True,
    )

    print(f"\nRunning live sim. DB: {DB_PATH}. Frames: ./frames/")
    print("In another terminal: streamlit run scripts/dashboard.py")
    print("Ctrl-C to stop.\n")
    try:
        sim.run()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()