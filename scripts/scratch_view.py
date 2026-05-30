# scratch_view.py
import random
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
    
from src.synthetic_images import ContainerImageGenerator, CLEAN, MODERATE, HARSH, save_image

gen = ContainerImageGenerator(rng=random.Random(0))
for i, (name, cfg) in enumerate([("clean", CLEAN), ("moderate", MODERATE), ("harsh", HARSH)]):
    img = gen.generate(cfg)
    save_image(img, f"sample_{name}.png")
    print(f"{name}: code = {img.true_code}")