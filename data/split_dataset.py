import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "extracted_raw/curated_datasets/obj_det_base"
IMG_DIR = BASE_DIR / "images"
LBL_DIR = BASE_DIR / "labels"

IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

img_count = 0
lbl_count = 0

for file_path in RAW_DIR.rglob("*"):
    if file_path.is_file():
        ext = file_path.suffix.lower()
        if ext == ".png":
            shutil.move(str(file_path), str(IMG_DIR / file_path.name))
            img_count += 1
        elif ext == ".txt":
            shutil.move(str(file_path), str(LBL_DIR / file_path.name))
            lbl_count += 1

print(f"Moved {img_count} images")
print(f"Moved {lbl_count} labels")