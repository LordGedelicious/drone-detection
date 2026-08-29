import os
import shutil
from pathlib import Path

RAW_DIR = Path("../data/extracted_raw")
IMG_DIR = Path("../data/images")
LBL_DIR = Path("../data/labels")

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