import os
from pathlib import Path

def verify_dataset_classes(label_dir, expected_class=0): # 0 is drone supposedly
    unique_classes = set()
    anomalous_files = {}

    label_path = Path(label_dir)
    if not label_path.exists():
        print(f"Directory not found: {label_path.absolute()}")
        return

    # Iterate through all .txt files in the directory
    for txt_file in label_path.glob("*.txt"):
        with open(txt_file, "r") as f:
            for line_num, line in enumerate(f, start=1):
                parts = line.strip().split()
                
                # Skip empty lines
                if not parts:
                    continue
                
                try:
                    # YOLO format: [class_id, x_center, y_center, width, height]
                    class_id = int(parts[0])
                    unique_classes.add(class_id)
                except ValueError:
                    print(f"Malformed data in {txt_file.name} on line {line_num}: {line.strip()}")

    print(f"Scan complete. Unique class IDs found: {sorted(list(unique_classes))}")

if __name__ == "__main__":
    verify_dataset_classes("data/labels")