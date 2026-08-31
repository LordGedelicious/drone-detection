import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.core.split import scene_of, condition_of

# OpenCV spawns worker threads by default; inside a multi-worker DataLoader that
# oversubscribes every CPU core. Pin it to 1 -- parallelism comes from the
# DataLoader workers, not from within each decode.
cv2.setNumThreads(0)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# OOP class to store dataset, handle data loading, and apply transformations
class DroneDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transforms, image_files=None):
        super().__init__()
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.transforms = transforms
        # When an explicit file list is supplied (e.g. from src.core.split) use it
        # verbatim so train/val/test membership is controlled by the caller.
        # Otherwise fall back to a *sorted* directory listing for determinism.
        if image_files is not None:
            self.image_files = list(image_files)
        else:
            self.image_files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".png"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base_name = os.path.splitext(img_name)[0]
        lbl_name = f"{base_name}.txt"

        img_path = os.path.join(self.img_dir, img_name)
        lbl_path = os.path.join(self.lbl_dir, lbl_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV BGR -> RGB for PyTorch

        bboxes = []
        class_ids = []
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            with open(lbl_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_ids.append(int(parts[0]))  # all zeros: drone = 0
                        # YOLO format: [x_center, y_center, width, height] (normalized)
                        bboxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=bboxes, class_labels=class_ids)
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_ids = transformed['class_labels']
        else:
            # Safeguard path: keep it consistent with the val pipeline (normalized).
            image = A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])(image=image)['image']

        target = {
            "boxes": torch.as_tensor(bboxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(class_ids, dtype=torch.int64).reshape(-1),
            # Per-sample provenance -- used for per-condition evaluation and for
            # mapping predictions back to source files during inference.
            "meta": {
                "file": img_name,
                "scene": scene_of(img_name),
                "condition": condition_of(img_name),
                "augmented": img_name.startswith("augmented_"),
            },
        }
        return image, target


def collate_fn(batch):
    # Images are all letterboxed to a fixed square, so they stack directly.
    # Targets stay a list because the number of drones per image varies.
    images, targets = list(zip(*batch))
    images = torch.stack(images, dim=0)
    return images, list(targets)


def _letterbox(img_size):
    """Scale the longest edge to img_size, then pad to a square with black
    borders. Same geometry for train / val / test / inference."""
    return [
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
        ),
    ]


def get_train_transforms(img_size: int = 640) -> A.Compose:
    return A.Compose(
        [
            *_letterbox(img_size),

            # Geometry: drones appear at many positions / slight orientations.
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),

            # Photometric: the data spans sunny / foggy conditions.
            # CLAHE ref: https://medium.com/imagecraft/histogram-equalization-clahe-algorithm-8841d402fc76
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
            # City edges can mimic drone edges -- perturb hue/sat a little.
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),

            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_area=4.0,        # drop boxes below ~2x2 px after transform
            min_visibility=0.25,
        ),
    )


def get_val_transforms(img_size: int = 640) -> A.Compose:
    return A.Compose(
        [
            *_letterbox(img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_area=0.0,
            min_visibility=0.0,
        ),
    )
