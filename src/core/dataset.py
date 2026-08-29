import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# OOP class to store dataset, handle data loading, and apply transformations
class DroneDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transforms):
        super().__init__()
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith(('.png'))]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base_name = os.path.splitext(img_name)[0]
        lbl_name = f"{base_name}.txt"
        
        img_path = os.path.join(self.img_dir, img_name)
        lbl_path = os.path.join(self.lbl_dir, lbl_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Converts OpenCV BGR to RGB for PyTorch
        
        bboxes = []
        class_ids = []
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            with open(lbl_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_ids.append(int(parts[0])) # Should be all zeros, drone = 0
                        # YOLO format: [x_center, y_center, width, height]
                        bboxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        
        # Albumentations for data augmentation and transformation, apparently standard for object detection problems and better than torchvision
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=bboxes, class_labels=class_ids)
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_ids = transformed['class_labels']
        else:
            # Fallback to base tensor conversion for safeguard
            image = ToTensorV2()(image=image)['image'] / 255.0

        target = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(class_ids, dtype=torch.int64)
        }
        
        return image, target

def collate_fn(batch):
    # To ensure each batch has the same number of images and drones, since manual sampling found that at least there are 
    # 2 images where one has 2 drones, and the other only has one.
    images, targets = list(zip(*batch))
    images = torch.stack(images, dim=0)
    return images, list(targets)

def get_train_transforms(img_size: int = 640) -> A.Compose:
    return A.Compose(
        [
            # Basic geometric transformations + shift, scale, rotate since the drones 
            # appear in many orientations and positions in the sampled images
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=0,
                p=0.5
            ),
            
            # Applying CLAHE (Contrast Limited Adaptive Histogram Equalization) and random brightness/contrast adjustments 
            # should help since the data are split into either sunny or foggy
            # Source: https://medium.com/imagecraft/histogram-equalization-clahe-algorithm-8841d402fc76
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
            
            # Specifically for the city images since edges of the buildings can be similar to the drone edges
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),
            
            # Normalization based on Mean and STD values of ImageNet-1k Dataset
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

            # Convert the image to a PyTorch tensor
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_area=4.0,       # Filter out bboxes smaller than 2x2 px, too small for the model to learn probably
            min_visibility=0.25 # Will be adjusted later during training
        )
    )

def get_val_transforms(img_size: int = 640) -> A.Compose:
    return A.Compose(
        [
            # Reproduce transformations for validation without the augmentations
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_area=0.0,
            min_visibility=0.0
        )
    )