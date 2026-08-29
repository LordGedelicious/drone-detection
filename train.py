import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split

from src.core.dataset import DroneDataset, get_train_transforms, get_val_transforms, collate_fn
from src.core.loss import DetectionLoss
from src.models.single_scale import SingleScaleDetector
from src.models.multi_scale_fpn import MultiScaleFPNDetector
from src.models.p2_granular import P2GranularDetector
from src.models.ab2d_yolo import AB2DYOLO
from src.models.yolov26 import YOLOv26Benchmark
from src.engine.singlegpu_trainer import SingleGPUTrainer
from src.engine.wnb_tracker import WandbTracker

def parse_args():
    parser = argparse.ArgumentParser()
    # Default set to the Single-Scale Baseline; yolo26 added to choices
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "fpn", "p2", "ab2d", "yolo26"], help="Select model architecture.")
    
    # Removed leading slashes so paths resolve relative to your repository root
    parser.add_argument("--img-dir", type=str, default="data/images", help="Path to images directory.")
    parser.add_argument("--lbl-dir", type=str, default="data/labels", help="Path to labels directory.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--loss-type", type=str, default="iciou", choices=["iou", "ciou", "eiou", "siou", "iciou"], help="Bounding box regression loss type.")
    
    # YOLOv26 specific argument
    parser.add_argument("--yolo-data", type=str, default="drone_data.yaml", help="Path to yaml config for YOLOv26 benchmark.")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Routing for Ultralytics YOLOv26 Benchmark (Bypasses custom trainer)
    if args.model == "yolo26":
        print(f"Launching YOLOv26 Benchmark Training for {args.epochs} epochs...")
        model = YOLOv26Benchmark(model_weight="yolov8n.pt", num_classes=1)
        model.train_benchmark(
            data_yaml=args.yolo_data,
            epochs=args.epochs,
            imgsz=640,
            batch=args.batch_size,
            project="drone-detection",
            name="yolov26_benchmark"
        )
        return

    # 2. Dynamic Dataset Splitting (80/20) for Custom Models
    base_dataset = DroneDataset(args.img_dir, args.lbl_dir, transforms=None)
    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size
    train_idx, val_idx = random_split(range(len(base_dataset)), [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_dataset = Subset(DroneDataset(args.img_dir, args.lbl_dir, transforms=get_train_transforms(640)), train_idx)
    val_dataset = Subset(DroneDataset(args.img_dir, args.lbl_dir, transforms=get_val_transforms(640)), val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # 3. Dynamic Model & Loss Configuration
    if args.model == "baseline":
        model = SingleScaleDetector(num_classes=1)
        scale_ranges = [(0.0, 1.0)]  # 1 Scale
    elif args.model == "fpn":
        model = MultiScaleFPNDetector(num_classes=1)
        scale_ranges = [(0.0, 0.10), (0.08, 0.25), (0.20, 1.00)]  # 3 Scales (P3, P4, P5)
    elif args.model == "p2":
        model = P2GranularDetector(num_classes=1)
        scale_ranges = [(0.0, 0.05), (0.04, 0.12), (0.10, 0.28), (0.25, 1.00)]  # 4 Scales (P2, P3, P4, P5)
    elif args.model == "ab2d":
        model = AB2DYOLO(num_classes=1)
        scale_ranges = [(0.0, 0.05), (0.04, 0.12), (0.10, 0.28), (0.25, 1.00)]  # 4 Scales

    criterion = DetectionLoss(loss_type=args.loss_type, scale_ranges=scale_ranges)

    # 4. Optimizer & Tracking
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tracker = WandbTracker(project_name="drone-detection", run_name=f"{args.model}_run", config=vars(args))

    # 5. Launch Custom Training
    trainer = SingleGPUTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs),
        tracker=tracker,
        model_name=args.model
    )
    
    trainer.fit(epochs=args.epochs)

if __name__ == "__main__":
    main()