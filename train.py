import argparse
from ultralytics import YOLO
import os
import sys
import torch

# Define constants - optimized for accuracy
EPOCHS = 150  # Increase epochs for better learning
MOSAIC = 0.6  # Slight increase in augmentation
OPTIMIZER = 'AdamW'  # Best optimizer for accuracy
MOMENTUM = 0.937
LR0 = 0.001
LRF = 0.01
SINGLE_CLS = False
IMGSZ = 640  # Higher resolution
BATCH = 16   # Adjusted batch size for GPU

# Check if CUDA is available
if torch.cuda.is_available():
    DEFAULT_DEVICE = '0'  # Use GPU if available
    print("CUDA is available! Using GPU for training.")
else:
    DEFAULT_DEVICE = 'cpu'  # Fall back to CPU
    print("CUDA is not available. Using CPU for training.")
    # Reduce batch size for CPU training
    BATCH = 8

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    # epochs
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    # mosaic
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    # optimizer
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    # momentum
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    # lr0
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    # lrf
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    # single_cls
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')
    # imgsz
    parser.add_argument('--imgsz', type=int, default=IMGSZ, help='Image size')
    # batch
    parser.add_argument('--batch', type=int, default=BATCH, help='Batch size')
    # Use GPU
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, help='Device to use (0 for GPU, cpu for CPU)')
    args = parser.parse_args()
    
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    
    # Use YOLOv8s (small) for better accuracy with reasonable speed
    model = YOLO(os.path.join(this_dir, "yolov8s.pt"))
    
    print(f"Training YOLOv8s model on {args.device}")
    print(f"Training for {args.epochs} epochs with batch size {args.batch}")
    
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"), 
        epochs=args.epochs,
        device=args.device,  # Use detected device
        single_cls=args.single_cls, 
        mosaic=args.mosaic,
        optimizer=args.optimizer, 
        lr0=args.lr0, 
        lrf=args.lrf, 
        momentum=args.momentum,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=20,  # More patience for better convergence
        augment=True, # Enable other augmentations
        cos_lr=True,  # Use cosine LR scheduler for better accuracy
        mixup=0.1,    # Slight mixup for better generalization
        hsv_h=0.015,  # HSV augmentations for better robustness
        hsv_s=0.7,
        hsv_v=0.4,
        cache=True,   # Cache images for faster training
        save=True     # Save best model
    )
    
    # Print final metrics
    print(f"Training completed. Best mAP50-95: {results.results_dict['metrics/mAP50-95(B)']}") 
    print(f"Best mAP50: {results.results_dict['metrics/mAP50(B)']}")
    
    # Validate on test set
    metrics = model.val(data=os.path.join(this_dir, "yolo_params.yaml"), split="test")
    print("\n=== TEST SET EVALUATION ===")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"Precision: {metrics.box.p}")
    print(f"Recall: {metrics.box.r}")
