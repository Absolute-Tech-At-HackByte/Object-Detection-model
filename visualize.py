import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import os
import yaml
from ultralytics import YOLO
import argparse
from matplotlib import rcParams
import torch

# Increase font size for better readability
rcParams['font.size'] = 12

def visualize_predictions(model_path, test_dir, num_images=4, conf=0.25, device='cpu'):
    """Visualize model predictions on test images"""
    # Load model
    model = YOLO(model_path)
    print(f"Loaded model from {model_path}")
    print(f"Using device: {device}")
    
    # Get test images
    image_files = list(Path(test_dir).glob('*.jpg')) + list(Path(test_dir).glob('*.png')) + list(Path(test_dir).glob('*.jpeg'))
    print(f"Found {len(image_files)} test images")
    
    if len(image_files) == 0:
        print(f"No images found in {test_dir}")
        return
    
    # Randomly select images
    if num_images > len(image_files):
        num_images = len(image_files)
    
    np.random.shuffle(image_files)
    selected_images = image_files[:num_images]
    
    # Calculate grid dimensions
    cols = 2
    rows = (num_images + 1) // cols
    
    # Create grid for visualization
    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1 and cols == 1:
        axs = np.array([axs])
    axs = axs.flatten()
    
    # Add a title to the figure
    fig.suptitle('License Plate Detection Results', fontsize=16)
    
    # Class names for color coding
    class_names = ['ordinary', 'hsrp']
    class_colors = ['green', 'blue']
    
    # Make predictions and display
    for i, img_path in enumerate(selected_images):
        if i >= len(axs):
            break
            
        # Run prediction on specified device
        results = model.predict(source=img_path, conf=conf, device=device, verbose=False)
        
        # Get result image with annotations
        result_img = results[0].plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Display in grid
        ax = axs[i]
        ax.imshow(result_img)
        ax.set_title(f"Image: {os.path.basename(img_path)}")
        
        # Add detections as text
        text_str = ""
        if len(results[0].boxes) > 0:
            for j, box in enumerate(results[0].boxes):
                cls_id = int(box.cls)
                conf = float(box.conf)
                class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                text_str += f"{class_name}: {conf:.2f}\n"
            
            # Add detection info
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
        ax.axis('off')
    
    # Hide extra subplots if needed
    for i in range(num_images, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust for the title
    
    # Save the visualization
    output_dir = Path("predictions")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "visualization.png", dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_dir / 'visualization.png'}")
    
    plt.show()

def visualize_confusion_matrix(model_path, data_yaml, device='cpu'):
    """Visualize confusion matrix for the model"""
    # Load model
    model = YOLO(model_path)
    
    # Run validation with confusion matrix
    results = model.val(data=data_yaml, device=device, verbose=False, plots=True)
    
    print(f"Evaluation complete. Check {Path(model_path).parent.parent / 'val'} for confusion matrix.")

if __name__ == '__main__':
    # Check if CUDA is available
    if torch.cuda.is_available():
        DEFAULT_DEVICE = '0'  # Use GPU if available
        print("CUDA is available! Using GPU for visualization.")
    else:
        DEFAULT_DEVICE = 'cpu'  # Fall back to CPU
        print("CUDA is not available. Using CPU for visualization.")
        
    parser = argparse.ArgumentParser(description="Visualize model predictions")
    parser.add_argument('--num_images', type=int, default=4, help='Number of images to visualize')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, help='Device to use (0 for GPU, cpu for CPU)')
    parser.add_argument('--source', type=str, default=None, help='Source directory with images')
    parser.add_argument('--conf_matrix', action='store_true', help='Generate confusion matrix')
    args = parser.parse_args()
    
    this_dir = Path(__file__).parent
    
    # Load configuration
    with open(this_dir / 'yolo_params.yaml', 'r') as file:
        data = yaml.safe_load(file)
        test_dir = Path(data['test'])
    
    # Override with command line source if provided
    if args.source is not None:
        test_dir = Path(args.source)
    
    # Load model
    detect_path = this_dir / "runs" / "detect"
    
    if not detect_path.exists():
        print(f"No detection runs found at {detect_path}. Please train the model first.")
        exit()
    
    train_folders = [f for f in os.listdir(detect_path) if os.path.isdir(detect_path / f) and f.startswith("train")]
    if len(train_folders) == 0:
        raise ValueError("No training folders found")
    
    # Use the latest training folder
    latest_folder = sorted(train_folders)[-1]
    model_path = detect_path / latest_folder / "weights" / "best.pt"
    
    print(f"Found model: {model_path}")
    
    # Visualize predictions
    visualize_predictions(model_path, test_dir, args.num_images, args.conf, args.device)
    
    # If requested, also generate confusion matrix
    if args.conf_matrix:
        visualize_confusion_matrix(model_path, this_dir / 'yolo_params.yaml', args.device) 