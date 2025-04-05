from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml
import argparse
import torch


# Function to predict and save images
def predict_and_save(model, image_path, output_path, output_path_txt, conf=0.5, device='cpu'):
    # Perform prediction with selected device
    results = model.predict(image_path, conf=conf, device=device)

    result = results[0]
    # Draw boxes on the image
    img = result.plot()  # Plots the predictions directly on the image

    # Save the result
    cv2.imwrite(str(output_path), img)
    # Save the bounding box data
    with open(output_path_txt, 'w') as f:
        for box in result.boxes:
            # Extract the class id and bounding box coordinates
            cls_id = int(box.cls)
            x_center, y_center, width, height = box.xywh[0].tolist()
            confidence = float(box.conf)
            
            # Write bbox information in the format [class_id, x_center, y_center, width, height, confidence]
            f.write(f"{cls_id} {x_center} {y_center} {width} {height} {confidence:.4f}\n")


if __name__ == '__main__': 
    # Check if CUDA is available
    if torch.cuda.is_available():
        DEFAULT_DEVICE = '0'  # Use GPU if available
        print("CUDA is available! Using GPU for prediction.")
    else:
        DEFAULT_DEVICE = 'cpu'  # Fall back to CPU
        print("CUDA is not available. Using CPU for prediction.")

    parser = argparse.ArgumentParser(description="Run prediction on images with trained model")
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, help='Device to use (0 for GPU, cpu for CPU)')
    parser.add_argument('--source', type=str, default=None, help='Source directory with images')
    args = parser.parse_args()

    this_dir = Path(__file__).parent
    os.chdir(this_dir)
    with open(this_dir / 'yolo_params.yaml', 'r') as file:
        data = yaml.safe_load(file)
        if 'test' in data and data['test'] is not None:
            images_dir = Path(data['test'])
        else:
            print("No test field found in yolo_params.yaml, please add the test field with the path to the test images")
            exit()
    
    # Override with command line source if provided
    if args.source is not None:
        images_dir = Path(args.source)
    
    # check that the images directory exists
    if not images_dir.exists():
        print(f"Images directory {images_dir} does not exist")
        exit()

    if not images_dir.is_dir():
        print(f"Images directory {images_dir} is not a directory")
        exit()
    
    if not any(images_dir.iterdir()):
        print(f"Images directory {images_dir} is empty")
        exit()

    # Load the YOLO model
    detect_path = this_dir / "runs" / "detect"
    
    if not detect_path.exists():
        print(f"No detection runs found at {detect_path}. Please train the model first.")
        exit()
        
    train_folders = [f for f in os.listdir(detect_path) if os.path.isdir(detect_path / f) and f.startswith("train")]
    if len(train_folders) == 0:
        raise ValueError("No training folders found")
    idx = 0
    if len(train_folders) > 1:
        choice = -1
        choices = list(range(len(train_folders)))
        while choice not in choices:
            print("Select the training folder:")
            for i, folder in enumerate(train_folders):
                print(f"{i}: {folder}")
            choice = input()
            if not choice.isdigit():
                choice = -1
            else:
                choice = int(choice)
        idx = choice

    model_path = detect_path / train_folders[idx] / "weights" / "best.pt"
    print(f"Loading model from {model_path}")
    
    # Load model on selected device
    model = YOLO(model_path)
    print(f"Running inference on {args.device} with confidence threshold {args.conf}")

    # Directory with images
    output_dir = this_dir / "predictions" # Replace with the directory where you want to save predictions
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create images and labels subdirectories
    images_output_dir = output_dir / 'images'
    labels_output_dir = output_dir / 'labels'
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through the images in the directory
    image_count = 0
    for img_path in images_dir.glob('*'):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
        output_path_img = images_output_dir / img_path.name  # Save image in 'images' folder
        output_path_txt = labels_output_dir / img_path.with_suffix('.txt').name  # Save label in 'labels' folder
        predict_and_save(model, img_path, output_path_img, output_path_txt, conf=args.conf, device=args.device)
        image_count += 1
        
    print(f"Processed {image_count} images")
    print(f"Predicted images saved in {images_output_dir}")
    print(f"Bounding box labels saved in {labels_output_dir}")
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    data_yaml = this_dir / 'yolo_params.yaml'
    metrics = model.val(data=data_yaml, split="test", device=args.device)
    
    print("\n=== EVALUATION METRICS ===")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.p:.4f}")
    print(f"Recall: {metrics.box.r:.4f}")
    print("==========================\n") 