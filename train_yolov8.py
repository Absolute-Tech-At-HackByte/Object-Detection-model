from ultralytics import YOLO
import os
import yaml

# Create dataset.yaml file
def create_yaml():
    yaml_content = {
        'path': os.path.abspath('data'),  # Path to data directory
        'train': 'train/images',  # Path to train images
        'val': 'val/images',      # Path to val images
        'test': 'test/images',    # Path to test images
        'names': {
            0: 'ordinary',
            1: 'hsrp'
        }
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print("Created dataset.yaml file")

# Function to train the model
def train_model(model_size='n', epochs=50, batch_size=16, imgsz=640):
    """
    Train a YOLOv8 model
    Args:
        model_size: Size of YOLOv8 model (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Batch size
        imgsz: Image size
    """
    # Create YAML file
    create_yaml()
    
    # Load the model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train the model
    results = model.train(
        data='dataset.yaml',
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=10,         # Early stopping patience
        save=True,           # Save checkpoints
        device='0',          # GPU device (use '0' for first GPU, 'cpu' for CPU)
        project='license_plate_detection',
        name=f'yolov8{model_size}_plates',
        pretrained=True,     # Use pretrained weights
        optimizer='AdamW',   # Optimizer
        lr0=0.001,           # Initial learning rate
        lrf=0.01,            # Final learning rate factor
        augment=True,        # Use augmentation
        cache=False,         # Cache images for faster training
    )
    
    return results

if __name__ == "__main__":
    # Train model with YOLOv8n (nano) as a starting point
    # You can change to 's', 'm', 'l', or 'x' for larger models
    print("Starting training with YOLOv8n...")
    results = train_model(model_size='n', epochs=100, batch_size=16)
    print(f"Training completed. Results saved in {results.save_dir}")
    
    # Evaluate the model on the test set
    print("Evaluating model on test set...")
    metrics = YOLO(f'license_plate_detection/yolov8n_plates/weights/best.pt').val(data='dataset.yaml')
    print(f"Test metrics: {metrics}") 