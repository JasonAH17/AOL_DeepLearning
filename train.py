from ultralytics import YOLO
import torch

def main():
    # Verify GPU availability
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("WARNING: GPU not available. Training will be slow.")
        device = 'cpu'

    # Load model
    # Using yolov8n.pt (nano) for faster training, can switch to yolov8s.pt or yolov8m.pt for better accuracy
    torch.backends.cudnn.enabled = False 
    model = YOLO('yolov8n.pt') 

    # Train
    print("Starting training...")
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        project='runs/detect',
        name='helmet_yolov8_run',
        exist_ok=True  # Allow overwriting existing project/name
    )
    
    print("Training complete.")

if __name__ == '__main__':
    main()
