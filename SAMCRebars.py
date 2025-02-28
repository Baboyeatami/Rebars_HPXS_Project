from ultralytics import YOLO

# Load the YOLOv11 model
model = YOLO("runs/detect/train4/weights/last.pt")

# Train the model with optimized parameters
train_results = model.train(
    data="data.yaml",  # Path to dataset YAML
    epochs=100,        # Number of training epochs
    imgsz=512,         # Reduced training image size (lower memory usage)
    batch=2,           # Reduced batch size to prevent OOM (Out of Memory)
    workers=1,         # Use fewer workers to prevent CPU overload
    device="cpu",      # Run on CPU
    cache=False,       # Disable caching to save memory
    amp=False,         # Disable Automatic Mixed Precision
    resume=True
    
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # Return path to exported model
