from ultralytics import YOLO
import cv2

# Load the trained YOLOv11 model using the best weights
model = YOLO("runs/detect/train4/weights/best.pt")  # Adjust path if needed

# Path to the image for inference
image_path = "test1.jpg"

# Perform object detection on the image
results = model(image_path)

# Show the results
for result in results:
    result.show()  # Display image with detections
    result.save("detection_output.jpg")  # Save the output image

# Optionally, save results in text format
for i, box in enumerate(results[0].boxes.xyxy):
    print(f"Detection {i + 1}: {box.tolist()}")  # Print detected bounding boxes
