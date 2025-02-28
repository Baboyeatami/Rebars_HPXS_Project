import cv2
import torch
from ultralytics import YOLO

# Load the trained YOLOv11 model with best weights
model = YOLO("runs/detect/train5/weights/best.pt")  # Update with your actual model path

# Path to the image for inference
image_path = "test3.png"  # Replace with your test image path

# Perform object detection
results = model(image_path)

# Load the original image
image = cv2.imread(image_path)

# Count the number of detected objects
num_objects = len(results[0].boxes)

# Define text properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 255, 0)  # Green text
thickness = 2

# Loop through detections and draw bounding boxes
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
    conf = box.conf[0].item()  # Confidence score
    label = f"Rebar {conf:.2f}"  # Class label with confidence

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Put label above the bounding box
    cv2.putText(image, label, (x1, y1 - 10), font, 0.5, (0, 255, 0), thickness)

# Display the number of objects detected on the image
text = f"Objects Detected: {num_objects}"
cv2.putText(image, text, (50, 50), font, font_scale, font_color, thickness)

# Save and display the output image
output_path = "detection_output5.jpg"
cv2.imwrite(output_path, image)
cv2.imshow("YOLOv11 Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print object count to console
print(f"Total objects detected: {num_objects}")
