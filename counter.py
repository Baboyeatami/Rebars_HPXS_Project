import cv2
from ultralytics import YOLO

# Load the trained YOLOv11 model with best weights
model = YOLO("runs/detect/train4/weights/best.pt")  # Update with your model path

# Path to the image for inference
image_path = "5.png"  # Replace with your image path

# Perform object detection
results = model(image_path)

# Get detected objects count
num_objects = len(results[0].boxes)

# Load the original image
image = cv2.imread(image_path)

# Define font and position for overlay text
font = cv2.FONT_HERSHEY_SIMPLEX
position = (50, 50)  # Text position (x, y)
font_scale = 1
font_color = (0, 255, 0)  # Green text
thickness = 2

# Put the object count text on the image
text = f"Objects Detected: {num_objects}"
cv2.putText(image, text, position, font, font_scale, font_color, thickness)

# Save and display the output image
output_path = "detection_output_counter.jpg"
cv2.imwrite(output_path, image)
cv2.imshow("YOLOv11 Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print count to console
print(f"Total objects detected: {num_objects}")
