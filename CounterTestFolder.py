import cv2
import torch
import os
from ultralytics import YOLO

# Load the trained YOLOv11 model with best weights
model = YOLO("runs/detect/train4/weights/best.pt")  # Update with your actual model path

# Define input and output folders
input_folder = "Cropped_10M_CSA_200pcs"  # Folder containing test images
output_folder = "output_images_Cropped"  # Folder to save processed images
os.makedirs(output_folder, exist_ok=True)

# Define text properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 255, 0)  # Green text
thickness = 2

total_objects_detected = 0

# Loop through all images in the input folder
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {image_name}: Unable to read image.")
        continue

    # Resize image to 640x640 before inference
    image = cv2.resize(image, (640, 640))

    # Perform object detection
    results = model(image)
    detections = results[0].boxes
    num_objects = len(detections)
    total_objects_detected += num_objects

    # Loop through detections and draw bounding boxes
    for box in detections:
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

    # Save the processed image
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, image)
    print(f"Processed {image_name}: {num_objects} objects detected")

print(f"Total objects detected across all images: {total_objects_detected}")
print("Inference completed for all images.")
