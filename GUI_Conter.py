import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
from PIL import Image, ImageTk
import torch
from ultralytics import YOLO

# Initialize YOLOv11 model
model = YOLO("runs/detect/train4/weights/best.pt")  # Update with your actual model path


def upload_image():
    global img_path, img_resized
    img_path = filedialog.askopenfilename(filetypes=[("All Files", "*.*"), ("Image Files", "*.jpg;*.png;*.jpeg;*.bmp;*.tiff;*.gif")])

    if img_path:
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (640, 640))
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_preview = Image.fromarray(img_resized)
        img_preview = ImageTk.PhotoImage(img_preview)
        panel.config(image=img_preview)
        panel.image = img_preview


def perform_inference():
    global img_resized, img_path
    if img_path:
        results = model(img_resized)
        detections = results[0]
        num_objects = len(detections.boxes)

        # Draw bounding boxes
        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Object {box.conf[0]:.2f}"
            cv2.putText(img_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert image for display
        img_with_boxes = Image.fromarray(img_resized)
        img_with_boxes = ImageTk.PhotoImage(img_with_boxes)
        panel.config(image=img_with_boxes)
        panel.image = img_with_boxes

        # Update label
        count_label.config(text=f"Objects Detected: {num_objects}")


# Create GUI window
root = tk.Tk()
root.title("YOLOv11 Object Detection")
root.geometry("700x700")

# Upload Button
upload_btn = Button(root, text="Upload Image", command=upload_image)
upload_btn.pack()

# Image Display Panel
panel = Label(root)
panel.pack()

# Inference Button
infer_btn = Button(root, text="Detect Objects", command=perform_inference)
infer_btn.pack()

# Object Count Label
count_label = Label(root, text="Objects Detected: 0", font=("Arial", 14))
count_label.pack()

# Run GUI
root.mainloop()
