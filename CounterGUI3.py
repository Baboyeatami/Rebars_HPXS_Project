import cv2
import torch
import tkinter as tk
from tkinter import filedialog, Label, Button, Entry, Checkbutton, BooleanVar
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import numpy as np

# Load the YOLOv11 model
model = YOLO("runs/detect/train4/weights/best.pt")  # Update with actual model path


class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HPX SANTOS Rebar Counter")
        self.root.geometry("900x700")

        # UI Elements
        self.upload_btn = Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack()

        self.detect_btn = Button(root, text="Detect Rebar Objects", command=self.detect_objects)
        self.detect_btn.pack()

        self.webcam_btn = Button(root, text="Start Webcam", command=self.start_webcam)
        self.webcam_btn.pack()

        self.capture_btn = Button(root, text="Capture Image", command=self.capture_image)
        self.capture_btn.pack()

        self.infer_btn = Button(root, text="Run Inference on Captured Image", command=self.detect_captured_image)
        self.infer_btn.pack()

        # Turn Off Camera Checkbox
        self.cam_var = BooleanVar(value=False)
        self.cam_checkbox = Checkbutton(root, text="Turn Off Camera", variable=self.cam_var, command=self.stop_webcam)
        self.cam_checkbox.pack()

        self.counter_label = Label(root, text="Enter Counter Range:")
        self.counter_label.pack()

        self.counter_entry = Entry(root)
        self.counter_entry.pack()

        self.result_label = Label(root, text="Rebar Objects Detected: 0")
        self.result_label.pack()

        self.diff_label = Label(root, text="Difference: 0")
        self.diff_label.pack()

        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.image_path = None
        self.captured_image = None  # Store captured image for inference
        self.cap = None
        self.webcam_active = False

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("All Files", "*.*"),("Image Files", "*.jpg;*.png;*.jpeg;*.bmp;*.tiff;*.gif"), ("All Files", "*.*")]
        )
        if self.image_path:
            self.display_image(self.image_path)

    def display_image(self, path):
        img = Image.open(path)
        img = img.resize((640, 640))  # Resize image to 640x640
        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

    def detect_objects(self):
        if not self.image_path:
            self.result_label.config(text="No image selected!")
            return

        # Perform inference
        results = model(self.image_path)
        img = cv2.imread(self.image_path)

        self.process_inference_results(results, img)

    def start_webcam(self):
        if self.webcam_active:
            return

        self.webcam_active = True
        self.cap = cv2.VideoCapture(0)

        def update_frame():
            while self.webcam_active:
                ret, frame = self.cap.read()
                if not ret or self.cam_var.get():
                    break

                results = model(frame)
                num_objects = len(results[0].boxes)

                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(frame, f"Objects: {num_objects}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Convert frame to Tkinter format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = frame.resize((640, 480))
                self.img_tk = ImageTk.PhotoImage(frame)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

            self.cap.release()

        threading.Thread(target=update_frame, daemon=True).start()

    def capture_image(self):
        if not self.cap or not self.webcam_active:
            self.result_label.config(text="Webcam not active!")
            return

        ret, frame = self.cap.read()
        if not ret:
            self.result_label.config(text="Failed to capture image!")
            return

        # Store captured image for later inference
        self.captured_image = frame.copy()

        # Stop webcam feed
        self.webcam_active = False
        self.stop_webcam()

        # Convert and display image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = frame.resize((640, 480))
        self.img_tk = ImageTk.PhotoImage(frame)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

        self.result_label.config(text="Image Captured! Click 'Run Inference'.")

    def detect_captured_image(self):
        if self.captured_image is None:
            self.result_label.config(text="No captured image available!")
            return

        # Perform inference on captured image
        results = model(self.captured_image)
        img = self.captured_image.copy()

        self.process_inference_results(results, img)

    def process_inference_results(self, results, img):
        num_objects = len(results[0].boxes)

        # Get counter range and compute difference
        try:
            counter_range = int(self.counter_entry.get())
        except ValueError:
            counter_range = 0
        difference = abs(num_objects - counter_range)

        # Draw bounding boxes
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(img, f"Rebars: {num_objects}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Update labels
        self.result_label.config(text=f"Rebars detected: {num_objects}")
        self.diff_label.config(text=f"Difference: {difference}")

        # Convert and display image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((640, 480))
        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

    def stop_webcam(self):
        self.webcam_active = False
        if self.cap:
            self.cap.release()
        self.cap = None


# Run the app
root = tk.Tk()
app = ObjectDetectionApp(root)
root.mainloop()
