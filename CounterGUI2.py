import cv2
import torch
import tkinter as tk
from tkinter import filedialog, Label, Button, Entry
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load the YOLOv11 model
model = YOLO("runs/detect/train4/weights/best.pt")  # Update with actual model path


class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HPX SANTOS Rebar Counter")
        self.root.geometry("800x600")

        # UI Elements
        self.upload_btn = Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack()

        self.detect_btn = Button(root, text="Detect Rebars Objects", command=self.detect_objects)
        self.detect_btn.pack()

        self.webcam_btn = Button(root, text="Start camera", command=self.start_webcam)
        self.webcam_btn.pack()

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

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("All Files", "*.*"), ("Image Files", "*.jpg;*.png;*.jpeg;*.bmp;*.tiff;*.gif")])
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
        img = cv2.resize(img, (640, 640))
        cv2.putText(img, f"Rebars: {num_objects}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Update labels
        self.result_label.config(text=f"Rebard detected: {num_objects}")
        self.diff_label.config(text=f"Difference: {difference}")

        # Convert and display image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

    def start_webcam(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            num_objects = len(results[0].boxes)

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, f"Objects: {num_objects}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Webcam Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


# Run the app
root = tk.Tk()
app = ObjectDetectionApp(root)
root.mainloop()
