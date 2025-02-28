import cv2
import torch
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
import datetime
from ultralytics import YOLO

# Load YOLO model
model = YOLO("runs/detect/train4/weights/best.pt")


class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HPX SANTOS Rebar Counter")
        self.root.geometry("1000x600")

        # Left Panel
        left_frame = tk.Frame(root, width=300, height=600)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(left_frame, text="Image Acquisition").pack()
        self.acquisition_combo = ttk.Combobox(left_frame, values=["Camera", "File Upload"])
        self.acquisition_combo.pack()
        self.acquisition_combo.bind("<<ComboboxSelected>>", self.toggle_acquisition)

        tk.Label(left_frame, text="Counting Pilot").pack()
        self.counting_pilot_combo = ttk.Combobox(left_frame, values=["Auto", "Manual"])
        self.counting_pilot_combo.pack()
        self.counting_pilot_combo.bind("<<ComboboxSelected>>", self.toggle_counting_pilot)

        tk.Label(left_frame, text="Counting Process").pack()
        self.counting_process_combo = ttk.Combobox(left_frame, values=["CSA", "PNS", "Manual"])
        self.counting_process_combo.pack()

        self.counting_process_combo.bind("<<ComboboxSelected>>", self.toggle_counting_process)

        self.manual_count_frame = tk.Frame(left_frame)
        self.manual_count_frame.pack()
        tk.Label(self.manual_count_frame, text="Counting Process Manual").pack()
        self.manual_count_entry = tk.Entry(self.manual_count_frame)
        self.manual_count_entry.pack()

        tk.Label(left_frame, text="Discrepancy").pack()
        self.discrepancy_label = tk.Label(left_frame, text="0")
        self.discrepancy_label.pack()

        # Right Panel
        right_frame = tk.Frame(root, width=700, height=600)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.datetime_label = tk.Label(right_frame, text="", font=("Arial", 12))
        self.datetime_label.pack()

        self.canvas = tk.Canvas(right_frame, width=640, height=480, bg="black")
        self.canvas.pack()

        self.capture_btn = tk.Button(right_frame, text="Capture", command=self.capture_frame)
        self.capture_btn.pack()

        self.cap = None
        self.running = False
        self.frame = None

    def toggle_acquisition(self, event):
        selection = self.acquisition_combo.get()
        if selection == "Camera":
            self.start_camera()
        elif selection == "File Upload":
            self.upload_image()

    def toggle_counting_pilot(self, event):
        if self.counting_pilot_combo.get() == "Auto":
            self.running = True
            threading.Thread(target=self.run_camera_inference, daemon=True).start()

    def toggle_counting_process(self, event):
        if self.counting_process_combo.get() == "Manual":
            self.manual_count_frame.pack()
        else:
            self.manual_count_frame.pack_forget()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        threading.Thread(target=self.update_camera, daemon=True).start()

    def update_camera(self):
        while self.running and self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if not ret:
                break
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(self.frame)
            img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img
            self.datetime_label.config(text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def run_camera_inference(self):
        while self.running and self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if not ret:
                break
            results = model(self.frame)
            num_objects = len(results[0].boxes)

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(self.frame)
            img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img
            self.datetime_label.config(text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            count_target = {"CSA": 200, "PNS": 150}.get(self.counting_process_combo.get(),
                                                        int(self.manual_count_entry.get() or 0))
            discrepancy = abs(num_objects - count_target)
            self.discrepancy_label.config(text=f"Discrepancy: {discrepancy}")

    def capture_frame(self):
        if self.frame is not None:
            img_path = "captured_image.jpg"
            cv2.imwrite(img_path, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
            self.perform_inference(img_path)

    def perform_inference(self, image_path):
        results = model(image_path)
        img = cv2.imread(image_path)
        num_objects = len(results[0].boxes)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.perform_inference(file_path)


# Run application
root = tk.Tk()
app = ObjectDetectionApp(root)
root.mainloop()