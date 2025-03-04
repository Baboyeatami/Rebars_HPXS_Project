import cv2
import torch
import os
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

        tk.Label(left_frame, text="Counting Pilot").pack()
        self.counting_pilot_combo = ttk.Combobox(left_frame, values=["Auto", "Manual"])
        self.counting_pilot_combo.pack()
        self.counting_pilot_combo.bind("<<ComboboxSelected>>", self.toggle_counting_pilot)

        tk.Label(left_frame, text="Counting Process").pack()
        self.counting_process_combo = ttk.Combobox(left_frame, values=["CSA", "PNS", "Manual"], state="disabled")
        self.counting_process_combo.pack()
        self.counting_process_combo.bind("<<ComboboxSelected>>", self.toggle_counting_process)

        self.manual_count_frame = tk.Frame(left_frame)
        tk.Label(self.manual_count_frame, text="Manual Count Input").pack()
        self.manual_count_entry = tk.Entry(self.manual_count_frame)
        self.manual_count_entry.pack()

        tk.Label(left_frame, text="Image Acquisition").pack()
        self.acquisition_combo = ttk.Combobox(left_frame, values=["Camera", "File Upload"], state="disabled")
        self.acquisition_combo.pack()
        self.acquisition_combo.bind("<<ComboboxSelected>>", self.toggle_acquisition)

        # Right Panel
        right_frame = tk.Frame(root, width=700, height=600)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.datetime_label = tk.Label(right_frame, text="", font=("Arial", 12))
        self.datetime_label.pack()

        self.canvas = tk.Canvas(right_frame, width=640, height=480, bg="black")
        self.canvas.pack()

        self.counted_label = tk.Label(right_frame, text="Objects Counted: 0", font=("Arial", 14))
        self.counted_label.pack()

        self.discrepancy_label = tk.Label(right_frame, text="Discrepancy: 0", font=("Arial", 14), fg="green")
        self.discrepancy_label.pack()

        self.capture_btn = tk.Button(right_frame, text="Capture", command=self.capture_and_infer, state="disabled")
        self.capture_btn.pack()

        self.cap = None
        self.running = False
        self.frame = None

    def toggle_counting_pilot(self, event):
        self.counting_process_combo.config(state="readonly")

    def toggle_counting_process(self, event):
        self.acquisition_combo.config(state="readonly")
        if self.counting_process_combo.get() == "Manual":
            self.manual_count_frame.pack()
        else:
            self.manual_count_frame.pack_forget()

    def toggle_acquisition(self, event):
        selection = self.acquisition_combo.get()
        if selection == "Camera":
            self.start_camera()
            self.capture_btn.config(state="normal")
        elif selection == "File Upload":
            self.stop_camera()
            self.upload_image()
            self.capture_btn.config(state="disabled")

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        threading.Thread(target=self.update_camera, daemon=True).start()

    def stop_camera(self):
        if self.cap:
            self.running = False
            self.cap.release()
            self.cap = None

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

    def capture_and_infer(self):
        if self.frame is not None:
            self.perform_inference(self.frame.copy())

    def perform_inference(self, frame):
        results = model(frame)
        num_objects = len(results[0].boxes)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        process = self.counting_process_combo.get()

        # Compute Discrepancy based on process
        if process == "CSA":
            discrepancy = num_objects - 200 if num_objects > 150 else 100 - num_objects
        elif process == "PNS":
            discrepancy = num_objects - 250 if num_objects > 200 else 150 - num_objects
        elif process == "Manual":
            target = int(self.manual_count_entry.get() or 0)
            discrepancy = num_objects - target
        else:
            discrepancy = 0

        # Draw text on the image
        text_color = (0, 255, 0) if discrepancy == 0 else (0, 0, 255)  # Green if 0, Red otherwise
        cv2.putText(frame, f"Objects Counted: {num_objects}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Counting Process: {process}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Discrepancy: {discrepancy}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        # Save Image to counting_logs folder
        save_folder = "counting_logs"
        os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_folder, f"counting_{timestamp}.jpg")
        cv2.imwrite(save_path, frame)

        # Convert for Tkinter display
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((640, 480), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk
        self.counted_label.config(text=f"Objects Counted: {num_objects}")
        self.discrepancy_label.config(text=f"Discrepancy: {discrepancy}", fg="green" if discrepancy == 0 else "red")

        # Show the preview in a separate OpenCV window
        threading.Thread(target=self.show_preview, args=(frame,), daemon=True).start()

    def show_preview(self, frame):
        """ Displays the inference image in a separate OpenCV window. """
        preview_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Inference Preview", preview_frame)
        cv2.waitKey(0)  # Wait until key press
        cv2.destroyAllWindows()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = cv2.imread(file_path)
            self.perform_inference(img)

root = tk.Tk()
app = ObjectDetectionApp(root)
root.mainloop()
