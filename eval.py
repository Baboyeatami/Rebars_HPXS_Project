import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

# Load trained YOLOv11n model
model_path = "runs/detect/train4/weights/best.pt"  # Adjust path if needed
model = YOLO(model_path)

# Evaluate model on test dataset
results = model.val(split='test', plots=True)  # Use split='val' if validating

# Extract performance metrics
metrics = results.box
map50 = metrics.map50  # Mean AP at IoU threshold 0.5
map50_95 = metrics.map  # Mean AP at IoU thresholds 0.5 to 0.95
precision = metrics.mp  # Mean Precision
recall = metrics.mr  # Mean Recall
f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)  # Avoid division by zero

# Print evaluation metrics
print(f"📊 Evaluation Metrics:")
print(f"  - mAP@50: {map50:.4f}")
print(f"  - mAP@50-95: {map50_95:.4f}")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall: {recall:.4f}")
print(f"  - F1-score: {f1_score:.4f}")

# 📌 Plot Precision-Recall Curve
if len(metrics.p) > 0 and len(metrics.r) > 0:
    plt.figure(figsize=(6, 6))
    plt.plot(metrics.r, metrics.p, marker='o', label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("⚠️ Precision-Recall data not available.")

# 📌 Plot Confusion Matrix
conf_matrix = results.confusion_matrix if hasattr(results, "confusion_matrix") else None
if conf_matrix is not None:
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
else:
    print("⚠️ Confusion matrix not available. Try running with plots=True in val().")

# ---------------------- 📊 TRAINING METRICS ---------------------- #
# 📌 Extract training metrics from results.json
log_dir = "runs/detect/train"  # Adjust path if needed
log_file = os.path.join(log_dir, "results.json")

if os.path.exists(log_file):
    with open(log_file, "r") as f:
        logs = json.load(f)

    # Extract loss curves
    epochs = range(1, len(logs["metrics"]["train/loss"]) + 1)
    box_loss = logs["metrics"]["train/loss/box"]
    obj_loss = logs["metrics"]["train/loss/obj"]
    cls_loss = logs["metrics"]["train/loss/cls"]

    # 📌 Plot Training Loss Curves
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, box_loss, label="Box Loss", color="blue")
    plt.plot(epochs, obj_loss, label="Objectness Loss", color="red")
    plt.plot(epochs, cls_loss, label="Classification Loss", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("⚠️ Training log file not found. Ensure training has completed properly.")
