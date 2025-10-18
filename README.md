import os
import pandas as pd
import cv2
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "best_yolo.pt"      # path to your trained YOLO classification model
DATASET_PATH = "dataset/val"     # folder containing makeup / no_makeup
OUTPUT_DIR = "yolo_class_results"
CSV_PATH = "yolo_class_results.csv"
# ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLO classification model
model = YOLO(MODEL_PATH)

CLASS_NAMES = ["makeup", "no_makeup"]  # same order as YOLO training

results = []
correct = 0
total = 0

# Loop through dataset
for true_class in CLASS_NAMES:
    folder_path = os.path.join(DATASET_PATH, true_class)
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(folder_path, filename)
        total += 1

        # Run classification inference
        preds = model.predict(img_path, verbose=False)[0]  # first image result
        pred_class_id = int(preds.probs.argmax())
        pred_class = CLASS_NAMES[pred_class_id]
        confidence = float(preds.probs[pred_class_id])

        # Accuracy
        is_correct = (pred_class == true_class)
        if is_correct:
            correct += 1

        # Draw GT + Pred on image
        img = cv2.imread(img_path)
        display_img = img.copy()
        text = f"GT: {true_class} | Pred: {pred_class} ({confidence:.2f})"
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        cv2.putText(display_img, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Save image
        save_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(save_path, display_img)

        # Append to results
        results.append({
            "filename": filename,
            "ground_truth": true_class,
            "predicted": pred_class,
            "confidence": confidence
        })

# Save CSV
df = pd.DataFrame(results)
df.to_csv(CSV_PATH, index=False)

# Print accuracy
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"Total images: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"CSV saved to: {CSV_PATH}")
print(f"Labeled images saved in: {OUTPUT_DIR}")
