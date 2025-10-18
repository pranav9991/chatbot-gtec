import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# ----------------------------------
# CONFIG
# ----------------------------------
MODEL_PATH = "best_resnet.h5"       # change if efficientnet/inception
DATASET_PATH = "dataset/val"        # folder containing makeup / no_makeup
OUTPUT_IMAGE_DIR = "results_images"
CSV_PATH = "results.csv"
IMG_SIZE = (224, 224)               # same as training
# ----------------------------------

# Load model
model = load_model(MODEL_PATH)

# You said: folder contains "makeup" first and "no_makeup" last
# Alphabetical order is the same: makeup=0, no_makeup=1
CLASS_NAMES = ["makeup", "no_makeup"]

# Create output directory
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

results = []
correct = 0
total = 0

# Loop through each class folder
for true_class in CLASS_NAMES:
    folder_path = os.path.join(DATASET_PATH, true_class)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, filename)

            # Load and preprocess image
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, IMG_SIZE)
            img_array = img_to_array(img_resized) / 255.0  # rescale=1./255 like training
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            pred = model.predict(img_array, verbose=0)
            pred_idx = np.argmax(pred, axis=1)[0]
            pred_class = CLASS_NAMES[pred_idx]

            # Accuracy calculation
            total += 1
            is_correct = (pred_class == true_class)
            if is_correct:
                correct += 1

            # Draw GT and Pred on image
            display_img = img.copy()
            text = f"GT: {true_class} | Pred: {pred_class}"
            color = (0, 255, 0) if is_correct else (0, 0, 255)  # Green if correct, Red if wrong

            cv2.putText(display_img,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2)

            # Save image with label
            save_path = os.path.join(OUTPUT_IMAGE_DIR, filename)
            cv2.imwrite(save_path, display_img)

            # Save row to CSV data
            results.append({
                "filename": filename,
                "ground_truth": true_class,
                "predicted": pred_class
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
print(f"Labeled images saved in: {OUTPUT_IMAGE_DIR}")
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# ----------------------------------
# CONFIG
# ----------------------------------
MODEL_PATH = "best_resnet.h5"       # change if efficientnet/inception
DATASET_PATH = "dataset/val"        # folder containing makeup / no_makeup
OUTPUT_IMAGE_DIR = "results_images"
CSV_PATH = "results.csv"
IMG_SIZE = (224, 224)               # same as training
# ----------------------------------

# Load model
model = load_model(MODEL_PATH)

# You said: folder contains "makeup" first and "no_makeup" last
# Alphabetical order is the same: makeup=0, no_makeup=1
CLASS_NAMES = ["makeup", "no_makeup"]

# Create output directory
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

results = []
correct = 0
total = 0

# Loop through each class folder
for true_class in CLASS_NAMES:
    folder_path = os.path.join(DATASET_PATH, true_class)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, filename)

            # Load and preprocess image
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, IMG_SIZE)
            img_array = img_to_array(img_resized) / 255.0  # rescale=1./255 like training
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            pred = model.predict(img_array, verbose=0)
            pred_idx = np.argmax(pred, axis=1)[0]
            pred_class = CLASS_NAMES[pred_idx]

            # Accuracy calculation
            total += 1
            is_correct = (pred_class == true_class)
            if is_correct:
                correct += 1

            # Draw GT and Pred on image
            display_img = img.copy()
            text = f"GT: {true_class} | Pred: {pred_class}"
            color = (0, 255, 0) if is_correct else (0, 0, 255)  # Green if correct, Red if wrong

            cv2.putText(display_img,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2)

            # Save image with label
            save_path = os.path.join(OUTPUT_IMAGE_DIR, filename)
            cv2.imwrite(save_path, display_img)

            # Save row to CSV data
            results.append({
                "filename": filename,
                "ground_truth": true_class,
                "predicted": pred_class
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
print(f"Labeled images saved in: {OUTPUT_IMAGE_DIR}")
