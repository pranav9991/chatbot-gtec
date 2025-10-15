import os
import shutil
from sklearn.model_selection import StratifiedShuffleSplit

DATASET_DIR = "dataset"  # original dataset folder
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
VAL_RATIO = 0.2  # 20%

# Step 1: Collect image paths and labels
file_paths = []
labels = []

for class_name in os.listdir(DATASET_DIR):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if os.path.isdir(class_dir) and class_name not in ["train", "val"]:
        for img_name in os.listdir(class_dir):
            file_paths.append(os.path.join(class_dir, img_name))
            labels.append(class_name)

# Step 2: Stratified split (preserves balance)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=42)
train_idx, val_idx = next(splitter.split(file_paths, labels))

train_files = [file_paths[i] for i in train_idx]
val_files = [file_paths[i] for i in val_idx]
train_labels = [labels[i] for i in train_idx]
val_labels = [labels[i] for i in val_idx]

# Step 3: Function to copy files
def copy_files(file_list, label_list, target_dir):
    for file_path, label in zip(file_list, label_list):
        class_dir = os.path.join(target_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        shutil.copy(file_path, os.path.join(class_dir, os.path.basename(file_path)))

# Step 4: Create train/val folders and copy
copy_files(train_files, train_labels, TRAIN_DIR)
copy_files(val_files, val_labels, VAL_DIR)

print(f"Done! Stratified split applied:")
print(f"Train size = {len(train_files)}, Val size = {len(val_files)}")
