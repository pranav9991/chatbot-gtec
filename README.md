from ultralytics import YOLO

# Load a YOLOv8 classification model (pretrained)
model = YOLO("yolov8n-cls.pt")  # Options: yolov8n-cls.pt, yolov8s-cls.pt, etc.

# Train
model.train(
    data="dataset",   # root folder with train/val
    epochs=20,
    imgsz=224,
    batch=32,
    lr0=0.001,
    project="makeup_classification",
    name="yolo_cls",
)
