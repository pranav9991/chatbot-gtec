from ultralytics import YOLO

model = YOLO("makeup_classification/yolo_cls/weights/best.pt")  # path to best model

results = model.predict("test.jpg")  # single image or folder

for r in results:
    print(r.probs)          # probabilities
    print(r.names)          # class names
    print(r.probs.top1)     # predicted class index
    print(r.names[r.probs.top1])  # predicted class label
