from ultralytics import YOLO

model = YOLO(r"yolov8s.pt")
print(model.task)
print(model.names)
print(sum(p.numel() for p in model.parameters()))