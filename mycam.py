from ultralytics import YOLO

import cv2

model = YOLO(r"yolov8s.pt")
results = model(
    source=0,
    stream=True,
)

for result in results:
    plotted = result.plot(0)
    cv2.imshow("yolo inference", plotted)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break