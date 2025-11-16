from ultralytics import YOLO

model = YOLO(r"D:\Program Files (x86)\deeplearning\ultralytics-8.3.163\runs\detect\train7\weights\best.pt")
model.predict(
    source = r"D:\Program Files (x86)\deeplearning\make_dataset\images",#可以直接在此更换想检测的图片/视频路径
    # source = 0,#摄像头
    save = True,
    show = False,
    # visualize = True,
    save_txt = True,
)
