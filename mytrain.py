from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"yolo11n.pt")
    model.train(
        data = r"zizhi.yaml",
        epochs = 100,
        #通过调节以下四个选项可以提高GPU的利用率
        imgsz = 640,#imgsz越小训练越快，一般情况下建议设置为640
        batch = 32,#批次，设置为-1yolo会自动寻找合适的batch
        cache = "ram",#缓存，默认不使用缓存
        workers = 1,#打包
        # val = False,#只训练不验证
    )
