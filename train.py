import torch
from ultralytics import YOLO
import ultralytics


if __name__ == '__main__':
    print(torch.cuda.is_available())
    print('===========================')
    print(ultralytics.checks())

    model = YOLO('yolov8n.pt')
    model.train(data=r'C:\Users\user\PycharmProjects\pythonProject4\Fish-44\data.yaml',imgsz=640,batch=4,epochs=30)
