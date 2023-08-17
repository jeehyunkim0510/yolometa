import torch
from ultralytics import YOLO
import ultralytics

print(torch.cuda.is_available())
print('===========================')
print(ultralytics.checks())


if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    model.train(data='D:/myprj/yolometa/Fish-44/data.yaml',imgsz=640,batch=4,epochs=30,device=0)
