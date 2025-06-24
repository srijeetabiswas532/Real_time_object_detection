import torch
import cv2
import pandas as pd

model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')
TARGET_CLASSES = ['book', 'bottle', 'cup', 'person']

frame = cv2.imread("data/example.jpg")  # use any real image
results = model(frame)
print(results.pandas().xyxy[0])
