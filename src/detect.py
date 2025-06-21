# clean wrapper to load webcam, model, filter for object classes, and save output.

import torch
import cv2
import pathlib as Path
import pandas as pd
from datetime import datetime

model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local') # using local weights

TARGET_CLASSES = ['book','bottle','cup','person']

cap = cv2.VideoCapture(0) # opens default camera - device index 0

logged_objects = set()
logged_rows = []

while cap.isOpened(): # as long as webcam is open & sending frames
    ret, frame = cap.read() # ret: that frame was read successfully; frame: actual camera image
    if not ret:
        break

    results = model(frame) # running YOLO on image
    detections = results.pandas().xyxy[0] # converts predictions to pandas df (xyxy gives boxing coordinates)

    for _, row in detections.iterrows(): # go through each detected object
        label = row['name']
        if label in TARGET_CLASSES: # only act if object is in the target classes
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']]) # boxing coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # drawing a box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) # writing name above box

            logged_objects.add(label)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            logged_rows.append({
                "object": label,
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
                "timestamp": timestamp
            })

    cv2.imshow("Detection", frame) # display results in a window called "Detection"
    if cv2.waitKey(1) & 0xFF == ord('q'): # if you press 'q', system quits
        break

cap.release() # close webcam
cv2.destroyAllWindows() # close all opencv windows

if logged_rows:
    df = pd.DataFrame(logged_rows)
    df.to_csv('./outputs/detections.csv', index=False)
    print(f"[✔] Saved {len(df)} detections to results/detections.csv")
else:
    print("[ℹ] No target objects detected.")
