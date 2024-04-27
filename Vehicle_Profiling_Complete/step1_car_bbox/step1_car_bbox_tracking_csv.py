from ultralytics import YOLO
import cv2
import numpy as np

from sort.sort import Sort
from step1_util import write_csv

# Initialize the tracker with modified parameters
mot_tracker = Sort(min_hits=3, max_age=5, iou_threshold=0.6)  # Adjust these values as needed

# load models
coco_model = YOLO('yolov8n.pt')

# load video
cap = cv2.VideoCapture('./traffic_edited1.mp4')

vehicles = [2, 3, 5, 7]  # class IDs for vehicles in COCO dataset

results = {}
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles and (x2 - x1) > 200 and (y2 - y1) > 200:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        if detections_:
            track_ids = mot_tracker.update(np.asarray(detections_))

            for track_id in track_ids:
                x1, y1, x2, y2, _ = track_id[:5]
                tracking_id = int(track_id[4])
                results[frame_nmr][tracking_id] = {'car': {'bbox': [x1, y1, x2, y2]}}
        
        else:
            results[frame_nmr][None] = {'car': {'bbox': [None,None,None,None]}}

# After processing all frames, write results to CSV
write_csv(results, 'step1_car_bbox_tracking_csv.csv')
