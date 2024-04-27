from ultralytics import YOLO
import cv2
import numpy as np

from sort.sort import Sort
from step2_util import write_csv

# Initialize the tracker with modified parameters
mot_tracker = Sort(min_hits=3, max_age=5, iou_threshold=0.6)

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/plate.pt')  # Load your license plate detection model

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
                x1, y1, x2, y2 = track_id[:4]
                tracking_id = int(track_id[4])
                car_bbox = [x1, y1, x2, y2]
                # Detect license plate within the car bounding box
                lp_info = license_plate_detector(frame[int(y1):int(y2), int(x1):int(x2)])[0]
                if lp_info.boxes:
                    lp_x1, lp_y1, lp_x2, lp_y2 = lp_info.boxes.data.tolist()[0][:4]  # Assuming first detection is the most accurate
                    lp_bbox = [lp_x1 + x1, lp_y1 + y1, lp_x2 + x1, lp_y2 + y1]
                else:
                    lp_bbox = [None, None, None, None]

                results[frame_nmr][tracking_id] = {
                    'car': {'bbox': car_bbox},
                    'license_plate': {'bbox': lp_bbox}
                }
        
        else:
            results[frame_nmr][None] = {'car': {'bbox': [None, None, None, None]}}
            results[frame_nmr][None] = {'car': {'bbox': [None, None, None, None]},
                                    'license_plate': {'bbox': [None, None, None, None]}}

# After processing all frames, write results to CSV
write_csv(results, 'step2_lp_detect.csv')
