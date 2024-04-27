from ultralytics import YOLO
import cv2
import numpy as np

from sort.sort import Sort
from step6_util import write_csv, car_make, read_license_plate, get_middle_color

# Initialize the tracker with modified parameters
mot_tracker = Sort(min_hits=3, max_age=5, iou_threshold=0.6)

# load models
coco_model = YOLO('./models/yolov8n.pt')
license_plate_detector = YOLO('./models/plate.pt')  # Load your license plate detection model
logo_detector = YOLO('./models/367.pt')

# load video
cap = cv2.VideoCapture('./traffic_edited1.mp4')

vehicles = [2, 3, 5, 7]  # class IDs for vehicles in COCO dataset

results = {}
logo_names = car_make()
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
            if int(class_id) in vehicles and (x2 - x1) > 50 and (y2 - y1) > 50:
                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue  # Skip detections with non-positive width or height
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        if detections_:
            track_ids = mot_tracker.update(np.asarray(detections_))

            for track_id in track_ids:
                x1, y1, x2, y2 = track_id[:4]
                tracking_id = int(track_id[4])
                car_bbox = [x1, y1, x2, y2]

                color = frame[int(y1):int(y2), int(x1):int(x2),:]
                if color.size > 0:
                    color_name = get_middle_color(color)

                    # Detect license plate within the car bounding box
                    lp_info = license_plate_detector(frame[int(y1):int(y2), int(x1):int(x2)])[0]
                    if lp_info.boxes: #///////////////////////////////////////////////////////////////////
                        lp_x1, lp_y1, lp_x2, lp_y2  = lp_info.boxes.data.tolist()[0][:4]  # Assuming first detection is the most accurate //////////////////////
                        lp_bbox = [lp_x1 + x1, lp_y1 + y1, lp_x2 + x1, lp_y2 + y1]

                        # crop license plate
                        license_plate_crop = frame[int(lp_y1 + y1):int(lp_y2 + y1), int(lp_x1 + x1):int(lp_x2 + x1),:]
                        # processing license plate
                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)# change '_" to what it means

                        # read license plate number
                        license_plate_text, text_score = read_license_plate(license_plate_crop_thresh)
                        lp_num= None
                        temp_num_score = 0
                        # lg_x1, lg_y1, lg_x2, lg_y2, lg_score, lg_class_id  = lg_info.boxes.data.tolist()[0][:6]  # Assuming first detection is the most accurate
                        # lg_bbox = [lg_x1 + x1, lg_y1 + y1, lg_x2 + x1, lg_y2 + y1]
                        if text_score is not None and text_score>= temp_num_score:
                            temp_num_score = text_score
                            lp_num = license_plate_text
                        print(f'Detected text: {lp_num}')
                        # print(f'Detected text: {license_plate_text}')
                            # lg_class_id_int = int(lg_class_id)
                            # manufacturer_name = logo_names.get(lg_class_id_int+1, 'Unknown')



                        # lp_num = license_plate_text
                        # print(f'Detected text: {lp_num}')
                        # print(f'Detected text: {license_plate_text}')

                    else:
                        lp_bbox = [None, None, None, None]
                        lp_num = None

                    lg_info = logo_detector(frame[int(y1):int(y2), int(x1):int(x2)])[0]
                    if lg_info.boxes.data.tolist():
                        temp_lg_score = 0
                        lg_x1, lg_y1, lg_x2, lg_y2, lg_score, lg_class_id  = lg_info.boxes.data.tolist()[0][:6]  # Assuming first detection is the most accurate
                        lg_bbox = [lg_x1 + x1, lg_y1 + y1, lg_x2 + x1, lg_y2 + y1]
                        if lg_score is not None and lg_score>= temp_lg_score:
                            temp_lg_score = lg_score
                            lg_class_id_int = int(lg_class_id)
                            manufacturer_name = logo_names.get(lg_class_id_int+1, 'Unknown')
                        
                        # print('hellllllllllllooooooooooooooooooooooooooooooooooooooooooo',manufacturer_name)
                    else:
                        lg_bbox = [None, None, None, None]
                        lg_class_id = [None]
                        manufacturer_name = None

                    results[frame_nmr][tracking_id] = {
                        'car': {'bbox' : car_bbox},
                        'car_color' : {'color' : color_name},
                        'license_plate': {'bbox': lp_bbox},
                        'plate_number' : {'lp_text' : lp_num},
                        'logo': {'bbox': lg_bbox},
                        'logo_ID': {'manufacturer' : manufacturer_name} 
                    }
        
        else:
            # results[frame_nmr][None] = {'car': {'bbox': [None, None, None, None]}}
            results[frame_nmr][None] = {'car': {'bbox': [None, None, None, None]},
                                        'car_color' : {'color' : None},
                                    'license_plate': {'bbox': [None, None, None, None]},
                                    'plate_number' : {'lp_text' : None},
                                    'logo': {'bbox': [None, None, None, None]},
                                    'logo_ID': {'manufacturer' : None} }

# After processing all frames, write results to CSV
write_csv(results, 'step6_lp_lg_detect.csv')
