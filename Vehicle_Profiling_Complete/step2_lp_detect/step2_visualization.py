import cv2
import csv
import numpy as np

def read_tracking_data(csv_file):
    tracking_results = {}
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            frame_number = int(row['frame_number'])
            tracking_id = int(row['tracking_id'])  # Assuming tracking ID will always be a valid integer

            car_bbox = [float(row['car_x1']), float(row['car_y1']), float(row['car_x2']), float(row['car_y2'])]
            
            # Check for NaN values for license plate coordinates
            lp_bbox = [row['lp_x1'], row['lp_y1'], row['lp_x2'], row['lp_y2']]
            lp_bbox = [float(x) for x in lp_bbox if str(x).replace('.', '', 1).isdigit()]

            if len(lp_bbox) != 4:
                lp_bbox = [None, None, None, None]  # Assign None if any LP coordinate is NaN

            if frame_number not in tracking_results:
                tracking_results[frame_number] = {}
            tracking_results[frame_number][tracking_id] = {
                'car': {'bbox': car_bbox},
                'license_plate': {'bbox': lp_bbox}
            }

    return tracking_results

def main():
    csv_file = './step2_lp_detect.csv'  # Update the path if the CSV is in a different directory
    tracking_results = read_tracking_data(csv_file)

    video_path = './traffic_edited1.mp4'  # Update the path if the video is in a different directory
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open the video file")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./step2_lp_detect.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_nmr = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_nmr += 1
        if frame_nmr in tracking_results:
            for tracking_id, data in tracking_results[frame_nmr].items():
                car_bbox = data['car']['bbox']
                lp_bbox = data['license_plate']['bbox']

                if None not in car_bbox:
                    x1, y1, x2, y2 = map(int, car_bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {tracking_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if lp_bbox and len(lp_bbox) == 4 and None not in lp_bbox:
                    lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_bbox)
                    cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (255, 0, 0), 2)
                    cv2.putText(frame, "LP", (lp_x1, lp_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()
