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

            # Check for NaN values for logo coordinates    
            lg_bbox = [row['lg_x1'], row['lg_y1'], row['lg_x2'], row['lg_y2']]
            lg_bbox = [float(x) for x in lg_bbox if str(x).replace('.', '', 1).isdigit()]

            if len(lg_bbox) != 4:
                lg_bbox = [None, None, None, None]  # Assign None if any LP coordinate is NaN

            # Read the logo name from the CSV
            lg_name = row['lg_name'] if 'lg_name' in row else 'Unknown'
            lp_num = row['lp_num'] if 'lp_num' in row else 'Unknown'

            if frame_number not in tracking_results:
                tracking_results[frame_number] = {}
            tracking_results[frame_number][tracking_id] = {
                'car': {'bbox': car_bbox},
                'license_plate': {'bbox': lp_bbox, 'number': lp_num},
                'logo': {'bbox': lg_bbox, 'name': lg_name}

            # if frame_number not in tracking_results:
            #     tracking_results[frame_number] = {}
            # tracking_results[frame_number][tracking_id] = {
            #     'car': {'bbox': car_bbox},
            #     'license_plate': {'bbox': lp_bbox},
            #     'logo': {'bbox': lg_bbox}
            }
            

    return tracking_results

def main():
    csv_file = './step4_lp_lg_detect.csv'  # Update the path if the CSV is in a different directory
    tracking_results = read_tracking_data(csv_file)

    video_path = './traffic_edited1.mp4'  # Update the path if the video is in a different directory
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open the video file")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./step4_lp_lg_detect.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

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
                lp_num = data['license_plate']['number']
                lg_bbox = data['logo']['bbox']
                lg_name = data['logo']['name']

                if None not in car_bbox:
                    x1, y1, x2, y2 = map(int, car_bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {tracking_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if lp_bbox and len(lp_bbox) == 4 and None not in lp_bbox:
                    lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_bbox)
                    cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (255, 0, 0), 2)
                    cv2.putText(frame, "LP", (lp_x1, lp_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    cv2.putText(frame, lp_num, (x1, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
                if lg_bbox and len(lg_bbox) == 4 and None not in lg_bbox:
                    lg_x1, lg_y1, lg_x2, lg_y2 = map(int, lg_bbox)
                    cv2.rectangle(frame, (lg_x1, lg_y1), (lg_x2, lg_y2), (0, 0, 255), 2)
                    # Now display the logo name instead of "LG"
                    cv2.putText(frame, lg_name, (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()


