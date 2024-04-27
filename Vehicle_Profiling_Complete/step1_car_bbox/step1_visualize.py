import cv2
import csv

def read_tracking_data(csv_file):
    tracking_results = {}
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            frame_number = int(row['frame_number'])
            tracking_id = row['tracking_id']
            if tracking_id.isdigit():
                tracking_id = int(tracking_id)
                bbox = [float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])]
            else:
                continue  # Skip this row if tracking ID is None

            if frame_number not in tracking_results:
                tracking_results[frame_number] = {}
            tracking_results[frame_number][tracking_id] = {'car': {'bbox': bbox}}
    
    return tracking_results

def main():
    csv_file = './step1_car_bbox_tracking_csv.csv'
    tracking_results = read_tracking_data(csv_file)

    video_path = './traffic_edited1.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open the video file")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./step1_car_bbox_tracking_csv.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_nmr = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_nmr += 1
        if frame_nmr in tracking_results:
            for tracking_id, data in tracking_results[frame_nmr].items():
                bbox = data['car']['bbox']
                if None not in bbox:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {tracking_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()
