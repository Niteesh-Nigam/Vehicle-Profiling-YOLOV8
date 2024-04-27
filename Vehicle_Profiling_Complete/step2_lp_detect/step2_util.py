import csv

def write_csv(results, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['frame_number', 'tracking_id', 'car_x1', 'car_y1', 'car_x2', 'car_y2', 'lp_x1', 'lp_y1', 'lp_x2', 'lp_y2'])
        
        # Write the data
        for frame_number, tracks in results.items():
            for tracking_id, data in tracks.items():
                car_bbox = data['car']['bbox']
                lp_bbox = data['license_plate']['bbox']
                writer.writerow([frame_number, tracking_id, *car_bbox, *lp_bbox])

# Example usage:
# write_csv(results, 'tracking_results.csv')
