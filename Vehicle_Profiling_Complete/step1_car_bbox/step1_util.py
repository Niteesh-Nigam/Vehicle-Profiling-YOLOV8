import csv

def write_csv(results, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['frame_number', 'tracking_id', 'x1', 'y1', 'x2', 'y2'])
        
        # Write the data
        for frame_number, tracks in results.items():
            for tracking_id, data in tracks.items():
                bbox = data['car']['bbox']
                writer.writerow([frame_number, tracking_id, *bbox])

# Example usage:
# write_csv(results, 'tracking_results.csv')
