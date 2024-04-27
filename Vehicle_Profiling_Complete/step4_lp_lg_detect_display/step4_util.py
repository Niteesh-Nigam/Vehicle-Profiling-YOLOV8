import csv
import easyocr

def car_make():
    logo_names = {
        1: 'Audi',
        2: 'Chrysler',
        3: 'Citroen',
        4: 'GMC',
        5: 'Honda',
        6: 'Hyundai',
        7: 'Infiniti',
        8: 'Mazda',
        9: 'Mercedes',
        10: 'Mercury',
        11: 'Mitsubishi',
        12: 'Nissan',
        13: 'Renault',
        14: 'Toyota',
        15: 'Volkswagen',
        16: 'acura',
        17: 'bmw',
        18: 'cadillac',
        19: 'chevrolet',
        20: 'dodge',
        21: 'ford',
        22: 'jeep',
        23: 'kia',
        24: 'lexus',
        25: 'lincoln',
        26: 'mini',
        27: 'no class',
        28: 'porsche',
        29: 'ram',
        30: 'range rover',
        31: 'skoda',
        32: 'subaru',
        33: 'suzuki',
        34: 'volvo'
    }
    return logo_names

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

def write_csv(results, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['frame_number', 'tracking_id', 'car_x1', 'car_y1', 'car_x2', 'car_y2', 'lp_x1', 'lp_y1', 'lp_x2', 'lp_y2', 'lp_num', 'lg_x1', 'lg_y1', 'lg_x2', 'lg_y2', 'lg_name'])
        
        # Write the data
        for frame_number, tracks in results.items():
            for tracking_id, data in tracks.items():
                car_bbox = data['car']['bbox']
                lp_bbox = data['license_plate']['bbox']
                lp_num  = data['plate_number']['lp_text']
                lg_bbox = data['logo']['bbox']
                lg_id = data['logo_ID']['manufacturer']
                # if not isinstance(lg_id, string):
                #     lg_id_list = [None]
                # else:
                lg_id_list = lg_id
                # 'logo_ID': {'manufacturer' : lg_class_id} 
                writer.writerow([frame_number, tracking_id, *car_bbox, *lp_bbox, lp_num, *lg_bbox, lg_id_list])

                
def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')

        return text

    return None
