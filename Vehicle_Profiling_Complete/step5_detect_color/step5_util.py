import csv
import easyocr
from PIL import Image
import webcolors
import cv2
import numpy as np


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
        writer.writerow(['frame_number', 'tracking_id', 'car_x1', 'car_y1', 'car_x2', 'car_y2', 'detected_color', 'lp_x1', 'lp_y1', 'lp_x2', 'lp_y2', 'lp_num', 'lg_x1', 'lg_y1', 'lg_x2', 'lg_y2', 'lg_name'])
        
        # Write the data
        for frame_number, tracks in results.items():
            for tracking_id, data in tracks.items():
                car_bbox = data['car']['bbox']
                vehicle_color = data['car_color']['color']
                # 'car_color' : {'color' : color_name},
                #     'license_plate': {'bbox': lp_bbox},
                lp_bbox = data['license_plate']['bbox']
                lp_num  = data['plate_number']['lp_text']
                lg_bbox = data['logo']['bbox']
                lg_id = data['logo_ID']['manufacturer']
                # if not isinstance(lg_id, string):
                #     lg_id_list = [None]
                # else:
                # lg_id_list = lg_id
                # 'logo_ID': {'manufacturer' : lg_class_id} 
                writer.writerow([frame_number, tracking_id, *car_bbox, vehicle_color, *lp_bbox, lp_num, *lg_bbox, lg_id])

                
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


def get_closest_color_name(rgb_color):
    # Get a dictionary mapping named colors to their RGB values
    named_colors = webcolors.CSS3_NAMES_TO_HEX

    # Convert RGB color to a numpy array for easier calculation
    color_array = np.array(rgb_color)

    # Calculate the Euclidean distance between the given color and each named color
    closest_color = min(named_colors.items(), key=lambda item: np.linalg.norm(np.array(webcolors.hex_to_rgb(item[1])) - color_array))[0]

    return closest_color

def get_middle_color(image):
    # Read the image using OpenCV (to handle BGR format)
    bgr_image = image

    # Convert BGR image to RGB format
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Convert RGB image to PIL image
    pil_image = Image.fromarray(rgb_image)

    # Get the size of the image
    width, height = pil_image.size

    # Calculate middle coordinates, adjusted for odd dimensions
    x_coord = width - (width // (4/5)) 
    y_coord = height - (height // (3/4)) 
    x_coord = int(x_coord)
    y_coord = int(y_coord)

    # Get the color at the calculated pixel coordinates
    color = pil_image.getpixel((x_coord, y_coord))

    # Convert color to RGB format if it's not already
    if not isinstance(color, tuple):
        color = (color, color, color)  # Assuming grayscale, convert to RGB

    # Convert RGB color to the closest named color
    closest_color_name = get_closest_color_name(color)

    # Check if the closest color is 'gray' and output 'white' instead
    if closest_color_name.lower() == 'gray':
        closest_color_name = 'white'

    return closest_color_name