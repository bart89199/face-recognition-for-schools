import cv2
from PIL import Image, ExifTags
import numpy as np


def read_frame_file(file):
    image = Image.open(file)
    orientation = 1
    try:
        exif = image.getexif()
        if exif:
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag) == "Orientation":
                    orientation = value
                    break
    except AttributeError:
        orientation = 1

    if orientation == 3:
        image = image.rotate(180, expand=True)
    elif orientation == 6:
        image = image.rotate(270, expand=True)
    elif orientation == 8:
        image = image.rotate(90, expand=True)

    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame

def get_rgb_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)