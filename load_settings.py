import json

import door
import settings

settings_file_path = "/home/danil/testweb/settings.json"


def load():
    try:
        with open(settings_file_path) as f:
            new_settings = json.load(f)

            settings.CLOSE_DELAY_MS = new_settings['close_delay_ms']
            settings.SAVE_DETECTION_STATUS = new_settings['save_detection']
            settings.USE_ARDUINO = new_settings['use_arduino']
            settings.FORMS_AUTOLOAD = new_settings['forms_autoload']
            settings.FORMS_CHECK_INTERVAL_MS = new_settings['forms_check_interval_ms']

            settings.LAST_FRAMES_AMOUNT = new_settings['last_frames_amount']
            settings.MIN_FRAMES_FOR_DETECTION = new_settings['min_frames_for_detection']
            settings.NEED_BLINKS = new_settings['need_blinks']
            settings.FRAMES_FOR_EYES_CHECK = new_settings['frames_for_eyes_check']
            settings.WAIT_FRAMES_FOR_DETECTION = new_settings['wait_frames_for_detection']
            settings.CAM_PORT = new_settings['cam_port']
            door.ARDUINO_PORT = new_settings['arduino_port']
            settings.MAX_FACES = new_settings['max_faces']
            settings.FACE_DETECTION_MODE = new_settings['face_detection_mode']
            settings.MAX_AVG_DISTANCE = new_settings['max_avg_distance']
            settings.MAX_PERCENT_DISTANCE = new_settings['max_percent_distance']
            settings.MIN_MATCH_FOR_PERSON = new_settings['min_match_for_person']
            settings.SAVE_DELAY_MS = new_settings['save_delay_ms']
            settings.MIN_EYES_DIFFERENCE = new_settings['min_eyes_difference']
            settings.MIN_DIF_FOR_BLICK = new_settings['min_dif_for_blink']
            settings.BLINKED_EYES_OPEN = new_settings['blinked_eyes_open']
            settings.CLOSE_EYES_THRESHOLD = new_settings['close_eyes_threshold']

    except Exception as e:
        print("Failed to load settings")
        print(e)