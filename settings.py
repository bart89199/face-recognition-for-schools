# # Global variable
# global last_blink_time
# global CAM_PORT
# global KNOWN_FACES_FILE
# global UNKNOWN_NAME
# global WINDOW_NAME
# global FACE_RECOGNITION_MODEL
# global FRAME_SCALE_TOP
# global FRAME_SCALE_LEFT
# global FRAME_SCALE_BOTTOM
# global FRAME_SCALE_RIGHT
# global MAX_FACES
# global LAST_FRAMES_AMOUNT
# global MIN_FRAMES_FOR_DETECTION
# global FACE_DETECTION_MODE
# global MAX_AVG_DISTANCE
# global MAX_PERCENT_DISTANCE
# global MIN_MATCH_FOR_PERSON
# global known_face_encodings
# global known_face_images
# global known_face_names
# global TEMP_PATH
# global last_saved_time
# global SAVE_DELAY
# global GOOGLE_DRIVE_FOLDER_ID
# global SAVE_DETECTION_STATUS
# global SCOPES
# global creds
# global MIN_EYES_DIFFERENCE
# global MIN_DIFS_FOR_BLICK
# global NEED_BLINKS
# global BLINKED_EYES_OPEN
# global FRAME_FOR_EYES_SCALE
# global CLOSE_EYES_THRESHOLD
# global RIGHT_EYE
# global LEFT_EYE
# global FRAMES_FOR_EYES_CHECK
# global eyes
# global raw_eyes
# global eyes_ready
# global last_frame_time
# global iteration
# global NEW_FRAMES_FOLDER
# global SAVED_FRAMES_FOLDER
# global OLD_FRAMES_FOLDER
# global CLOSE_DELAY
import asyncio
import os
from enum import Enum

CLOSE_DELAY_MS = 3000
SAVE_DETECTION_STATUS = True
FORMS_AUTOLOAD = True
RECORD_VIDEO = True

HLS_DIR = '/home/danil/testweb/hls'
PLAYLIST = 'stream.m3u8'
VIDEO_HEIGHT = 480
VIDEO_WIDTH = 640
VIDEO_FPS = 30

frame_queue = asyncio.Queue(10)

FORMS_CHECK_INTERVAL_MS = 10_000

class SystemStatus(Enum):
    STARTING = 0
    RUNNING = 1
    STOPPING = 2

system_status: SystemStatus = SystemStatus.STARTING

#-----------FRAMES------------

LAST_FRAMES_AMOUNT = 25

# Be careful, it mustn't be bigger than LAST_FRAMES_AMOUNT
MIN_FRAMES_FOR_DETECTION = 15

NEED_BLINKS = 1

FRAMES_FOR_EYES_CHECK = 9

WAIT_FRAMES_FOR_DETECTION = 5



CAM_PORT = '/dev/video0'

KNOWN_FACES_FILE = "known_faces.pkl"
SAVED_FORM_ANSWERS_FILE = "saved_form_answers.json"
BLOCKED_GOOGLE_FILES_FILE = "blocked_google_files.json"
UNKNOWN_NAME = "unknown"
WINDOW_NAME = "Face Recognition"

NEW_FRAMES_FOLDER = "new_frames"
OLD_FRAMES_FOLDER = "old_frames"
SAVED_FRAMES_FOLDER = "saved_faces"
VIDEOS_FOLDER = "/home/danil/testweb/records"

FACE_RECOGNITION_MODEL = "large"

face_video_detector = None
face_image_detector = None
landmarker_image = None
landmarker = None
out_video = None

stream_is_run = False

CONNECT_KOTLIN = True
KTOR_AUTH_TOKEN = os.getenv("PY_TOKEN")


known_face_encodings = []
known_face_images = []
known_face_names = []
saved_form_answers = []
blocked_google_files = []
eyes = [{}] * LAST_FRAMES_AMOUNT
recognition_count = {}

KTOR_STATUS_URL = "http://0.0.0.0/api/py/status"

last_blink_time = 0
last_forms_check_time = 0
frames_counter = [0]

# pixels scale
# FRAME_SCALE_HEIGHT = 1.2
# FRAME_SCALE_WIDTH = 1.1

MAX_FACES = 12

# 1 - check avg distance
# 2 - check encodings coincidence percent
FACE_DETECTION_MODE = 2

# For avg distance
MAX_AVG_DISTANCE = 0.54

# For encodings coincidence percent
MAX_PERCENT_DISTANCE = 0.55
MIN_MATCH_FOR_PERSON = 0.34


# ---------------SAVING AND LOADING FROM FORMS----------------

TEMP_PATH = "tmp"
last_saved_time = {}
SAVE_DELAY_MS = 5_000
GOOGLE_DRIVE_FOLDER_ID = "1i-DoUmzkX9USiMLOOCeXjog52rma7RPW"
GOOGLE_FORM_ID = "1xfl6lFFMeqXw-B9e1Eqa4MkxE4Gmv-qaBBRTTd2ozvM"
FORM_ANSWER_ID = '7d1f44f4'
EMAIL_DOMAIN = '@fmschool72.ru'

SCOPES = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/forms.responses.readonly"]
creds = None

# ----------------EYES----------------
MIN_EYES_DIFFERENCE = 0.15
MIN_DIF_FOR_BLICK = 0.3

BLINKED_EYES_OPEN = False

# MAX VALUE FOR CLOSED EYE
CLOSE_EYES_THRESHOLD = 0.2

# Eyes points for mediapipe
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

last_settings_load = 0

iteration = 0

cur_alive_names = set()
door_opened = False
