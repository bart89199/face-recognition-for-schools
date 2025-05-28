#-------------MAIN------------------

CLOSE_DELAY = 3
SAVE_DETECTION_STATUS = True
USE_ARDUINO = True
FORMS_AUTOLOAD = True
RECORD_VIDEO = True
VIDEO_FPS = 30.0


FORMS_CHECK_INTERVAL = 10


#-----------FRAMES------------

LAST_FRAMES_AMOUNT = 25

# Be careful, it mustn't be bigger than LAST_FRAMES_AMOUNT
MIN_FRAMES_FOR_DETECTION = 15

NEED_BLINKS = 1

FRAMES_FOR_EYES_CHECK = 7

WAIT_FRAMES_FOR_DETECTION = 5

#----------------PORTS AND PATHS---------------

CAM_PORT = '/dev/video0'
ARDUINO_PORT = '/dev/ttyUSB0'

KNOWN_FACES_FILE = "known_faces.pkl"
SAVED_FORM_ANSWERS_FILE = "saved_form_answers.json"
BLOCKED_GOOGLE_FILES_FILE = "blocked_google_files.json"
UNKNOWN_NAME = "unknown"
WINDOW_NAME = "Face Recognition"

NEW_FRAMES_FOLDER = "new_frames"
OLD_FRAMES_FOLDER = "old_frames"
SAVED_FRAMES_FOLDER = "saved_faces"
VIDEOS_FOLDER = "videos"

FACE_RECOGNITION_MODEL = "large"


#----------GLOBAL VARS--------------

face_video_detector = None
face_image_detector = None
landmarker_image = None
landmarker = None
arduino = None
out_video = None


known_face_encodings = []
known_face_images = []
known_face_names = []
saved_form_answers = []
blocked_google_files = []
eyes = [{}] * LAST_FRAMES_AMOUNT
recognition_count = {}


last_blink_time = 0
last_forms_check_time = 0
frames_counter = [0]


#-----------FACE DETECTION----------------

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
SAVE_DELAY = 5
GOOGLE_DRIVE_FOLDER_ID = "1i-DoUmzkX9USiMLOOCeXjog52rma7RPW"
GOOGLE_FORM_ID = "1xfl6lFFMeqXw-B9e1Eqa4MkxE4Gmv-qaBBRTTd2ozvM"
FORM_ANSWER_ID = '7d1f44f4'
EMAIL_DOMAIN = '@fmschool72.ru'

SCOPES = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/forms.responses.readonly"]
creds = None

# ----------------EYES----------------
MIN_EYES_DIFFERENCE = 0.15
MIN_DIFS_FOR_BLICK = 0.3

BLINKED_EYES_OPEN = False

# MAX VALUE FOR CLOSED EYE
CLOSE_EYES_THRESHOLD = 0.2

# top, left, right, bottom faces frame scale for eyes owner finding
FRAME_FOR_EYES_SCALE = 0.5

# Eyes points for mediapipe
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


