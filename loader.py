import os
import pickle

import serial
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from mediapipe.tasks import python
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker, RunningMode

import global_vars


def load_googleapi():
    os.makedirs(global_vars.TEMP_PATH, exist_ok=True)

    if os.path.exists("token.json"):
        global_vars.creds = Credentials.from_authorized_user_file("token.json", global_vars.SCOPES)

    if not global_vars.creds or not global_vars.creds.valid:
        if global_vars.creds and global_vars.creds.expired and global_vars.creds.refresh_token:
            global_vars.creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", global_vars.SCOPES
            )
            global_vars.creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(global_vars.creds.to_json())


def load_mediapipe_video():
    base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite',
                                      delegate=python.BaseOptions.Delegate.GPU)
    face_detector_options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5, running_mode=RunningMode.VIDEO)

    global_vars.face_video_detector = vision.FaceDetector.create_from_options(face_detector_options)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker.task', delegate=python.BaseOptions.Delegate.GPU),
        num_faces=10, running_mode=RunningMode.VIDEO,
        min_face_detection_confidence=0.5
    )
    global_vars.landmarker = FaceLandmarker.create_from_options(options)

def load_mediapipe_image():
    base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite',
                                      delegate=python.BaseOptions.Delegate.GPU)
    face_detector_options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5,
                                                       running_mode=RunningMode.IMAGE)

    global_vars.face_image_detector = vision.FaceDetector.create_from_options(face_detector_options)


def load_known_data():
    # Загрузка базы известных лиц
    if os.path.exists(global_vars.KNOWN_FACES_FILE):
        with open(global_vars.KNOWN_FACES_FILE, "rb") as f:
            global_vars.known_face_encodings, global_vars.known_face_names, global_vars.known_face_images = pickle.load(
                f)
    if os.path.exists(global_vars.SAVED_FORM_ANSWERS_FILE):
        with open(global_vars.SAVED_FORM_ANSWERS_FILE, "rb") as f:
            global_vars.saved_form_answers = pickle.load(f)


def load_arduino():
    if global_vars.USE_ARDUINO:
        global_vars.arduino = serial.Serial(port=global_vars.ARDUINO_PORT, baudrate=115200, timeout=.1)
        print("Arduino connected")
    else:
        print("Arduino off")
