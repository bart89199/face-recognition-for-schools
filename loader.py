import json
import os
import pickle

import serial
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from mediapipe.tasks import python
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker, RunningMode
from pillow_heif import register_heif_opener
from serial.serialutil import SerialException

import settings


def load_googleapi():
    os.makedirs(settings.TEMP_PATH, exist_ok=True)

    if os.path.exists("token.json"):
        settings.creds = Credentials.from_authorized_user_file("token.json", settings.SCOPES)

    if not settings.creds or not settings.creds.valid:
        if settings.creds and settings.creds.expired and settings.creds.refresh_token:
            settings.creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", settings.SCOPES
            )
            settings.creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(settings.creds.to_json())


def load_mediapipe():
    # base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite',
    #                                   delegate=python.BaseOptions.Delegate.GPU)
    # face_detector_options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5, running_mode=RunningMode.VIDEO)
    #
    # settings.face_video_detector = vision.FaceDetector.create_from_options(face_detector_options)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker.task', delegate=python.BaseOptions.Delegate.GPU),
        num_faces=5, running_mode=RunningMode.VIDEO,
        min_face_detection_confidence=0.5
    )
    settings.landmarker = FaceLandmarker.create_from_options(options)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker.task', delegate=python.BaseOptions.Delegate.GPU),
        num_faces=5, running_mode=RunningMode.IMAGE,
        min_face_detection_confidence=0.5
    )
    settings.landmarker_image = FaceLandmarker.create_from_options(options)
#     base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite',
#                                       delegate=python.BaseOptions.Delegate.GPU)
#     face_detector_options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5,
#                                                        running_mode=RunningMode.IMAGE)
#
#     settings.face_image_detector = vision.FaceDetector.create_from_options(face_detector_options)


def load_known_data():
    # Загрузка базы известных лиц
    if os.path.exists(settings.KNOWN_FACES_FILE):
        with open(settings.KNOWN_FACES_FILE, "rb") as f:
            settings.known_face_encodings, settings.known_face_names, settings.known_face_images = pickle.load(
                f)
    if os.path.exists(settings.SAVED_FORM_ANSWERS_FILE):
        with open(settings.SAVED_FORM_ANSWERS_FILE, "r") as f:
            settings.saved_form_answers = json.load(f)
    if os.path.exists(settings.BLOCKED_GOOGLE_FILES_FILE):
        with open(settings.BLOCKED_GOOGLE_FILES_FILE, "r") as f:
            settings.blocked_google_files = json.load(f)


def load_main():
    register_heif_opener()