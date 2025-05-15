import pickle
import threading
from time import sleep

import cv2

import os
import time

import uuid
import cv2
import face_recognition
import numpy
import numpy as np
import serial
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from googleapiclient.http import MediaFileUpload

import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import global_vars


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode



def eye_aspect_ratio(landmarks, eye_indices, img_width, img_height):
    def _point(index):  # Перевод нормализованных координат в пиксели
        p = landmarks[index]
        return int(p.x * img_width), int(p.y * img_height)

    p1, p2, p3, p4, p5, p6 = [_point(i) for i in eye_indices]
    # Вертикальные расстояния
    a = np.linalg.norm(np.array(p2) - np.array(p6))
    b = np.linalg.norm(np.array(p3) - np.array(p5))
    # Горизонтальное расстояние
    c = np.linalg.norm(np.array(p1) - np.array(p4))
    ear = (a + b) / (2.0 * c)

    return ear


def process_eyes(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    h, w = output_image.height, output_image.width

    for landmarks in result.face_landmarks:
        left_eye = eye_aspect_ratio(landmarks, global_vars.LEFT_EYE, w, h)
        right_eye = eye_aspect_ratio(landmarks, global_vars.RIGHT_EYE, w, h)
        avg_ear = (left_eye + right_eye) / 2.0

        pos_x = 0
        pos_y = 0
        for p in landmarks:
            pos_x += p.x * w
            pos_y += p.y * h

        pos_x /= len(landmarks)
        pos_y /= len(landmarks)

        global_vars.raw_eyes.append((pos_x, pos_y, avg_ear))
    global_vars.eyes_ready = True


def get_locations(frame, image):
    results = face_detector.detect(image=image)
    face_locations = []
    if results.detections:
        h, w, _ = frame.shape
        for detection in results.detections:

            bbox = detection.bounding_box
            left = bbox.origin_x
            top = bbox.origin_y
            right = bbox.origin_x + bbox.width
            bottom = bbox.origin_y + bbox.height

            top -= global_vars.FRAME_SCALE_TOP
            left -= global_vars.FRAME_SCALE_LEFT
            bottom += global_vars.FRAME_SCALE_BOTTOM
            right += global_vars.FRAME_SCALE_RIGHT

            top = max(top, 0)
            left = max(left, 0)
            bottom = min(bottom, h)
            right = min(right, w)

            face_locations.append((top, right, bottom, left))
    return face_locations

def save_frame(frame, filepath: str):
    save_result = cv2.imwrite(filepath, frame)
    return save_result

def save_recognition(name, frame):
    if not global_vars.SAVE_DETECTION_STATUS:
        return
    current_time = time.time()

    if not global_vars.last_saved_time.__contains__(name) or current_time - global_vars.last_saved_time[name] >= global_vars.SAVE_DELAY:
        timestamp = time.strftime("%Y%m%d_%H%M%S")  # Текущее время как часть имени файла
        filename = f"{name}_{timestamp}.jpg"


        # Печать пути для отладки
        print(f"[INFO] Сохранение изображения {filename}")

        filepath = os.path.join(global_vars.TEMP_PATH, filename)

        save_result = save_frame(frame, filepath)

        # Проверяем, удалось ли сохранить изображение
        if save_result:
            print(f"[INFO] Сохранено изображение {filename}")
            global_vars.last_saved_time[name] = current_time  # Обновляем время последнего сохранения

            try:
                # create drive api client
                service = build("drive", "v3", credentials=creds)

                file_metadata = {"name": filename, "parents": [global_vars.GOOGLE_DRIVE_FOLDER_ID]}
                media = MediaFileUpload(filepath, mimetype="image/jpeg")
                # pylint: disable=maybe-no-member
                file = (
                    service.files()
                    .create(body=file_metadata, media_body=media, fields="id")
                    .execute()
                )
                print(f'Изображение отправлено на сервер. File ID: {file.get("id")}')

            except HttpError as error:
                print(f"An error occurred: {error}")
                file = None

        else:
            print(f"[ERROR] Не удалось сохранить изображение в {filename}")


def check_encoding_avg(face_encoding):
    mx = [0, 0]
    for idx, known_faces in enumerate(global_vars.known_face_encodings):
        if len(known_faces) == 0:
            continue
        distances = face_recognition.face_distance(known_faces, face_encoding)
        avg = np.mean(distances)
        if global_vars.MAX_AVG_DISTANCE >= avg > mx[0]:
            mx = [avg, idx]
    name = global_vars.UNKNOWN_NAME
    best_match_index = mx[1]
    if mx[0] != 0:
        name = global_vars.known_face_names[best_match_index] + "(" + str(mx[0]) + ")"
    return name


def check_encoding_percent(face_encoding):
    mx = [0, 0]
    for idx, known_faces in enumerate(global_vars.known_face_encodings):
        if len(known_faces) == 0:
            continue
        distances = face_recognition.face_distance(known_faces, face_encoding)
        cnt = 0
        for d in distances:
            if d <= global_vars.MAX_PERCENT_DISTANCE:
                cnt += 1
        percent = cnt / len(known_faces)
        if mx[0] < percent and percent >= global_vars.MIN_MATCH_FOR_PERSON:
            mx = [percent, idx]
    name = global_vars.UNKNOWN_NAME
    best_match_index = mx[1]
    if mx[0] != 0:
        name = global_vars.known_face_names[best_match_index] + "(" + str(mx[0]) + ")"
    return name


def recognition(face_encoding):
    if global_vars.FACE_DETECTION_MODE == 1:
        return check_encoding_avg(face_encoding)
    else:
        return check_encoding_percent(face_encoding)


def analyze_eyes(face_names, face_locations):
    for x, y, avg in global_vars.raw_eyes:
        for idx, (top, left, bottom, right) in enumerate(face_locations):
            if face_names[idx] == global_vars.UNKNOWN_NAME:
                continue
            mid_x = (right + left) // 2
            mid_y = (top + bottom) // 2
            h = bottom - top
            w = left - right
            h = h * global_vars.FRAME_FOR_EYES_SCALE
            w = w * global_vars.FRAME_FOR_EYES_SCALE
            top = mid_y - h // 2
            bottom = mid_y + h // 2
            left = mid_x - w // 2
            right = mid_x + w // 2
            if top < y < bottom and left < x < right:
                global_vars.eyes[0][face_names[idx]] = (avg, 0)


def check_face(name):
    if not global_vars.eyes[0].__contains__(name):
        return False, False, False
    avg, _ = global_vars.eyes[0][name]
    difs = 0
    cnt = 0
    blinks = 0
    detect_cnt = 0
    for i in range(1, global_vars.LAST_FRAMES_AMOUNT):
        if not global_vars.eyes[i].__contains__(name):
            break
        detect_cnt += 1

    for i in range(1, global_vars.FRAMES_FOR_EYES_CHECK):
        if not global_vars.eyes[i].__contains__(name):
            break
        cnt += 1
        avgd, bls = global_vars.eyes[i][name]
        blinks += bls
        dif = abs(avg - avgd)
        if dif >= global_vars.MIN_EYES_DIFFERENCE:
            difs += 1
    blinked = cnt > 0 and (difs / cnt) > global_vars.MIN_DIFS_FOR_BLICK
    blinks += blinked
    global_vars.eyes[0][name] = (avg, blinked)
    return detect_cnt >= global_vars.MIN_FRAMES_FOR_DETECTION, blinks >= global_vars.NEED_BLINKS, avg <= global_vars.CLOSE_EYES_THRESHOLD


def clear():
    global_vars.eyes = [{}] + global_vars.eyes[:-1]
    global_vars.eyes_ready = False
    global_vars.raw_eyes = []




def process(frame):

    global_vars.iteration += 1
    clear()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    cur = int(time.time() * 1000)
    landmarker.detect_async(image, cur)

    face_locations = get_locations(frame, image)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model=global_vars.FACE_RECOGNITION_MODEL)

    face_names = []

    for i, face_encoding in enumerate(face_encodings):
        name = recognition(face_encoding)

        face_names.append(name)

    while not global_vars.eyes_ready:
        sleep(0.010)

    analyze_eyes(face_names, face_locations)

    alive_names = process_ready_faces(frame, face_locations, face_names)


    if len(alive_names) > 0:
        global_vars.last_blink_time = time.time()
    if time.time() - global_vars.last_blink_time < global_vars.CLOSE_DELAY:
        open_door()
        cv2.putText(frame, "People Allowed", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        close_door()
        cv2.putText(frame, "People Not Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cur_time = time.time()
    fps = 1 / (cur_time - global_vars.last_frame_time)
    cv2.putText(frame, "fps: " + str(int(fps)), (500, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    global_vars.last_frame_time = cur_time
    cv2.imshow(global_vars.WINDOW_NAME, frame)
    return face_encodings


def process_ready_faces(frame, face_locations, face_names):
    alive_names = []
    save = False
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 0, 255)
        recogn, alive, blinked = check_face(name)
        if recogn:
            save = True
            color = (230, 224, 76)
            if alive:
                color = (0, 255, 0)
            if blinked:
                color = (255, 33, 170)
            if alive and (not blinked or global_vars.BLINKED_EYES_OPEN):
                alive_names.append(name)
        else:
            if name != global_vars.UNKNOWN_NAME:
                color = (80, 127, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 5, bottom - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    for name in alive_names:
        th = threading.Thread(target=save_recognition, args=(name, frame))
        th.start()
    if save:
        th = threading.Thread(target=save_recognition, args=("recogn", frame))
        th.start()
    return alive_names



def open_door():
    write_arduino(1)

def close_door():
    write_arduino(0)

def write_arduino(x):
    arduino.write(bytes(str(x), 'utf-8'))

def save(face_encoding, name, frame):
    filename = name + "-" + str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(global_vars.SAVED_FRAMES_FOLDER, name)
    savepath = os.path.join(filepath, filename)
    save_result = save_frame(frame, savepath)
    filepath = os.path.join(filepath, filename)
    if not save_result:
        print("Can't save frame to folder(name = " + name + ")")
    if name not in global_vars.known_face_names:
        global_vars.known_face_names.append(name)
        global_vars.known_face_encodings.append([])
        global_vars.known_face_images.append([])

    index = global_vars.known_face_names.index(name)
    global_vars.known_face_encodings[index].append(face_encoding)
    global_vars.known_face_images[index].append(filepath)


def save_data():
    with open(global_vars.KNOWN_FACES_FILE, "wb") as f:
        pickle.dump((global_vars.known_face_encodings, global_vars.known_face_names, global_vars.known_face_images), f)



def load_drive():
    os.makedirs(global_vars.TEMP_PATH, exist_ok=True)
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    global creds
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", global_vars.SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", global_vars.SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

def load_mediapipe():

    base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite',
                                      delegate=python.BaseOptions.Delegate.GPU)
    face_detector_options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.6)

    global face_detector
    global face_mesh
    face_detector = vision.FaceDetector.create_from_options(face_detector_options)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker.task', delegate=python.BaseOptions.Delegate.GPU),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_faces=10,
        result_callback=process_eyes)
    global landmarker
    landmarker = FaceLandmarker.create_from_options(options)

def load_known_data():

    # Загрузка базы известных лиц
    if os.path.exists(global_vars.KNOWN_FACES_FILE):
        with open(global_vars.KNOWN_FACES_FILE, "rb") as f:
            global_vars.known_face_encodings, global_vars.known_face_names, global_vars.known_face_images = pickle.load(f)

def load_arduino():
    global arduino
    arduino = serial.Serial(port=global_vars.ARDUINO_PORT, baudrate=115200, timeout=.1)

def main():
    load_mediapipe()
    load_drive()
    load_known_data()
    load_arduino()


    print("Starting video")
    cap = cv2.VideoCapture(global_vars.CAM_PORT)
    print("Video started")
    cv2.namedWindow(global_vars.WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("[INFO] 'q' чтобы выйти.")
    while True:
        key = cv2.waitKey(1)
        ret, frame = cap.read()
        if not ret:
            print("Something went wrong with camera")
            break
        process(frame)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
