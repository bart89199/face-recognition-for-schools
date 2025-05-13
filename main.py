import pickle
import threading
from time import sleep

import cv2

import os
import time

import cv2
import face_recognition
import numpy
import numpy as np
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from googleapiclient.http import MediaFileUpload

import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from numpy import asarray

# Global variable
last_blink_time = 0  # store timestamp of last blink
BLINK_DISPLAY_DURATION = 3  # seconds to keep "People Allowed" on screen

CAM_PORT = '/dev/video2'

KNOWN_FACES_FILE = "known_faces.pkl"
UNKNOWN_NAME = "unknown"
WINDOW_NAME = "Face Recognition"

# pixels scale
FRAME_SCALE_TOP = 27
FRAME_SCALE_LEFT = 7
FRAME_SCALE_BOTTOM = 12
FRAME_SCALE_RIGHT = 7

MAX_FACES = 6

LAST_FRAMES_AMOUNT = 25

# Be careful, it mustn't be bigger than LAST_FRAMES_AMOUNT
MIN_FRAMES_FOR_DETECTION = 12

# 1 - check avg distance
# 2 - check encodings coincidence percent
FACE_DETECTION_MODE = 2

# For avg distance
MAX_AVG_DISTANCE = 0.54

# For encodings coincidence percent
MAX_PERCENT_DISTANCE = 0.6
MIN_MATCH_FOR_PERSON = 0.3

known_face_encodings = []
known_face_names = []

# -------SAVING--------

TEMP_PATH = "tmp"
last_saved_time = {}
SAVE_DELAY = 5
GOOGLE_DRIVE_FOLDER_ID = "1i-DoUmzkX9USiMLOOCeXjog52rma7RPW"
SAVE_DETECTION_STATUS = False

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
creds = None

# -------EYES-----------
MIN_EYES_DIFFERENCE = 0.15
MIN_DIFS_FOR_BLICK = 0.3

NEED_BLINKS = 2

BLINKED_EYES_OPEN = False

# MAX VALUE FOR CLOSED EYE
CLOSE_EYES_THRESHOLD = 0.2

# top, left, right, bottom faces frame scale for eyes owner finding
FRAME_FOR_EYES_SCALE = 0.5

# Eyes points for mediapipe
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

FRAMES_FOR_EYES_CHECK = 9

eyes = [{}] * LAST_FRAMES_AMOUNT
raw_eyes = []
eyes_ready = False

last_frame_time = 0

iteration = 0

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

    global raw_eyes
    global eyes_ready
    for landmarks in result.face_landmarks:
        left_eye = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_eye = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_eye + right_eye) / 2.0

        pos_x = 0
        pos_y = 0
        for p in landmarks:
            pos_x += p.x * w
            pos_y += p.y * h

        pos_x /= len(landmarks)
        pos_y /= len(landmarks)

        raw_eyes.append((pos_x, pos_y, avg_ear))
    eyes_ready = True


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

            top -= FRAME_SCALE_TOP
            left -= FRAME_SCALE_LEFT
            bottom += FRAME_SCALE_BOTTOM
            right += FRAME_SCALE_RIGHT

            top = max(top, 0)
            left = max(left, 0)
            bottom = min(bottom, h)
            right = min(right, w)

            face_locations.append((top, right, bottom, left))
    return face_locations


def save_frame(name, frame):
    if not SAVE_DETECTION_STATUS:
        return
    global last_saved_time
    current_time = time.time()

    if not last_saved_time.__contains__(name) or current_time - last_saved_time[name] >= SAVE_DELAY:
        timestamp = time.strftime("%Y%m%d_%H%M%S")  # Текущее время как часть имени файла
        filename = f"{name}_{timestamp}.jpg"
        filepath = os.path.join(TEMP_PATH, filename)

        # Печать пути для отладки
        print(f"[INFO] Сохранение изображения в: {filepath}")

        # Пытаемся сохранить изображение
        save_result = cv2.imwrite(filepath, frame)

        # Проверяем, удалось ли сохранить изображение
        if save_result:
            print(f"[INFO] Сохранено изображение: {filepath}")
            last_saved_time[name] = current_time  # Обновляем время последнего сохранения

            try:
                # create drive api client
                service = build("drive", "v3", credentials=creds)

                file_metadata = {"name": filename, "parents": [GOOGLE_DRIVE_FOLDER_ID]}
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
            print(f"[ERROR] Не удалось сохранить изображение в: {filepath}")


def check_encoding_avg(face_encoding):
    mx = [0, 0]
    for idx, known_faces in enumerate(known_face_encodings):
        if len(known_faces) == 0:
            continue
        distances = face_recognition.face_distance(known_faces, face_encoding)
        avg = np.mean(distances)
        if MAX_AVG_DISTANCE >= avg > mx[0]:
            mx = [avg, idx]
    name = UNKNOWN_NAME
    best_match_index = mx[1]
    if mx[0] != 0:
        name = known_face_names[best_match_index] + "(" + str(mx[0]) + ")"
    return name


def check_encoding_percent(face_encoding):
    mx = [0, 0]
    for idx, known_faces in enumerate(known_face_encodings):
        if len(known_faces) == 0:
            continue
        distances = face_recognition.face_distance(known_faces, face_encoding)
        cnt = 0
        for d in distances:
            if d <= MAX_PERCENT_DISTANCE:
                cnt += 1
        percent = cnt / len(known_faces)
        if mx[0] < percent:
            mx = [percent, idx]
    name = UNKNOWN_NAME
    best_match_index = mx[1]
    if mx[0] != 0:
        name = known_face_names[best_match_index] + "(" + str(mx[0]) + ")"
    return name


def recognition(face_encoding):
    if FACE_DETECTION_MODE == 1:
        return check_encoding_avg(face_encoding)
    else:
        return check_encoding_percent(face_encoding)


def analyze_eyes(face_names, face_locations):
    for x, y, avg in raw_eyes:
        for idx, (top, left, bottom, right) in enumerate(face_locations):
            if face_names[idx] == UNKNOWN_NAME:
                continue
            mid_x = (right + left) // 2
            mid_y = (top + bottom) // 2
            h = bottom - top
            w = left - right
            h = h * FRAME_FOR_EYES_SCALE
            w = w * FRAME_FOR_EYES_SCALE
            top = mid_y - h // 2
            bottom = mid_y + h // 2
            left = mid_x - w // 2
            right = mid_x + w // 2
            if top < y < bottom and left < x < right:
                eyes[0][face_names[idx]] = (avg, 0)


def check_face(name):
    if not eyes[0].__contains__(name):
        return False, False, False
    avg, _ = eyes[0][name]
    difs = 0
    cnt = 0
    blinks = 0
    detect_cnt = 0
    for i in range(1, LAST_FRAMES_AMOUNT):
        if not eyes[i].__contains__(name):
            break
        detect_cnt += 1

    for i in range(1, FRAMES_FOR_EYES_CHECK):
        if not eyes[i].__contains__(name):
            break
        cnt += 1
        avgd, bls = eyes[i][name]
        blinks += bls
        dif = abs(avg - avgd)
        if dif >= MIN_EYES_DIFFERENCE:
            difs += 1
    blinked = cnt > 0 and (difs / cnt) > MIN_DIFS_FOR_BLICK
    blinks += blinked
    eyes[0][name] = (avg, blinked)
    return detect_cnt >= MIN_FRAMES_FOR_DETECTION, blinks >= NEED_BLINKS, avg <= CLOSE_EYES_THRESHOLD


def clear():
    global eyes
    global eyes_ready
    global raw_eyes
    eyes = [{}] + eyes[:-1]
    eyes_ready = False
    raw_eyes = []


def process(cap):
    global last_blink_time
    ret, frame = cap.read()
    if not ret:
        exit(6)
    global iteration
    iteration += 1
    clear()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    cur = int(time.time() * 1000)
    landmarker.detect_async(image, cur)

    face_locations = get_locations(frame, image)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model="large")

    face_names = []

    for i, face_encoding in enumerate(face_encodings):
        name = recognition(face_encoding)

        face_names.append(name)

    while not eyes_ready:
        sleep(0.010)

    analyze_eyes(face_names, face_locations)

    alive_names = process_ready_faces(frame, face_locations, face_names)

    if len(alive_names) > 0:
        last_blink_time = time.time()
    if time.time() - last_blink_time < BLINK_DISPLAY_DURATION:
        cv2.putText(frame, "People Allowed", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "People Not Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cur_time = time.time()
    global last_frame_time
    fps = 1 / (cur_time - last_frame_time)
    cv2.putText(frame, "fps: " + str(int(fps)), (500, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    last_frame_time = cur_time
    cv2.imshow(WINDOW_NAME, frame)
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
            if alive and (not blinked or BLINKED_EYES_OPEN):
                alive_names.append(name)
        else:
            if name != UNKNOWN_NAME:
                color = (80, 127, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 5, bottom - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    for name in alive_names:
        th = threading.Thread(target=save_frame, args=(name, frame))
        th.start()
    if save:
        th = threading.Thread(target=save_frame, args=("recogn", frame))
        th.start()
    return alive_names


def save(face_encoding, name):
    if name not in known_face_names:
        known_face_names.append(name)
        known_face_encodings.append([face_encoding])
    else:
        index = known_face_names.index(name)
        known_face_encodings[index].append(face_encoding)


def save_data():
    with open(KNOWN_FACES_FILE, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)


def main():
    os.makedirs(TEMP_PATH, exist_ok=True)
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    global creds
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

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

    global known_face_encodings
    global known_face_names

    # Загрузка базы известных лиц
    if os.path.exists(KNOWN_FACES_FILE):
        with open(KNOWN_FACES_FILE, "rb") as f:
            known_face_encodings, known_face_names = pickle.load(f)

    print("Starting video")
    cap = cv2.VideoCapture(CAM_PORT)
    print("Video started")
    frame_count = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("[INFO] Нажмите 'U' чтобы сохранить лицо или обновить имеющееся, 'D' чтобы удалить лицо, 'Q' чтобы выйти.")
    while True:
        face_encodings = process(cap)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('u'):
            name = input("Введите имя для лица: ")
            while name == UNKNOWN_NAME:
                print("Это имя зарезервировано")
                name = input("Введите имя для лица: ")
            print(
                "Теперь сохраните несколько кадров(клавиша 'S'), а потом выйдите(клавиша 'F'), так же вы можете удалить предыдущий кадр(клавиша 'R')")
            key = cv2.waitKey(1)
            while key != ord('f'):
                #     if frame_count % FRAME_SKIP != 0:
                #          continue
                face_encodings = process(cap)
                if key == ord('s'):
                    if face_encodings:
                        save(face_encodings[0], name)
                        print(f"[INFO] Лицо {name} сохранено.")
                    else:
                        print("[WARNING] Нет лица для сохранения!")
                elif key == ord('r'):
                    if name not in known_face_names or len(known_face_encodings[known_face_names.index(name)]) == 0:
                        print("[WARNING] Для данного имени отсутствуют сохранения!")
                    else:
                        index = known_face_names.index(name)
                        known_face_encodings[index] = known_face_encodings[index][:-1]
                        print(f"[INFO] Последний кадр лица {name} удалён.")
                key = cv2.waitKey(1)
            save_data()
            print(f"[INFO] Изменения сохранены!")
        elif key == ord('d'):
            if len(known_face_names) > 0:
                # Показываем список лиц и предлагаем удалить
                print("Доступные лица для удаления:")
                for idx, name in enumerate(known_face_names):
                    print(f"{idx + 1}. {name}")

                choice = input("Введите номер лица для удаления (или 'q' для отмены): ")
                if choice.lower() == 'q':
                    continue
                try:
                    index_to_delete = int(choice) - 1
                    if 0 <= index_to_delete < len(known_face_names):
                        deleted_name = known_face_names.pop(index_to_delete)
                        known_face_encodings.pop(index_to_delete)
                        save_data()
                        print(f"[INFO] Лицо '{deleted_name}' удалено.")
                    else:
                        print("[WARNING] Неверный номер.")
                except ValueError:
                    print("[WARNING] Введите корректный номер.")
            else:
                print("[WARNING] Нет известных лиц для удаления.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
