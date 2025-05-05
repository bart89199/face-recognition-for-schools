import pickle
import threading

import cv2

import os
import time

import cv2
import face_recognition
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

# Global variable
last_blink_time = 0  # store timestamp of last blink
BLINK_DISPLAY_DURATION = 3  # seconds to keep "People Allowed" on screen

CAM_PORT = '/dev/video0'

KNOWN_FACES_FILE = "known_faces.pkl"
UNKNOWN_NAME = "unknown"
WINDOW_NAME = "Face Recognition"

MAX_AVG_DISTANCE = 0.54

MAX_PERCENT_DISTANCE = 0.6
MIN_MATCH_FOR_PERSON = 0.3

# 1 - check avg distance
# 2 - check encodings coincidence percent
FACE_DETECTION_MODE = 2

# Параметры для моргания
EAR_THRESHOLD = 0.2
CONSEC_FRAMES = 2
blink_counter = 0

# Точки глаз в face_mesh (правый и левый глаз)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

TEMP_PATH = "tmp"
last_saved_time = 0
SAVE_DELAY = 7
GOOGLE_DRIVE_FOLDER_ID = "1i-DoUmzkX9USiMLOOCeXjog52rma7RPW"
SAVE_DETECTION_STATUS = False

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
creds = None

known_face_encodings = []
known_face_names = []

face_mesh = None
face_detector = None

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


# Global dictionary to store blink counters for each face
face_blink_states = {}  # {face_index: {'counter': int, 'blinked': bool}}

def check_eyes_all_faces(frame, rgb_frame):
    results_mesh = face_mesh.process(rgb_frame)
    h, w, _ = frame.shape
    blink_results = []

    if results_mesh.multi_face_landmarks:
        for i, face_landmarks in enumerate(results_mesh.multi_face_landmarks):
            landmarks = face_landmarks.landmark

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            # Init state if not exists
            if i not in face_blink_states:
                face_blink_states[i] = {'counter': 0, 'blinked': False}

            state = face_blink_states[i]
            blinked = False

            if avg_ear < EAR_THRESHOLD:
                state['counter'] += 1
            else:
                if state['counter'] >= CONSEC_FRAMES:
                    blinked = True
                state['counter'] = 0

            state['blinked'] = blinked
            blink_results.append(blinked)
    return blink_results


def get_locations(frame):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    results = face_detector.detect(image=image)
    face_locations = []
    if results.detections:
        h, w, _ = frame.shape
        for detection in results.detections:
            # box = detection.bounding_box
            # top = int(box.origin_y)
            # left = int(box.xmin * w)
            # bottom = top + int(box.height * h)
            # right = left + int(box.width * w)

            bbox = detection.bounding_box
            left = bbox.origin_x
            top = bbox.origin_y
            right = bbox.origin_x + bbox.width
            bottom = bbox.origin_y + bbox.height

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

    if current_time - last_saved_time >= SAVE_DELAY:
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
            last_saved_time = current_time  # Обновляем время последнего сохранения

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
        if avg <= MAX_AVG_DISTANCE and mx[0] < avg:
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


import time

# Global or external variable to track message and its display time
last_message_time = 0
current_message = ""

def process(cap):
    global last_blink_time
    ret, frame = cap.read()
    if not ret:
        exit(6)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = get_locations(frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model="cnn")

    face_names = []
    blinked_list = check_eyes_all_faces(frame, rgb_frame)

    for i, face_encoding in enumerate(face_encodings):
        name = recognition(face_encoding)  # returns known name or "unknown"
        blinked = blinked_list[i] if i < len(blinked_list) else False

        if name.lower() != UNKNOWN_NAME and blinked:
            last_blink_time = time.time()
            name += " [blinked]"

        face_names.append(name)

    # Status display
    if time.time() - last_blink_time < BLINK_DISPLAY_DURATION:
        cv2.putText(frame, "People Allowed", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "People Not Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    show(frame, face_locations, face_names)
    return face_encodings


def show(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 255, 0) if name != UNKNOWN_NAME else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 5, bottom - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    for (_, _, _, _), name in zip(face_locations, face_names):
        if name != UNKNOWN_NAME:
            th = threading.Thread(target=save_frame, args=(name, frame))
            th.start()
    cv2.imshow(WINDOW_NAME, frame)

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

    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    # Загрузка базы известных лиц
    if os.path.exists(KNOWN_FACES_FILE):
        with open(KNOWN_FACES_FILE, "rb") as f:
            global known_face_encodings
            global known_face_names
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