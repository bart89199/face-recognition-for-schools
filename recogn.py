import time

import cv2
import mediapipe as mp
import face_recognition
import pickle

import numpy as np

import google.auth
from googleapiclient.http import MediaFileUpload
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

KNOWN_FACES_FILE = "known_faces.pkl"
UNKNOWN_NAME = "unknown"
MAX_DISTANCE = 0.5
# MIN_MATCH_FOR_PERSON = 0.3
FRAME_SKIP = 3
CAM_PORT = 0
WINDOW_NAME = "Face Recognition"
TEMP_PATH = "tmp"

last_saved_time = 0
SAVE_DELAY = 7
GOOGLE_DRIVE_FOLDER_ID = "1i-DoUmzkX9USiMLOOCeXjog52rma7RPW"

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# Инициализация MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)
print("Starting video")
cap = cv2.VideoCapture(CAM_PORT)
print("Video started")
frame_count = 0
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Загрузка базы известных лиц
if os.path.exists(KNOWN_FACES_FILE):
    with open(KNOWN_FACES_FILE, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings = []
    known_face_names = []

creds = None
# The file token.json stores the user's access and refresh tokens, and is
# created automatically when the authorization flow completes for the first
# time.
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


def save_frame(name, frame):
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


def save_frame1(name, frame):
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

                file_metadata = {"name": filepath}
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


def show(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if name != UNKNOWN_NAME:
            save_frame(name, frame)
        color = (0, 255, 0) if name != UNKNOWN_NAME else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 5, bottom - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.imshow(WINDOW_NAME, frame)


def recognition(face_encoding):
    mx = [0, 0]
    for idx, known_faces in enumerate(known_face_encodings):
        if len(known_faces) == 0:
            continue
        distances = face_recognition.face_distance(known_faces, face_encoding)
        mean_distance = np.mean(distances)
        if mean_distance < MAX_DISTANCE:
            mx = [mean_distance, idx]
    name = UNKNOWN_NAME
    best_match_index = mx[1]
    if mx[0] != 0:
        name = known_face_names[best_match_index] + "(" + str(mx[0]) + ")"
    return name


def get_locations(frame, results):
    face_locations = []
    if results.detections:
        h, w, _ = frame.shape
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            top = int(box.ymin * h)
            left = int(box.xmin * w)
            bottom = top + int(box.height * h)
            right = left + int(box.width * w)

            top = max(top, 0)
            left = max(left, 0)
            bottom = min(bottom, h)
            right = min(right, w)

            face_locations.append((top, right, bottom, left))
    return face_locations


def process(max_faces=10):
    ret, frame = cap.read()
    if not ret:
        exit(6)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)

    face_locations = get_locations(frame, results)
    while len(face_locations) > max_faces:
        face_locations = face_locations[:max_faces]
    face_names = []

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model="cnn")

    for face_encoding in face_encodings:
        name = recognition(face_encoding)
        face_names.append(name)
    show(frame, face_locations, face_names)

    return face_encodings


def save(face_encoding):
    if name not in known_face_names:
        known_face_names.append(name)
        known_face_encodings.append([face_encoding])
    else:
        index = known_face_names.index(name)
        # known_face_encodings[index] = np.concatenate((known_face_encodings[index], [face_encodings[0]]), axis=0)
        known_face_encodings[index].append(face_encoding)


def save_data():
    with open(KNOWN_FACES_FILE, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)


print("[INFO] Нажмите 'U' чтобы сохранить лицо или обновить имеющееся, 'D' чтобы удалить лицо, 'Q' чтобы выйти.")
while True:
    # ret, frame = cap.read()
    # if not ret:
    #     exit(6)
    # cv2.imshow("Распознавание лиц", frame)
    # key = cv2.waitKey(1)
    # if key == ord('q'):
    #     break
    # continue
    # if frame_count % FRAME_SKIP != 0:
    #  continue
    face_encodings = process()
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
            face_encodings = process(1)
            if key == ord('s'):
                if face_encodings:
                    save(face_encodings[0])
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
    elif key == ord('r'):
        if known_face_names:
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
