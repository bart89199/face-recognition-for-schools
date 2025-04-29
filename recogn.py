import cv2
import mediapipe as mp
import face_recognition
import pickle
import os
import numpy as np
import time

KNOWN_FACES_FILE = "known_faces.pkl"
UNKNOWN_NAME = "unknown"
MAX_DISTANCE = 0.5
FRAME_SKIP = 3
CAM_PORT = 0
WINDOW_NAME = "Face Recognition"

# === НАСТРОЙКА СОХРАНЕНИЯ ===
SAVE_DIR = "C:/Users/suslo/PycharmProjects/face-recognition-for-schools1/checkers"  # Поменяй на путь к Google Диску
os.makedirs(SAVE_DIR, exist_ok=True)
last_saved_time = 0
SAVE_DELAY = 5  # секунд

# MediaPipe и камера
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

print("Starting video")
cap = cv2.VideoCapture(CAM_PORT)
if not cap.isOpened():
    print("[ERROR] Не удалось открыть камеру.")
    exit(1)
print("Video started")
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Загрузка известных лиц
if os.path.exists(KNOWN_FACES_FILE):
    with open(KNOWN_FACES_FILE, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings = []
    known_face_names = []

def show(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
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
        name = known_face_names[best_match_index] + "(" + str(round(mx[0], 2)) + ")"
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

def save_encoding(face_encoding, name):
    if name not in known_face_names:
        known_face_names.append(name)
        known_face_encodings.append([face_encoding])
    else:
        index = known_face_names.index(name)
        known_face_encodings[index].append(face_encoding)

def save_data():
    with open(KNOWN_FACES_FILE, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)

print("[INFO] Нажмите 'U' чтобы сохранить лицо или обновить имеющееся, 'D' чтобы удалить лицо, 'Q' чтобы выйти.")
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[ERROR] Не удалось считать кадр с камеры.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)
    face_locations = get_locations(frame, results)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model="cnn")
    face_names = []

    current_time = time.time()
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        name = recognition(face_encoding)
        face_names.append(name)
        if name != UNKNOWN_NAME:
            if current_time - last_saved_time >= SAVE_DELAY:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp}.jpg"
                filepath = os.path.join(SAVE_DIR, filename)

                # Добавляем проверку, чтобы увидеть, что происходит
                print(f"[INFO] Попытка сохранить: {filepath}")  # Печатаем путь
                save_result = cv2.imwrite(filepath, frame)

                if save_result:
                    print(f"[INFO] Сохранено изображение: {filepath}")
                else:
                    print(f"[ERROR] Не удалось сохранить изображение в: {filepath}")

                last_saved_time = current_time

    show(frame, face_locations, face_names)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('u'):
        name = input("Введите имя для лица: ")
        while name == UNKNOWN_NAME:
            print("Это имя зарезервировано")
            name = input("Введите имя для лица: ")
        print("Сохраняйте кадры ('S'), выходите ('F'), удаляйте ('R')")
        key = cv2.waitKey(1)
        while key != ord('f'):
            ret, frame = cap.read()
            if not ret:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_frame)
            face_locations = get_locations(frame, results)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model="cnn")
            if key == ord('s'):
                if face_encodings:
                    save_encoding(face_encodings[0], name)
                    print(f"[INFO] Лицо {name} сохранено.")
                else:
                    print("[WARNING] Лицо не обнаружено.")
            elif key == ord('r'):
                if name not in known_face_names:
                    print("[WARNING] Нет такого имени.")
                else:
                    index = known_face_names.index(name)
                    if known_face_encodings[index]:
                        known_face_encodings[index] = known_face_encodings[index][:-1]
                        print(f"[INFO] Последнее фото удалено.")
                    else:
                        print("[WARNING] У этого имени нет данных.")
            key = cv2.waitKey(1)
        save_data()
        print("[INFO] Данные сохранены.")
    elif key == ord('r'):
        if known_face_names:
            print("Список лиц:")
            for idx, name in enumerate(known_face_names):
                print(f"{idx + 1}. {name}")
            choice = input("Введите номер для удаления (или 'q' для отмены): ")
            if choice.lower() == 'q':
                continue
            try:
                index = int(choice) - 1
                if 0 <= index < len(known_face_names):
                    deleted = known_face_names.pop(index)
                    known_face_encodings.pop(index)
                    save_data()
                    print(f"[INFO] Удалено лицо: {deleted}")
                else:
                    print("[WARNING] Неверный номер.")
            except ValueError:
                print("[WARNING] Введите число.")
        else:
            print("[WARNING] Нет лиц для удаления.")

cap.release()
cv2.destroyAllWindows()
