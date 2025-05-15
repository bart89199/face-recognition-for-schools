import os.path
import time

import mediapipe as mp
import cv2
import shutil
import face_recognition
import serial
from matplotlib.image import imread

import global_vars
from main import load_mediapipe
from main import load_known_data
from main import get_locations
from main import save_data
from main import save



def forget_face(name, j):
    try:
        i = global_vars.known_face_names.index(name)
        filepath = global_vars.known_face_images[i].pop(j)
        global_vars.known_face_encodings[i].pop(j)
        os.remove(filepath)
    except IndexError and ValueError:
        print(f'Can\'t remove face {name} {j}')

def forget_all_faces(name):
    try:
        i = global_vars.known_face_names.index(name)
        while len(global_vars.known_face_images[i]) > 0:
            forget_face(name, 0)
    except ValueError:
        print(f'Can\'t find face {name}')

def clean():
    rem = []
    for idx, name in enumerate(global_vars.known_face_names):
        if len(global_vars.known_face_encodings[idx]) == 0:
            rem.append((idx, name))
    rem.reverse()
    for idx, name in rem:
        try:
            os.rmdir(os.path.join(global_vars.SAVED_FRAMES_FOLDER, name))
        except FileNotFoundError:
            print(f'Folder for {name} didn\'t exist')
        except OSError:
            print(f'Not all files for {name} deleted')
        global_vars.known_face_names.pop(idx)
        global_vars.known_face_encodings.pop(idx)
        global_vars.known_face_images.pop(idx)
    save_data()


def save_from_file(filepath: str, name: str):
    frame = imread(filepath)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    face_locations = get_locations(frame, image)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model=global_vars.FACE_RECOGNITION_MODEL)
    if face_encodings:
        save(face_encodings[0], name, frame)
        save_data()
        print(f"[INFO] Лицо {name} сохранено.")
    else:
        print("[WARNING] Нет лица для сохранения!")

def main_saver():

    load_mediapipe()
    load_known_data()
    while True:
        key = input("If you want save type - 's', delete frame - 'd', autoload - 'a', to exit - 'q'")

        if key == 'q':
            break
        elif key == 's':
            name = input("Введите имя для лица: ")
            while name == global_vars.UNKNOWN_NAME:
                print("Это имя зарезервировано")
                name = input("Введите имя для лица: ")
            while True:
                path = input("Now type file path(type 'q' to quite)")
                if path == 'q':
                    break
                filepath = os.path.join(global_vars.NEW_FRAMES_FOLDER, path)
                movefilepath = os.path.join(global_vars.OLD_FRAMES_FOLDER, path)
                try:
                    save_from_file(filepath, name)
                    shutil.move(filepath, movefilepath)
                except FileNotFoundError:
                    print("File not found")
        elif key == 'd':
            while True:
                if len(global_vars.known_face_names) > 0:
                    clean()
                    print("Доступные лица для удаления:")
                    for idx, name in enumerate(global_vars.known_face_names):
                        print(f"{idx + 1}. {name}")
                    choice = input("Введите номер лица для удаления (или 'q' для отмены): ")
                    if choice.lower() == 'q':
                        break
                    try:
                        index = int(choice) - 1
                        if 0 <= index < len(global_vars.known_face_names):
                            while True:
                                print("Choose frame to delete('q' - to quite, 'all' - to delete all):")
                                for idx, path in enumerate(global_vars.known_face_images[index]):
                                    print(f'{idx + 1} - {path}')
                                choice = input()
                                if choice == 'q':
                                    break
                                if choice == 'all':
                                    deleted_name = global_vars.known_face_names[index]
                                    forget_all_faces(deleted_name)
                                    save_data()
                                    print(f"[INFO] Лицо '{deleted_name}' удалено.")
                                    break
                                try:
                                    j = int(choice) - 1
                                    if 0 <= j < len(global_vars.known_face_images[index]):
                                        name = global_vars.known_face_names[index]
                                        forget_face(name, j)
                                        save_data()
                                        print(f"[INFO] Frame deleted.")
                                    else:
                                        print("[WARNING] Неверный номер.")
                                except ValueError:
                                    print("[WARNING] Введите корректный номер.")
                        else:
                            print("[WARNING] Неверный номер.")
                    except ValueError:
                        print("[WARNING] Введите корректный номер.")
                else:
                    print("[WARNING] Нет известных лиц для удаления.")
                    break
        elif key == 'a':
            for (_, dirs, _) in os.walk(global_vars.NEW_FRAMES_FOLDER):
                for name in dirs:
                    path = str(os.path.join(global_vars.NEW_FRAMES_FOLDER, name))
                    movepath = str(os.path.join(global_vars.OLD_FRAMES_FOLDER, name))
                    os.makedirs(movepath, exist_ok=True)
                    for (_, _, file) in os.walk(path):
                        for f in file:
                            if '.jpg' in f:
                                filepath = os.path.join(path, f)
                                save_from_file(filepath, name)
                                movefilepath = os.path.join(movepath, f)
                                shutil.move(filepath, movefilepath)
    clean()

if __name__ == '__main__':
    main_saver()
