import os.path
import pickle
import time
import uuid

import mediapipe as mp
import cv2
import shutil
import face_recognition
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

import global_vars
from frame_handler import get_rgb_frame, read_frame_file
from loader import load_known_data, load_googleapi, load_mediapipe
from face_recogition_logic import get_locations_and_eyes


def save_frame_on_disk(frame, filepath: str):
    save_result = cv2.imwrite(filepath, frame)
    return save_result


def save_recognition(name, frame):
    if not global_vars.SAVE_DETECTION_STATUS:
        return
    current_time = time.time()

    if not global_vars.last_saved_time.__contains__(name) or current_time - global_vars.last_saved_time[
        name] >= global_vars.SAVE_DELAY:
        timestamp = time.strftime("%Y%m%d_%H%M%S")  # Текущее время как часть имени файла
        filename = f"{name}_{timestamp}.jpg"

        filepath = os.path.join(global_vars.TEMP_PATH, filename)
        save_result = save_frame_on_disk(frame, filepath)

        if save_result:
            #      print(f"[INFO] Сохранено изображение {filename}")
            global_vars.last_saved_time[name] = current_time  # Обновляем время последнего сохранения

            try:
                # create drive api client
                drive_service = build("drive", "v3", credentials=global_vars.creds)
                file_metadata = {"name": filename, "parents": [global_vars.GOOGLE_DRIVE_FOLDER_ID]}
                media = MediaFileUpload(filepath, mimetype="image/jpeg")
                file = (
                    drive_service.files()
                    .create(body=file_metadata, media_body=media, fields="id")
                    .execute()
                )
                drive_service.close()

            #        print(f'Изображение отправлено на сервер. File ID: {file.get("id")}')

            except HttpError as error:
                print(f"An error occurred: {error}")

        else:
            print(f"[ERROR] Не удалось сохранить изображение в {filename}")


def save_data_on_disk():
    with open(global_vars.KNOWN_FACES_FILE, "wb") as f:
        pickle.dump((global_vars.known_face_encodings, global_vars.known_face_names, global_vars.known_face_images), f)
    with open(global_vars.SAVED_FORM_ANSWERS_FILE, "wb") as f:
        pickle.dump(global_vars.saved_form_answers, f)


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
            print(f'Folder for {name} don\'t exist')
        except OSError:
            print(f'Not all files for {name} deleted')
        global_vars.known_face_names.pop(idx)
        global_vars.known_face_encodings.pop(idx)
        global_vars.known_face_images.pop(idx)
    save_data_on_disk()


def save_from_encoding(face_encoding, name, frame, filename, additional_info=None):
    filepath = os.path.join(global_vars.SAVED_FRAMES_FOLDER, name)
    os.makedirs(filepath, exist_ok=True)
    savepath = os.path.join(filepath, filename)
    save_result = save_frame_on_disk(frame, savepath)
    filepath = os.path.join(filepath, filename)
    if not save_result:
        print(f'Can\'t save frame to folder(name = {name}, additional info = {additional_info}')
    if name not in global_vars.known_face_names:
        global_vars.known_face_names.append(name)
        global_vars.known_face_encodings.append([])
        global_vars.known_face_images.append([])

    index = global_vars.known_face_names.index(name)
    global_vars.known_face_encodings[index].append(face_encoding)
    global_vars.known_face_images[index].append(filepath)


def save_from_frame(frame, name: str, filename, additional_info=None):
    rgb_frame = get_rgb_frame(frame)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    landmarkers = global_vars.landmarker_image.detect(get_rgb_frame(image))
    face_locations, _ = get_locations_and_eyes(landmarkers, frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations,
                                                     model=global_vars.FACE_RECOGNITION_MODEL)
    if len(face_encodings) == 1:
        save_from_encoding(face_encodings[0], name, frame, filename, additional_info)
        save_data_on_disk()
        print(f"[INFO] Лицо {name} сохранено. additional info = {additional_info}")
    elif len(face_encodings) > 1:
        print(f'[WARNING] Несколько лиц для сохранения {name} additional info = {additional_info}')
    else:
        print(f"[WARNING] Нет лица для сохранения {name} additional info = {additional_info}")


def save_from_file(name, filepath, filename, additional_info=None):
    try:
        frame = read_frame_file(filepath)
        save_from_frame(frame, name, filename, additional_info)
    except FileNotFoundError:
        print(f"File not found(name = {name} filepath = {filepath}, additional_info = {additional_info}")


def save_from_file_and_move(name, path, filename, additional_info=None):
    filepath = os.path.join(global_vars.NEW_FRAMES_FOLDER, path)
    movefilepath = os.path.join(global_vars.OLD_FRAMES_FOLDER, path)
    try:
        frame = read_frame_file(filepath)
        save_from_frame(frame, name, filename, additional_info)
        shutil.move(filepath, movefilepath)
    except FileNotFoundError:
        print(f"File not found(name = {name} filepath = {filepath}, additional_info = {additional_info}")


def autosave():
    for (_, dirs, _) in os.walk(global_vars.NEW_FRAMES_FOLDER):
        for name in dirs:
            path = str(os.path.join(global_vars.NEW_FRAMES_FOLDER, name))
            movepath = str(os.path.join(global_vars.OLD_FRAMES_FOLDER, name))
            os.makedirs(movepath, exist_ok=True)
            for (_, _, file) in os.walk(path):
                for f in file:
                    if '.jpg' in f:
                        filepath = os.path.join(path, f)
                        frame = read_frame_file(filepath)
                        filename = name + "-" + str(uuid.uuid4()) + ".jpg"
                        save_from_frame(frame, name, filename)
                        movefilepath = os.path.join(movepath, f)
                        shutil.move(filepath, movefilepath)


def main_saver():
    from google_form_saver import get_forms_answers, forget_forms_response, load_data_from_forms, load_response

    load_known_data()
    load_mediapipe()
    load_googleapi()
    while True:
        key = input(
            "If you want save type - 's', delete frame - 'd', autoload - 'a', save from camera - 'c', forget form answer - 'ff', load all unloaded forms answers - 'af', load forms answer - 'lf', to exit - 'q'")

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
                filename = name + "-" + str(uuid.uuid4()) + ".jpg"
                save_from_file_and_move(name, path, filename)
        elif key == 'd':
            while True:
                if len(global_vars.known_face_names) > 0:
                    clean()
                    print("Доступные лица для удаления:")
                    for idx, name in enumerate(global_vars.known_face_names):
                        print(f"{idx + 1}. {name}")
                    choice = input("Введите номер лица для удаления ('all' - для удаления всех или 'q' для отмены): ")
                    if choice.lower() == 'q':
                        break
                    if choice.lower() == 'all':
                        while len(global_vars.known_face_names) > 0:
                            forget_all_faces(global_vars.known_face_names[0])
                            clean()
                        save_data_on_disk()
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
                                    save_data_on_disk()
                                    print(f"[INFO] Лицо '{deleted_name}' удалено.")
                                    break
                                try:
                                    j = int(choice) - 1
                                    if 0 <= j < len(global_vars.known_face_images[index]):
                                        name = global_vars.known_face_names[index]
                                        forget_face(name, j)
                                        save_data_on_disk()
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
            autosave()
        elif key == 'c':
            name = input("Enter name: ")
            print("Starting video")
            cap = cv2.VideoCapture(global_vars.CAM_PORT)
            print("Video started")
            cv2.namedWindow(global_vars.WINDOW_NAME, cv2.WINDOW_NORMAL)
            print("'q' - exit, 's' - save")
            while True:
                key = cv2.waitKey(1)
                ret, frame = cap.read()
                cv2.imshow(global_vars.WINDOW_NAME, frame)
                if not ret:
                    print("Something went wrong with camera")
                    break
                if key == ord('q'):
                    cv2.destroyWindow(global_vars.WINDOW_NAME)
                    break
                elif key == ord('s'):
                    filename = name + "-" + str(uuid.uuid4()) + ".jpg"
                    save_from_frame(frame, name, filename)
            cap.release()
        elif key == 'ff':
            while True:
                responses = get_forms_answers()['responses']
                clean()
                if len(responses) > 0:
                    print("Доступные ответы для удаления:")
                    for idx, response in enumerate(responses):
                        if global_vars.saved_form_answers.__contains__(response['responseId']):
                            print(f"{idx + 1}. {response['respondentEmail']}")
                    choice = input("Введите номер ответа для удаления ('all' - для удаления всех или 'q' для отмены): ")
                    if choice.lower() == 'q':
                        break
                    if choice.lower() == 'all':
                        delete_format = input(
                            "Enter '0' - to delete data and response id, '1' - to delete just data, '2' - to delete just response id")
                        for response in responses:
                            forget_forms_response(response, delete_format)
                        save_data_on_disk()
                        clean()
                        break
                    try:
                        index = int(choice) - 1
                        if 0 <= index < len(responses):
                            delete_format = input(
                                "Enter '0' - to delete data and response id, '1' - to delete just data, '2' - to delete just response id")
                            forget_forms_response(responses[index], delete_format)
                        else:
                            print("[WARNING] Неверный номер.")
                    except ValueError:
                        print("[WARNING] Введите корректный номер.")
                else:
                    print("[WARNING] Нет известных сохранённых форм для удаления.")
                    break
        elif key == 'af':
            load_data_from_forms()
        elif key == 'lf':
            while True:
                responses = get_forms_answers()['responses']
                clean()
                if len(responses) > 0:
                    print("Доступные ответы для добавления:")
                    for idx, response in enumerate(responses):
                        print(f"{idx + 1}. {response['respondentEmail']}")
                    choice = input(
                        "Введите номер ответа для добавления ('all' - для добавления всех или 'q' для отмены): ")
                    if choice.lower() == 'q':
                        break
                    if choice.lower() == 'all':
                        save_format = input(
                            "Enter '0' - to save data and response id, '1' - to save just data, '2' - to save just response id")
                        for response in responses:
                            load_response(response, False, save_format)
                        break
                    try:
                        index = int(choice) - 1
                        if 0 <= index < len(responses):
                            save_format = input(
                                "Enter '0' - to save data and response id, '1' - to save just data, '2' - to save just response id")
                            load_response(responses[index], False, save_format)
                        else:
                            print("[WARNING] Неверный номер.")
                    except ValueError:
                        print("[WARNING] Введите корректный номер.")
                else:
                    print("[WARNING] Нет известных форм.")
                    break

    clean()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main_saver()
