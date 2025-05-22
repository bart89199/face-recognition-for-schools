import os.path
import threading

import time
import uuid

import cv2
import face_recognition

import mediapipe as mp

import global_vars
from face_recogition_logic import get_locations_and_eyes, recognition, process_ready_faces, \
    clear_double_detection, get_locations_and_eyes
from frame_handler import get_rgb_frame
from loader import load_googleapi, load_known_data, load_arduino, load_mediapipe, load_main
from google_form_saver import load_data_from_forms
from saver import save_recognition


def clear():
    global_vars.eyes = [{}] + global_vars.eyes[:-1]
    global_vars.eyes_ready = False
    global_vars.raw_eyes = []


def process(frame):
    global_vars.iteration += 1
    clear()
    rgb_frame = get_rgb_frame(frame)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=get_rgb_frame(frame))

    timestamp_ms = int(time.time() * 1000)

    # results = global_vars.face_video_detector.detect_for_video(image, timestamp_ms)
    landmarkers = global_vars.landmarker.detect_for_video(image, timestamp_ms)
    face_locations, raw_eyes = get_locations_and_eyes(landmarkers, frame)

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations,
                                                     model=global_vars.FACE_RECOGNITION_MODEL)

    face_names = []

    for i, face_encoding in enumerate(face_encodings):
        name = recognition(face_encoding)

        face_names.append(name)

    face_names, (face_encodings, face_locations, raw_eyes) = clear_double_detection(face_names, [face_encodings, face_locations, raw_eyes])

    alive_names, save = process_ready_faces(frame, face_locations, face_names, raw_eyes)
    if save:
        th = threading.Thread(target=save_recognition, args=('recogn', frame))
        th.start()
    for name in alive_names:
        th = threading.Thread(target=save_recognition, args=(name, frame))
        th.start()

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

    if global_vars.FORMS_AUTOLOAD and cur_time - global_vars.last_forms_check_time >= global_vars.FORMS_CHECK_INTERVAL:
        th = threading.Thread(target=load_data_from_forms)
        th.start()
        global_vars.last_forms_check_time = 1e18

    while len(global_vars.frames_counter) > 1 and (cur_time - global_vars.frames_counter[0]) > 1:
        global_vars.frames_counter.pop(0)
    global_vars.frames_counter.append(cur_time)

    fps = len(global_vars.frames_counter) / (cur_time - global_vars.frames_counter[0])

    cv2.putText(frame, "fps: " + str(int(fps)), (500, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    global_vars.last_frame_time = cur_time
    if global_vars.RECORD_VIDEO:
        global_vars.out_video.write(frame)
    cv2.imshow(global_vars.WINDOW_NAME, frame)
    return face_encodings


def open_door():
    if global_vars.USE_ARDUINO:
        write_arduino(1)


def close_door():
    if global_vars.USE_ARDUINO:
        write_arduino(0)


def write_arduino(x):
    global_vars.arduino.write(bytes(str(x), 'utf-8'))


def main():
    load_main()
    load_mediapipe()
    load_googleapi()
    load_known_data()
    load_arduino()

    print("Starting video")
    cap = cv2.VideoCapture(global_vars.CAM_PORT)
    print("Video started")
    cv2.namedWindow(global_vars.WINDOW_NAME, cv2.WINDOW_NORMAL)

    if global_vars.RECORD_VIDEO:
        name = f'{uuid.uuid4()}.avi'
        os.makedirs(global_vars.VIDEOS_FOLDER, exist_ok=True)
        filepath = os.path.join(global_vars.VIDEOS_FOLDER, name)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        global_vars.out_video = cv2.VideoWriter(filepath, 0x44495658, global_vars.VIDEO_FPS, (frame_width, frame_height))

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
    close_door()
    if global_vars.RECORD_VIDEO:
        global_vars.out_video.release()


if __name__ == '__main__':
    main()
