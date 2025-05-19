import threading

import time

import cv2
import face_recognition

import mediapipe as mp

import global_vars
from face_recogition_logic import get_locations, process_eyes, recognition, analyze_eyes, process_ready_faces
from loader import load_googleapi, load_known_data, load_arduino, load_mediapipe_video, load_mediapipe_image
from google_form_saver import load_data_from_forms
from saver import save_recognition


def clear():
    global_vars.eyes = [{}] + global_vars.eyes[:-1]
    global_vars.eyes_ready = False
    global_vars.raw_eyes = []


def process(frame):
    global_vars.iteration += 1
    clear()
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    timestamp_ms = int(time.time() * 1000)

    results = global_vars.face_video_detector.detect_video(image, timestamp_ms)
    face_locations = get_locations(results, frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations, model=global_vars.FACE_RECOGNITION_MODEL)

    landmarkers = global_vars.landmarker.detect_for_video(image, timestamp_ms)
    raw_eyes = process_eyes(landmarkers, frame)

    face_names = []

    for i, face_encoding in enumerate(face_encodings):
        name = recognition(face_encoding)

        face_names.append(name)

    analyze_eyes(raw_eyes, face_names, face_locations)

    alive_names = process_ready_faces(frame, face_locations, face_names)

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
    load_mediapipe_video()
    load_mediapipe_image()
    load_googleapi()
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
    close_door()


if __name__ == '__main__':
    main()
