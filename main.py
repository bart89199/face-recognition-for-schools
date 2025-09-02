import asyncio
import os.path
import threading
import time
from datetime import datetime

import cv2
import face_recognition
import mediapipe as mp
from serial.serialutil import SerialException

import door
import kotlin_connection
import load_settings
import settings
import streaming
from face_recogition_logic import recognition, process_ready_faces, \
    clear_double_detection, get_locations_and_eyes
from frame_handler import get_rgb_frame
from google_form_saver import load_data_from_forms
from loader import load_googleapi, load_known_data, load_arduino, load_mediapipe, load_main
from saver import save_recognition


def clear():
    settings.eyes = [{}] + settings.eyes[:-1]
    settings.eyes_ready = False
    settings.raw_eyes = []


def process(frame):
    settings.iteration += 1
    clear()
    rgb_frame = get_rgb_frame(frame)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=get_rgb_frame(frame))

    timestamp_ms = int(time.time() * 1000)

    # results = settings.face_video_detector.detect_for_video(image, timestamp_ms)
    landmarkers = settings.landmarker.detect_for_video(image, timestamp_ms)
    face_locations, raw_eyes = get_locations_and_eyes(landmarkers, frame)

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations,
                                                     model=settings.FACE_RECOGNITION_MODEL)

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
        settings.last_blink_time = time.time()


    if time.time() - settings.last_blink_time < settings.CLOSE_DELAY_MS / 1000:
        settings.door_status = True
        cv2.putText(frame, "People Allowed", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        settings.door_status = False
        cv2.putText(frame, "People Not Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cur_time = time.time()

    if settings.CONNECT_KOTLIN:
        java_server.send_data(settings.door_status, alive_names)

    if settings.door_status:
        open_door()
    else:
        close_door()

    if settings.FORMS_AUTOLOAD and cur_time - settings.last_forms_check_time >= settings.FORMS_CHECK_INTERVAL_MS / 1000:
        th = threading.Thread(target=load_data_from_forms)
        th.start()
        settings.last_forms_check_time = 1e18

    while len(settings.frames_counter) > 1 and (cur_time - settings.frames_counter[0]) > 1:
        settings.frames_counter.pop(0)
    settings.frames_counter.append(cur_time)

    fps = len(settings.frames_counter) / (cur_time - settings.frames_counter[0])

    cv2.putText(frame, "fps: " + str(int(fps)), (500, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    settings.last_frame_time = cur_time
    if settings.RECORD_VIDEO:
        settings.out_video.write(frame)
    cv2.imshow(settings.WINDOW_NAME, frame)

    return frame


def open_door():
    if settings.USE_ARDUINO:
        write_arduino(1)


def close_door():
    if settings.USE_ARDUINO:
        write_arduino(0)


def write_arduino(x):
    if not door.arduino_loaded:
        load_arduino()

    try:
        if settings.cur_arduino != x:
            door.arduino.write(bytes(str(x), 'utf-8'))
            settings.cur_arduino = x
    except SerialException as e:
        print("Can't connect arduino " + str(e))
        load_arduino()


async def start_system():
    load_settings.load()

    if settings.USE_ARDUINO:
        load_arduino()

    load_main()
    load_mediapipe()
    load_googleapi()
    load_known_data()

    print("Starting video")
    cap = cv2.VideoCapture(settings.CAM_PORT)
    cap.set(cv2.CAP_PROP_FPS, settings.VIDEO_FPS)
    print("Video started")
    cv2.namedWindow(settings.WINDOW_NAME, cv2.WINDOW_NORMAL)

    if settings.RECORD_VIDEO:
        name = f'{datetime.now().strftime("%Y.%m.%d %H:%M:%S")}.avi'
        os.makedirs(settings.VIDEOS_FOLDER, exist_ok=True)
        filepath = os.path.join(settings.VIDEOS_FOLDER, name)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        settings.out_video = cv2.VideoWriter(filepath, 0x44495658, settings.VIDEO_FPS, (frame_width, frame_height))

    print("[INFO] 'q' чтобы выйти.")
    while True:
        if time.time() - settings.last_settings_load >= 30.0:
            load_settings.load()
            settings.last_settings_load = time.time()

        key = cv2.waitKey(1)
        ret, frame = cap.read()
        if not ret:
            print("Something went wrong with camera")
            break
        frame = process(frame)
        await settings.frame_queue.put(frame)
        await asyncio.sleep(0)
        if key == ord('q'):
            break

    settings.stream_is_run = False
    settings.frame_queue.task_done()
    cap.release()
    cv2.destroyAllWindows()
    close_door()
    if settings.RECORD_VIDEO:
        settings.out_video.release()

async def main():

    ff_task = asyncio.create_task(streaming.start_ffmpeg_writer())
    main_task = asyncio.create_task(start_system())
    try:
        await asyncio.gather(main_task, ff_task)
    except asyncio.CancelledError:
        pass


if __name__ == '__main__':
    asyncio.run(main())
