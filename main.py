import asyncio
import os.path
import threading
import time
from datetime import datetime
from time import sleep

import cv2
import face_recognition
import mediapipe as mp
import numpy as np

import kotlin_connection
import load_settings
import loader
import settings
import streaming
from face_recogition_logic import recognition, process_ready_faces, \
    clear_double_detection, get_locations_and_eyes
from frame_handler import get_rgb_frame
from google_form_saver import load_data_from_forms
from loader import load_googleapi, load_known_data, load_mediapipe, load_main
from settings import SystemStatus


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
    # if save:
    #     th = threading.Thread(target=save_recognition, args=('recogn', frame))
    #     th.start()
    # for name in alive_names:
    #     th = threading.Thread(target=save_recognition, args=(name, frame))
    #     th.start()

    for name in alive_names:
        settings.cur_alive_names.add(name)
    if len(alive_names) > 0:
        settings.last_blink_time = time.time()

    if time.time() - settings.last_blink_time < settings.CLOSE_DELAY_MS / 1000:
        settings.door_opened = True
        cv2.putText(frame, "People Allowed", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        settings.door_opened = False
        cv2.putText(frame, "People Not Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cur_time = time.time()

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

    return frame


async def start_system():
    load_settings.load()

    load_main()
    load_mediapipe()
    load_googleapi()
    load_known_data()

    print("Starting video")
    loader.setup_cap()
    print("Video started")
    cv2.namedWindow(settings.WINDOW_NAME, cv2.WINDOW_NORMAL)

    if settings.RECORD_VIDEO:
        name = f'{datetime.now().strftime("%Y.%m.%d %H:%M:%S")}.avi'
        os.makedirs(settings.VIDEOS_FOLDER, exist_ok=True)
        filepath = os.path.join(settings.VIDEOS_FOLDER, name)
        frame_width = int(settings.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(settings.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        settings.out_video = cv2.VideoWriter(filepath, 0x44495658, settings.VIDEO_FPS, (frame_width, frame_height))

    await asyncio.sleep(3)
    print("[INFO] 'q' чтобы выйти.")
    settings.system_status = SystemStatus.RUNNING
    while True:
        if time.time() - settings.last_settings_load >= 30.0:
            prev_cam = settings.CAM_PORT
            load_settings.load()
            if prev_cam != settings.CAM_PORT:
                loader.setup_cap()
            settings.last_settings_load = time.time()

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        ret, frame = settings.cap.read()
        if not ret:
            print("Something went wrong with camera")
            frame = np.zeros((settings.VIDEO_WIDTH, settings.VIDEO_HEIGHT, 3), np.uint8)
            cv2.putText(frame, "Camera Error", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            start_time = time.time()
            while time.time() - start_time < 1:
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                await asyncio.sleep(0.1)
            loader.setup_cap()
        else:
            frame = process(frame)

        if settings.RECORD_VIDEO:
            settings.out_video.write(frame)
        cv2.imshow(settings.WINDOW_NAME, frame)

        if settings.stream_is_run and (not settings.frame_queue.full()):
            await settings.frame_queue.put(frame)
        await asyncio.sleep(0.001)

    settings.door_opened = False
    settings.system_status = SystemStatus.STOPPING
    settings.stream_is_run = False
    await asyncio.sleep(1)
    frame = np.zeros((settings.VIDEO_WIDTH, settings.VIDEO_HEIGHT, 3), np.uint8)
    cv2.putText(frame, "System stopped", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    if settings.frame_queue.full():
        settings.frame_queue.get_nowait()
    await settings.frame_queue.put(frame)
    await asyncio.sleep(3)

    print("Can releasing...")
    settings.cap.release()
    cv2.destroyAllWindows()
    print("Stoping recording...")
    if settings.RECORD_VIDEO:
        settings.out_video.release()
    print("All stopped...")

async def main():

    ff_task = asyncio.create_task(streaming.start_ffmpeg_writer())
    main_task = asyncio.create_task(start_system())
    kotlin_connection_task = asyncio.create_task(kotlin_connection.send_data_loop())
    try:
        await asyncio.gather(main_task, ff_task, kotlin_connection_task)
    except asyncio.CancelledError:
        pass


if __name__ == '__main__':
    asyncio.run(main())
