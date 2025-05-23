import threading

from enum import Enum
import cv2
import face_recognition
import numpy as np

from mediapipe.tasks.python.components.containers import DetectionResult
from mediapipe.tasks.python.vision import FaceLandmarkerResult

import global_vars


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


# def process_eyes(result: FaceLandmarkerResult, frame):
#     h, w, _ = frame.shape
#
#     raw_eyes = []
#     for landmarks in result.face_landmarks:
#         left_eye = eye_aspect_ratio(landmarks, global_vars.LEFT_EYE, w, h)
#         right_eye = eye_aspect_ratio(landmarks, global_vars.RIGHT_EYE, w, h)
#         avg_ear = (left_eye + right_eye) / 2.0
#
#         pos_x = 0
#         pos_y = 0
#         for p in landmarks:
#             pos_x += p.x * w
#             pos_y += p.y * h
#
#         pos_x /= len(landmarks)
#         pos_y /= len(landmarks)
#
#         raw_eyes.append((pos_x, pos_y, avg_ear))
#     return raw_eyes
#

# def get_locations(results: DetectionResult, frame):
#     face_locations = []
#     if results.detections:
#         h, w, _ = frame.shape
#         for detection in results.detections:
#             bbox = detection.bounding_box
#
#             mid_x = bbox.origin_x + bbox.width // 2
#             mid_y = bbox.origin_y + bbox.height // 2
#
#             nh = (bbox.height * global_vars.FRAME_SCALE_HEIGHT) // 2
#             nw = (bbox.width * global_vars.FRAME_SCALE_WIDTH) // 2
#
#             left = int(mid_x - nw)
#             top = int(mid_y - nh)
#             right = int(mid_x + nw)
#             bottom = int(mid_y + nh)
#
#             top = max(top, 0)
#             left = max(left, 0)
#             bottom = min(bottom, h)
#             right = min(right, w)
#
#             face_locations.append((top, right, bottom, left))
#     return face_locations

def get_locations_and_eyes(results: FaceLandmarkerResult, frame):
    h, w, _ = frame.shape
    face_locations = []
    raw_eyes = []
    for landmarks in results.face_landmarks:
        left_eye = eye_aspect_ratio(landmarks, global_vars.LEFT_EYE, w, h)
        right_eye = eye_aspect_ratio(landmarks, global_vars.RIGHT_EYE, w, h)
        avg_eyes = (left_eye + right_eye) / 2.0
        raw_eyes.append(avg_eyes)
        left = 1e9
        right = 0
        top = 1e9
        bottom = 0
        for p in landmarks:
            x = p.x * w
            y = p.y * h
            left = int(min(left, x))
            right = int(max(right, x))
            top = int(min(top, y))
            bottom = int(max(bottom, y))

            top = max(top, 0)
            left = max(left, 0)
            bottom = min(bottom, h)
            right = min(right, w)

        face_locations.append((top, right, bottom, left))
    return face_locations, raw_eyes


def check_encoding_avg(face_encoding):
    mx = [0, 0]
    for idx, known_faces in enumerate(global_vars.known_face_encodings):
        if len(known_faces) == 0:
            continue
        distances = face_recognition.face_distance(known_faces, face_encoding)
        avg = np.mean(distances)
        if global_vars.MAX_AVG_DISTANCE >= avg > mx[0]:
            mx = [avg, idx]
    name = global_vars.UNKNOWN_NAME
    best_match_index = mx[1]
    if mx[0] != 0:
        name = global_vars.known_face_names[best_match_index] + "(" + str(mx[0]) + ")"
    return name


def check_encoding_percent(face_encoding):
    mx = [0, 0, 0]
    for idx, known_faces in enumerate(global_vars.known_face_encodings):
        if len(known_faces) == 0:
            continue
        distances = face_recognition.face_distance(known_faces, face_encoding)
        cnt = 0
        for d in distances:
            if d <= global_vars.MAX_PERCENT_DISTANCE:
                cnt += 1
        percent = cnt / len(known_faces)
        avg = np.mean(distances)
        if mx[0] < percent and percent >= global_vars.MIN_MATCH_FOR_PERSON:
            mx = [percent, avg, idx]
        if mx[0] == percent and percent >= global_vars.MIN_MATCH_FOR_PERSON and mx[1] > avg:
            mx = [percent, avg, idx]
    name = global_vars.UNKNOWN_NAME
    best_match_index = mx[2]
    if mx[0] != 0:
        name = global_vars.known_face_names[best_match_index] + "(" + str(mx[0]) + ")"
    return name


def recognition(face_encoding):
    if global_vars.FACE_DETECTION_MODE == 1:
        return check_encoding_avg(face_encoding)
    else:
        return check_encoding_percent(face_encoding)


def clear_double_detection(face_names, face_metrics):
    already = []
    delete = []
    for i in range(len(face_names) - 1, -1, -1):
        name = face_names[i]
        if name == global_vars.UNKNOWN_NAME:
            continue
        if already.__contains__(name):
            delete.append(name)
        already.append(name)


    for i in range(len(face_names) - 1, -1, -1):
        name = face_names[i]
        if name == global_vars.UNKNOWN_NAME:
            continue
        if delete.__contains__(name):
            face_names.pop(i)
            for j in range(0, len(face_metrics)):
                face_metrics[j].pop(i)

    return face_names, face_metrics




def check_face(name, avg):
    if name == global_vars.UNKNOWN_NAME:
        return False, False, False
    difs = 0
    cnt = 0
    blinks = 0
    detect_cnt = 0
    for i in range(1, global_vars.LAST_FRAMES_AMOUNT):
        if not global_vars.eyes[i].__contains__(name):
            break
        detect_cnt += 1

    for i in range(1, global_vars.FRAMES_FOR_EYES_CHECK):
        if not global_vars.eyes[i].__contains__(name):
            break
        cnt += 1
        avgd, bls = global_vars.eyes[i][name]
        blinks += bls
        dif = abs(avg - avgd)
        if dif >= global_vars.MIN_EYES_DIFFERENCE:
            difs += 1
    blinked = cnt > 0 and (difs / cnt) > global_vars.MIN_DIFS_FOR_BLICK
    blinks += blinked
    global_vars.eyes[0][name] = (avg, blinked)
    return detect_cnt >= global_vars.MIN_FRAMES_FOR_DETECTION, blinks >= global_vars.NEED_BLINKS, avg <= global_vars.CLOSE_EYES_THRESHOLD


# class Recognition_status(Enum):
#     unknown = (0, 0, 0)
#     recognized = (1, 0, 0)
#     blinked = (1, 0, 1)
#     alive = (1, 1, 0)
#     alive_blinked = (1, 1, 1)


def process_ready_faces(frame, face_locations, face_names, raw_eyes):


    alive_names = []
    save = False
    new_counter = {}
    for (top, right, bottom, left), name, avg_eyes in zip(face_locations, face_names, raw_eyes):
        color = (0, 0, 255)
        recogn, alive, blinked = check_face(name, avg_eyes)
        if recogn:
            save = True
            color = (230, 224, 76)
            if alive:
                color = (0, 255, 0)
            if blinked:
                color = (255, 33, 170)
            if alive and (not blinked or global_vars.BLINKED_EYES_OPEN):
                new_counter[name] = (global_vars.recognition_count[
                                         name] if global_vars.recognition_count.__contains__(name) else 0) + 1
                if new_counter[name] >= global_vars.WAIT_FRAMES_FOR_DETECTION:
                    alive_names.append(name)
        else:
            if name != global_vars.UNKNOWN_NAME:
                color = (80, 127, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 5, bottom - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    global_vars.recognition_count = new_counter
    return alive_names, save
