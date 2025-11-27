import time

import cv2

import settings

print("Starting video")
cap = cv2.VideoCapture('/dev/video0')
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FPS, settings.VIDEO_FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.VIDEO_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.VIDEO_HEIGHT)
print("Video started")

while True:
    ret, frame = cap.read()
    if not ret:
        exit(6)
    cur_time = time.time()
    while len(settings.frames_counter) > 1 and (cur_time - settings.frames_counter[0]) > 1:
        settings.frames_counter.pop(0)
    settings.frames_counter.append(cur_time)
    fps = len(settings.frames_counter) / (cur_time - settings.frames_counter[0])

    cv2.putText(frame, "fps: " + str(int(fps)), (500, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("test", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
