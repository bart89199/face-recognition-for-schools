import time

import cv2

import global_vars

print("Starting video")
cap = cv2.VideoCapture('/dev/video0')
print("Video started")

while True:
    ret, frame = cap.read()
    if not ret:
        exit(6)
    cur_time = time.time()
    while len(global_vars.frames_counter) > 1 and (cur_time - global_vars.frames_counter[0]) > 1:
        global_vars.frames_counter.pop(0)
    global_vars.frames_counter.append(cur_time)
    fps = len(global_vars.frames_counter) / (cur_time - global_vars.frames_counter[0])

    cv2.putText(frame, "fps: " + str(int(fps)), (500, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("test", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
