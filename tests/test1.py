import cv2

print("Starting video")
cap = cv2.VideoCapture(1)
print("Video started")

while True:
    ret, frame = cap.read()
    if not ret:
        exit(6)
    cv2.imshow("Распознавание лиц", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
