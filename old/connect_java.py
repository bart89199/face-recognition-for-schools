import socket
import struct
import time
from array import array, ArrayType
from time import sleep

import cv2
import numpy as np


HOST = 'localhost'
PORT = 8082
java_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def example_connection_java():
    msg = receive_string_from_java()
    if msg == "Start Test":
        test_java_connection()
        close_java_connection()
        return

    if msg != "work":
        close_java_connection()
        print("Bad instruction: " + msg)
        return

    while True:
        try:
            msg = receive_string_from_java()
            if msg == "test message":
                send_string_to_java("I like bananas")
        except ConnectionResetError:
            print("Connection reset by peer")
            break

    close_java_connection()

def connect_java():
    global java_socket
    while True:
        try:
            java_socket.connect((HOST, PORT))
        except ConnectionRefusedError:
            sleep(3)

def close_java_connection():
    global java_socket
    java_socket.close()

def send_staff(size: ArrayType[int], send_size: bool = True, message: str | None = None):
    if message is not None:
        send_string_to_java(message, False)
    if send_size:
        send_to_java(size.tobytes(), False)


def send_to_java(byte_array: bytes, send_size: bool = True, message: str | None = None):

    send_staff(array('i', [len(byte_array)]), send_size, message)

    global java_socket
    java_socket.sendall(byte_array)

def send_frame_to_java(frame: np.ndarray, send_size: bool = True, message: str | None = None):
    _, encoded = cv2.imencode('.jpg', frame)
    data = encoded.tobytes()

    send_to_java(data, send_size, message)
    # height, width, _ = frame.shape
    # pixels = bytearray(height * width * 3)
    #
    # send_staff(array('i', [height, width]), send_size, message)
    #
    # for i in range(height):
    #     for j in range(width):
    #         for k in range(3):
    #             pixels[(i * width + j) * 3 + k] = frame[i][j][k]
    #
    # send_to_java(array('i', [height, width]).tobytes(), False)
    # send_to_java(pixels, False)

def send_string_to_java(string: str, send_size: bool = True, message: str | None = None):

    send_staff(array('i', [len(string)]), send_size, message)

    send_to_java(string.encode('utf-8'), False)

def receive_bytes_from_java(size: int | None = None):

    if size is None:
        size = receive_ints_from_java(1)[0]

    global java_socket
    data = java_socket.recv(size)
    while len(data) < size:
        data += java_socket.recv(size - len(data))

    return data



def receive_ints_from_java(size: int | None = None):
    if size is None:
        size = receive_ints_from_java(1)[0]

    response = receive_bytes_from_java(size * 4)
    result = array('i', [0] * size)
    for i in range(size):
        result[i] = response[i * 4] | (response[i * 4 + 1] << 8) | (response[i * 4 + 2] << 16) | (response[i * 4 + 3] << 24)
    return result

def receive_frame_from_java():
    size = receive_ints_from_java(1)[0]
    bytes = receive_bytes_from_java(size)
    np_data = np.frombuffer(bytes, dtype=np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    return img

# def receive_frame_from_java(height: int | None = None, width: int | None = None):
    # if height is None or width is None:
    #     height, width = receive_ints_from_java(2)
    #
    # pixels = receive_bytes_from_java(height * width * 3)
    #
    # frame = np.ndarray((height, width, 3), dtype=np.uint8)
    # for i in range(height):
    #     for j in range(width):
    #         for k in range(3):
    #             frame[i][j][k] = pixels[(i * width + j) * 3 + k]
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # return frame


def receive_string_from_java(size: int | None = None):
    if size is None:
        size = receive_ints_from_java(1)[0]
    data = receive_bytes_from_java(size)
    return data.decode('utf-8')


def get_rgb_frame(frame: np.ndarray):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

test_bytes = bytearray([1, 10, 100, 254, 255])
test_ints = array('i', [1, 10, 100, 12345678, 123456789])
test_frame = cv2.imread("../test-img.png")

def test_java_connection():
    start_time = time.time()
    send_string_to_java("ok")

    recv_bytes = receive_bytes_from_java()
    if recv_bytes == test_bytes:
        send_string_to_java("ok")
    else:
        send_string_to_java("error, get " + str(recv_bytes))
        return

    recv_ints = receive_ints_from_java()
    if recv_ints == test_ints:
        send_string_to_java("ok")
    else:
        send_string_to_java("error, get " + str(recv_ints))
        return

    recv_frame = receive_frame_from_java()



    if checkDist(recv_frame, test_frame):
        send_string_to_java("ok")
    else:
        send_string_to_java("er, check recv.jpg")
        cv2.imwrite("recv.jpg", recv_frame)
        return

    send_frame_to_java(test_frame)

    send_to_java(test_bytes)

    send_to_java(test_ints.tobytes())

    spent_time = int((time.time() - start_time) * 1000)
    send_to_java(array('i', [spent_time]).tobytes())

def checkDist(frame1, frame2, max_dist: int = 15, max_bad = 0.1) -> bool:
    if len(frame1) != len(frame2) or len(frame1[0]) != len(frame2[0]):
        return False
    bad = 0
    for i in range(len(frame1)):
        for j in range(len(frame1[i])):
            for k in range(len(frame1[i][j])):
                if abs(int(frame1[i][j][k]) - int(frame2[i][j][k])) > max_dist:
                    bad += 1
    return (bad / (len(frame1) * len(frame1[0]) * 3)) <= max_bad


HOST = 'localhost'
PORT = 8082
java_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def main():
    # cap = cv2.VideoCapture(0)
    #
    # ret, frame = cap.read()
    # cv2.imwrite("test-img.png", frame)
    # return
    # Инициализация JavaGateway
    # gateway = JavaGateway()

    # connect_java()


    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit(0)

    cv2.imshow('frame',frame)

    connect_java()

    send_to_java(bytearray([1, 188, 255]))
    send_to_java(array('i', [10, 12345678, 123456789]).tobytes())

    r1 = None
    r2 = None
    try:
        r2 = receive_ints_from_java(3)
        r1 = receive_bytes_from_java(3)
    except:
        print("a")
    rr1 = ""
    for x in r1:
        rr1 += str(x) + " "
    print(rr1)
    print(r2)



    close_java_connection()


    # Получение ответа: 4 байта = int32





    # print("Starting...")
    # java_list = ListConverter().convert(pixels, gateway._gateway_client)
    # gateway.entry_point.fromBytes(height, width, java_list)
    # print("Completed")

    while True:
        if cv2.waitKey(1) != ord('q'):
            break




    # Получение доступа к объекту класса Dict



if __name__ == "__main__":
    main()