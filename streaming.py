
import asyncio
import os
import shutil
import subprocess
import cv2

import settings


# ffmpeg command: читаем rawvideo из stdin и пишем HLS
FFMPEG_CMD = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-pixel_format', 'bgr24',
    '-video_size', f'{settings.VIDEO_WIDTH}x{settings.VIDEO_HEIGHT}',
    '-framerate', str(settings.VIDEO_FPS),
    '-i', 'pipe:0',
    '-c:v', 'libx264',
    '-preset', 'veryfast',
    '-g', '50',
    '-sc_threshold', '0',
    '-profile:v', 'baseline',
    '-pix_fmt', 'yuv420p',
    '-f', 'hls',
    '-hls_time', '1',  # длительность сегмента (сек) -> влияет на задержку
    '-hls_list_size', '2',
    '-hls_flags', 'delete_segments',
    os.path.join(settings.HLS_DIR, settings.PLAYLIST)
]


async def start_ffmpeg_writer():

    if os.path.exists(settings.HLS_DIR):
        shutil.rmtree(settings.HLS_DIR)
    os.makedirs(settings.HLS_DIR, exist_ok=True)

    settings.stream_is_run = True

    print("Starting hls")

    proc = await asyncio.create_subprocess_exec(
        *FFMPEG_CMD,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    print("Hls started")


    try:
        while settings.stream_is_run:
            frame = await settings.frame_queue.get()

            if frame is None:
                continue
            # Ensure shape and dtype
            frame = cv2.resize(frame, (settings.VIDEO_WIDTH, settings.VIDEO_HEIGHT))

            # put frame into queue (bytes will be read by ffmpeg writer)
            proc.stdin.write(frame.tobytes())
            await proc.stdin.drain()
    finally:
        print("Hls stopped")
        proc.stdin.close()
        await proc.wait()


# async def capture_loop(frame_queue: asyncio.Queue):
#     print("Video is starting...")
#     cap = cv2.VideoCapture(1)
#     print("Video started")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             await asyncio.sleep(0.1)
#             continue
#         # Ensure shape and dtype
#         frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break
#         cv2.imshow("lol", frame)
#         # put frame into queue (bytes will be read by ffmpeg writer)
#         await frame_queue.put(frame)
#         await asyncio.sleep(0)  # give control


async def main():
    # frame_queue = asyncio.Queue(maxsize=10)
    # старт ffmpeg writer
    ff_task = asyncio.create_task(start_ffmpeg_writer())
    # cap_task = asyncio.create_task(capture_loop(frame_queue))

    try:
        await asyncio.gather(ff_task)
    except asyncio.CancelledError:
        pass


if __name__ == '__main__':
    asyncio.run(main())