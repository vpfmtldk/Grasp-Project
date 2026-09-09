"""
cam_view.py -- live view of the USB camera in an OpenCV window.

  python robot/cam_view.py                 # camera 2 (Innomaker), 1280x720
  keys:  s = save a snapshot to output/cam_snapshots/
         SPACE = save + also overwrite output/cam_latest.png
         q / ESC = quit
"""
import argparse
import os
import time

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=2)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()

    snap_dir = "output/cam_snapshots"
    os.makedirs(snap_dir, exist_ok=True)

    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW if os.name == "nt" else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    for _ in range(10):
        cap.read()
    if not cap.isOpened():
        print(f"camera {a.camera} did not open"); return

    win = "USB camera (s=save  SPACE=save+latest  q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, a.width, a.height)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame grab failed"); break
        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k in (ord("s"), 32):
            p = os.path.join(snap_dir, time.strftime("shot_%Y%m%d_%H%M%S.png"))
            cv2.imwrite(p, frame)
            if k == 32:
                cv2.imwrite("output/cam_latest.png", frame)
            n += 1
            print("saved", p, "(latest updated)" if k == 32 else "")
    cap.release()
    cv2.destroyAllWindows()
    print(f"done, {n} snapshots in {snap_dir}")


if __name__ == "__main__":
    main()
