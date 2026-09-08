"""Calibrate the webcam using a printed chessboard pattern.

Example for a board with 9 x 6 INNER corners and 24 mm squares:
    python camera_calibration.py --cols 9 --rows 6 --square-size 0.024

Keys:
    S - save the currently detected chessboard view
    C - calculate and save camera_calibration.npz (after at least 12 views)
    Q / ESC - quit
"""

import argparse
from getpass import getpass
import os
from urllib.parse import quote

import cv2
import numpy as np


def open_camera(source: str, width: int, height: int, rtsp_transport: str):
    if source.isdecimal():
        camera = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera = cv2.VideoCapture(int(source))
    else:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{rtsp_transport}"
        camera = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not camera.isOpened():
            camera = cv2.VideoCapture(source)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return camera


def hikvision_rtsp_url(host: str, username: str, password: str, channel: int) -> str:
    return (
        f"rtsp://{quote(username, safe='')}:{quote(password, safe='')}@{host}:554/"
        f"Streaming/Channels/{channel}"
    )


def main():
    parser = argparse.ArgumentParser(description="Webcam camera calibration")
    parser.add_argument("--camera", default="0", help="local camera index used with --local-camera")
    parser.add_argument("--local-camera", action="store_true", help="use a local webcam instead")
    parser.add_argument("--hikvision-host", default=os.getenv("HIKVISION_HOST", "192.168.1.102"),
                        help="Hikvision camera IP/hostname")
    parser.add_argument("--hikvision-user", default=os.getenv("HIKVISION_USER", "admin"))
    parser.add_argument("--hikvision-password", default=os.getenv("HIKVISION_PASSWORD"),
                        help="recommended: set HIKVISION_PASSWORD instead")
    parser.add_argument("--hikvision-channel", type=int, default=101,
                        help="101=main stream, 102=substream for channel 1")
    parser.add_argument("--rtsp-transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--cols", type=int, default=9, help="number of INNER corners across")
    parser.add_argument("--rows", type=int, default=6, help="number of INNER corners down")
    parser.add_argument("--square-size", type=float, default=0.024,
                        help="printed square side length in metres")
    parser.add_argument("--output", default="camera_calibration.npz")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    pattern_size = (args.cols, args.rows)
    object_template = np.zeros((args.cols * args.rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    object_template *= args.square_size
    object_points, image_points = [], []

    if args.local_camera:
        source = args.camera
    else:
        if not args.hikvision_password:
            args.hikvision_password = getpass("Hikvision password: ")
        if not args.hikvision_password:
            parser.error("Provide a password or set HIKVISION_PASSWORD")
        source = hikvision_rtsp_url(
            args.hikvision_host, args.hikvision_user,
            args.hikvision_password, args.hikvision_channel,
        )

    cap = open_camera(source, args.width, args.height, args.rtsp_transport)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Check RTSP settings and credentials.")

    print("Show the chessboard from different angles. Press S to save a view, C to calibrate.")
    image_size = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        display = frame.copy()
        if found:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            cv2.drawChessboardCorners(display, pattern_size, corners, found)
            cv2.putText(display, "Chessboard found - press S to save this view", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Chessboard not found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(display, f"Saved views: {len(image_points)} | S: save  C: calibrate  Q: quit",
                    (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)
        cv2.imshow("Camera Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s") and found:
            object_points.append(object_template.copy())
            image_points.append(corners)
            image_size = gray.shape[::-1]
            print(f"Saved view {len(image_points)}")
        if key == ord("c"):
            if len(image_points) < 12:
                print("Save at least 12 different views before calibration.")
                continue
            rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                object_points, image_points, image_size, None, None
            )
            total_error = 0.0
            total_points = 0
            for object_pts, image_pts, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
                projected, _ = cv2.projectPoints(object_pts, rvec, tvec, camera_matrix, dist_coeffs)
                total_error += cv2.norm(image_pts, projected, cv2.NORM_L2SQR)
                total_points += len(object_pts)
            reprojection_error = float(np.sqrt(total_error / total_points))
            np.savez(
                args.output,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                image_width=image_size[0],
                image_height=image_size[1],
                rms=rms,
                reprojection_error=reprojection_error,
            )
            print(f"Saved {args.output}")
            print(f"RMS: {rms:.4f}, mean reprojection error: {reprojection_error:.4f} pixels")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
