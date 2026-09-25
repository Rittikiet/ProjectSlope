"""Calibrate the camera from chessboard images captured in a folder.

Example for a 9 x 7-square board with 25 mm squares:
    python calibrate_from_images.py
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def calibration_error(object_points, image_points, camera_matrix, dist_coeffs, rvecs, tvecs):
    total_error = 0.0
    total_points = 0
    for object_pts, image_pts, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(object_pts, rvec, tvec, camera_matrix, dist_coeffs)
        # OpenCV can represent the same 2-D points as either one Nx2 channel or
        # two channels. Reshape both arrays before calculating squared error.
        difference = (
            image_pts.reshape(-1, 2).astype(np.float64)
            - projected.reshape(-1, 2).astype(np.float64)
        )
        total_error += float(np.sum(difference ** 2))
        total_points += len(object_pts)
    return float(np.sqrt(total_error / total_points))


def main():
    parser = argparse.ArgumentParser(description="สร้าง camera calibration จากภาพตารางหมากรุก")
    parser.add_argument("--image-directory", default="calibration_images",
                        help="โฟลเดอร์ภาพจาก capture_calibration_images.py")
    parser.add_argument("--cols", type=int, default=8, help="จำนวนจุดตัดด้านในแนวนอน")
    parser.add_argument("--rows", type=int, default=6, help="จำนวนจุดตัดด้านในแนวตั้ง")
    parser.add_argument("--square-size", type=float, default=0.025,
                        help="ความยาวด้านช่องตาราง หน่วยเมตร")
    parser.add_argument("--output", default="camera_calibration.npz")
    args = parser.parse_args()

    image_directory = Path(args.image_directory)
    image_paths = sorted(
        path for extension in ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        for path in image_directory.glob(extension)
    )
    if not image_paths:
        raise FileNotFoundError(f"ไม่พบภาพในโฟลเดอร์: {image_directory}")

    pattern_size = (args.cols, args.rows)
    object_template = np.zeros((args.cols * args.rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    object_template *= args.square_size
    object_points, image_points = [], []
    image_size = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"ข้ามไฟล์ที่อ่านไม่ได้: {image_path.name}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]
        if image_size is None:
            image_size = current_size
        if current_size != image_size:
            print(f"ข้ามภาพขนาดไม่ตรงกัน: {image_path.name}")
            continue
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            print(f"ไม่พบตาราง: {image_path.name}")
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        object_points.append(object_template.copy())
        image_points.append(corners)
        print(f"ใช้ภาพ {len(image_points)}: {image_path.name}")

    if len(image_points) < 12:
        raise RuntimeError(
            f"พบภาพตารางเพียง {len(image_points)} ภาพ ต้องมีอย่างน้อย 12 ภาพ"
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None,
    )
    reprojection_error = calibration_error(
        object_points, image_points, camera_matrix, dist_coeffs, rvecs, tvecs,
    )
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
    print(f"Used {len(image_points)} images | RMS: {rms:.4f} | "
          f"mean reprojection error: {reprojection_error:.4f} pixels")


if __name__ == "__main__":
    main()
