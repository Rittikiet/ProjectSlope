"""ตรวจจับ ArUco ด้วยกล้องโน้ตบุ๊ก แสดงตำแหน่ง 3 มิติและติดตามจุดศูนย์กลาง

ตัวอย่าง:
    python aruco_tracker.py --marker-length 0.05

marker-length มีหน่วยเป็นเมตร (0.05 = เป้ากว้าง 5 ซม.)
กด Q หรือ ESC เพื่อปิดโปรแกรม
"""

import argparse
from collections import deque

import cv2
import numpy as np


def make_camera_matrix(frame_width: int, frame_height: int, fov_degrees: float) -> np.ndarray:
    """สร้างค่า intrinsic โดยประมาณจาก FOV ของกล้อง.

    สำหรับความแม่นยำสูง ควรเปลี่ยน camera_matrix และ dist_coeffs เป็นค่าจาก
    การ camera calibration ของกล้องจริง (เช่น chessboard calibration)
    """
    focal_length = frame_width / (2.0 * np.tan(np.deg2rad(fov_degrees / 2.0)))
    return np.array(
        [[focal_length, 0.0, frame_width / 2.0],
         [0.0, focal_length, frame_height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def get_detector(dictionary_id: int):
    """รองรับทั้ง OpenCV รุ่นใหม่และรุ่นเก่า."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        return aruco_dict, parameters, cv2.aruco.ArucoDetector(aruco_dict, parameters)
    return aruco_dict, parameters, None


def detect_markers(gray, aruco_dict, parameters, detector):
    if detector is not None:
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)


def estimate_marker_pose(marker_corners, marker_length, camera_matrix, dist_coeffs):
    """คำนวณ pose ของ ArUco หนึ่งแผ่นด้วย solvePnP.

    ใช้วิธีนี้แทน estimatePoseSingleMarkers เพื่อให้ทำงานได้กับ OpenCV รุ่นใหม่
    ที่อาจไม่มีฟังก์ชันเดิมแล้ว.
    """
    half = marker_length / 2.0
    object_points = np.array(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
        dtype=np.float32,
    )
    image_points = marker_corners.reshape(4, 2).astype(np.float32)
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None, None
    return rvec, tvec


def main():
    parser = argparse.ArgumentParser(description="ArUco pose and center tracker")
    parser.add_argument("--camera", type=int, default=0, help="หมายเลขกล้อง (ค่าเริ่มต้น 0)")
    parser.add_argument("--marker-length", type=float, default=0.05,
                        help="ความยาวด้านของ ArUco หน่วยเมตร (ค่าเริ่มต้น 0.05)")
    parser.add_argument("--fov", type=float, default=60.0,
                        help="horizontal FOV โดยประมาณของกล้อง (องศา)")
    parser.add_argument("--dictionary", default="DICT_4X4_50",
                        help="เช่น DICT_4X4_50, DICT_5X5_100")
    args = parser.parse_args()

    if not hasattr(cv2.aruco, args.dictionary):
        raise ValueError(f"ไม่พบ ArUco dictionary: {args.dictionary}")
    dictionary_id = getattr(cv2.aruco, args.dictionary)
    aruco_dict, parameters, detector = get_detector(dictionary_id)

    # CAP_DSHOW ทำให้เปิด webcam บน Windows ได้เร็วขึ้น; ถ้าใช้ระบบอื่น OpenCV จะเลือก backend เอง
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("เปิดกล้องไม่ได้: ลองเปลี่ยนค่า --camera เช่น --camera 1")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    camera_matrix = None
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)  # ค่าประมาณเมื่อยังไม่ได้ calibrate
    center_history = deque(maxlen=30)

    print("เริ่มตรวจจับ ArUco แล้ว — กด Q หรือ ESC เพื่อปิด")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("อ่านภาพจากกล้องไม่สำเร็จ")
            break

        height, width = frame.shape[:2]
        if camera_matrix is None:
            camera_matrix = make_camera_matrix(width, height, args.fov)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect_markers(gray, aruco_dict, parameters, detector)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for marker_corners, marker_id in zip(corners, ids.flatten()):
                rvec, tvec = estimate_marker_pose(
                    marker_corners, args.marker_length, camera_matrix, dist_coeffs
                )
                if rvec is None:
                    continue
                # พิกัด pixel ของจุดกึ่งกลางเป้า
                center = marker_corners[0].mean(axis=0).astype(int)
                center_history.append(tuple(center))
                smooth_center = np.mean(center_history, axis=0).astype(int)

                # tvec เป็นพิกัดของเป้าเทียบกับกล้อง: +X ขวา, +Y ลง, +Z ออกจากกล้อง
                x, y, z = tvec.ravel()
                text = f"ID {marker_id} | X={x:.3f}m Y={y:.3f}m Z={z:.3f}m"
                cv2.putText(frame, text, (10, 32 + 28 * int(marker_id % 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(frame, tuple(center), 6, (0, 0, 255), -1)
                cv2.circle(frame, tuple(smooth_center), 9, (255, 0, 255), 2)
                cv2.putText(frame, f"center: ({smooth_center[0]}, {smooth_center[1]})",
                            (smooth_center[0] + 10, smooth_center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, args.marker_length * 0.5)

            # วาดเส้นทางของ center ที่ track ไว้ (สีเหลือง)
            if len(center_history) > 1:
                cv2.polylines(frame, [np.array(center_history, dtype=np.int32)], False, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "No ArUco marker detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("ArUco Pose & Center Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
