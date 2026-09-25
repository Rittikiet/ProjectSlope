"""Capture chessboard photos for offline camera calibration.

The program deliberately does not search for chessboard corners while streaming.
Press S after moving the chessboard to save a lossless PNG image. Press Q or ESC
when at least 15 to 20 images have been saved.
"""

import argparse
from datetime import datetime
from getpass import getpass
import os
from pathlib import Path
import time

import cv2

from camera_calibration import hikvision_rtsp_url, open_camera


def main():
    parser = argparse.ArgumentParser(description="บันทึกภาพตารางหมากรุกเพื่อทำ calibration ภายหลัง")
    parser.add_argument("--camera", default="0", help="หมายเลขเว็บแคมเมื่อใช้ --local-camera")
    parser.add_argument("--local-camera", action="store_true", help="ใช้เว็บแคมแทนกล้อง IP")
    parser.add_argument("--hikvision-host", default=os.getenv("HIKVISION_HOST", "192.168.1.102"))
    parser.add_argument("--hikvision-user", default=os.getenv("HIKVISION_USER", "admin"))
    parser.add_argument("--hikvision-password", default=os.getenv("HIKVISION_PASSWORD"))
    parser.add_argument("--hikvision-channel", type=int, default=101,
                        help="101=สตรีมหลัก, 102=สตรีมย่อย ของ channel 1")
    parser.add_argument("--rtsp-transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--image-directory", default="calibration_images",
                        help="โฟลเดอร์เก็บภาพ calibration")
    args = parser.parse_args()

    if args.local_camera:
        source = args.camera
    else:
        if not args.hikvision_password:
            args.hikvision_password = getpass("Hikvision password: ")
        if not args.hikvision_password:
            parser.error("ระบุรหัสผ่าน หรือกำหนด HIKVISION_PASSWORD")
        source = hikvision_rtsp_url(
            args.hikvision_host, args.hikvision_user,
            args.hikvision_password, args.hikvision_channel,
        )

    output_directory = Path(args.image_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    cap = open_camera(source, args.width, args.height, args.rtsp_transport)
    if not cap.isOpened():
        raise RuntimeError("เปิดกล้องไม่ได้: ตรวจสอบ IP/RTSP และ username/password")

    saved_count = len(list(output_directory.glob("calibration_*.png")))
    failed_reads = 0
    print("เลื่อนตารางไปหลายตำแหน่งและหลายมุม แล้วกด S เพื่อบันทึกภาพ | Q เพื่อจบ")
    while True:
        ok, frame = cap.read()
        if not ok:
            failed_reads += 1
            if failed_reads < 10:
                time.sleep(0.05)
                continue
            print("สตรีมสะดุด กำลังเชื่อมต่อกล้องใหม่...")
            cap.release()
            time.sleep(1)
            cap = open_camera(source, args.width, args.height, args.rtsp_transport)
            failed_reads = 0
            if not cap.isOpened():
                raise RuntimeError("เชื่อมต่อสตรีมกล้องใหม่ไม่สำเร็จ")
            continue
        failed_reads = 0

        display = frame.copy()
        cv2.putText(display, f"Saved photos: {saved_count} | S: save  Q: quit", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display, "Move the chessboard before each saved photo", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Capture calibration images", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_path = output_directory / f"calibration_{timestamp}.png"
            if cv2.imwrite(str(image_path), frame):
                saved_count += 1
                print(f"Saved image {saved_count}: {image_path.name}")
            else:
                print("บันทึกภาพไม่สำเร็จ")

    cap.release()
    cv2.destroyAllWindows()
    print(f"บันทึกภาพทั้งหมด {saved_count} ภาพใน: {output_directory}")


if __name__ == "__main__":
    main()
