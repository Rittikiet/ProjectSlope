"""ตรวจจับ ArUco ด้วยกล้องโน้ตบุ๊ก แสดงตำแหน่ง 3 มิติและติดตามจุดศูนย์กลาง

ตัวอย่าง:
    python aruco_tracker.py --marker-length 0.03

marker-length มีหน่วยเป็นเมตร (0.03 = เป้ากว้าง 3 ซม.)
กด Q หรือ ESC เพื่อปิดโปรแกรม
"""

import argparse
from collections import defaultdict, deque
import csv
from datetime import datetime
from getpass import getpass
import os
from pathlib import Path
import sqlite3
import time
from urllib.parse import quote

import cv2
import numpy as np


def make_camera_matrix(
    frame_width: int,
    frame_height: int,
    horizontal_fov_degrees: float,
    vertical_fov_degrees: float,
) -> np.ndarray:
    """สร้างค่า intrinsic โดยประมาณจาก horizontal/vertical FOV ของกล้อง.

    สำหรับความแม่นยำสูง ควรเปลี่ยน camera_matrix และ dist_coeffs เป็นค่าจาก
    การ camera calibration ของกล้องจริง (เช่น chessboard calibration)
    """
    focal_length_x = frame_width / (
        2.0 * np.tan(np.deg2rad(horizontal_fov_degrees / 2.0))
    )
    focal_length_y = frame_height / (
        2.0 * np.tan(np.deg2rad(vertical_fov_degrees / 2.0))
    )
    return np.array(
        [[focal_length_x, 0.0, frame_width / 2.0],
         [0.0, focal_length_y, frame_height / 2.0],
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


def marker_tilt_degrees(rvec):
    """Return the angle between the marker plane and the camera image plane."""
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    marker_normal = rotation_matrix[:, 2]
    # abs allows either normal direction; 0 degrees means the marker faces the camera.
    cosine = np.clip(abs(marker_normal[2]), 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def open_database(database_path: str):
    """Create the SQLite database schema when it does not exist yet."""
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS markers (
            marker_id INTEGER PRIMARY KEY,
            first_seen_at TEXT NOT NULL,
            baseline_x_m REAL,
            baseline_y_m REAL,
            baseline_z_m REAL,
            baseline_set_at TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at TEXT NOT NULL,
            session_id TEXT,
            marker_id INTEGER NOT NULL,
            x_m REAL NOT NULL,
            y_m REAL NOT NULL,
            z_m REAL NOT NULL,
            image_file TEXT,
            center_x_px INTEGER,
            center_y_px INTEGER,
            dx_mm REAL,
            dy_mm REAL,
            dz_mm REAL,
            distance_mm REAL,
            elapsed_seconds REAL,
            speed_mm_s REAL,
            tilt_deg REAL NOT NULL,
            FOREIGN KEY(marker_id) REFERENCES markers(marker_id)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_measurements_marker_time "
        "ON measurements(marker_id, recorded_at)"
    )
    # Migrate databases that were created by an older version of this program.
    existing_columns = {row[1] for row in connection.execute("PRAGMA table_info(measurements)")}
    if "elapsed_seconds" not in existing_columns:
        connection.execute("ALTER TABLE measurements ADD COLUMN elapsed_seconds REAL")
    if "speed_mm_s" not in existing_columns:
        connection.execute("ALTER TABLE measurements ADD COLUMN speed_mm_s REAL")
    if "center_x_px" not in existing_columns:
        connection.execute("ALTER TABLE measurements ADD COLUMN center_x_px INTEGER")
    if "center_y_px" not in existing_columns:
        connection.execute("ALTER TABLE measurements ADD COLUMN center_y_px INTEGER")
    if "image_file" not in existing_columns:
        connection.execute("ALTER TABLE measurements ADD COLUMN image_file TEXT")
    if "session_id" not in existing_columns:
        connection.execute("ALTER TABLE measurements ADD COLUMN session_id TEXT")
    connection.commit()
    return connection


def current_time_text():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def open_camera(source: str, width: int, height: int, rtsp_transport: str):
    """Open either a local camera index or an RTSP URL."""
    if source.isdecimal():
        camera = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera = cv2.VideoCapture(int(source))
    else:
        # Request TCP before creating the capture.  TCP is generally more reliable
        # than UDP on a LAN where packet loss causes broken video frames.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{rtsp_transport}"
        camera = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not camera.isOpened():
            camera = cv2.VideoCapture(source)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Keep latency low when an RTSP camera delivers frames faster than processing.
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def hikvision_rtsp_url(host: str, username: str, password: str, channel: int) -> str:
    """Build a Hikvision RTSP URL, escaping credentials safely."""
    return (
        f"rtsp://{quote(username, safe='')}:{quote(password, safe='')}@{host}:554/"
        f"Streaming/Channels/{channel}"
    )


def open_video_writer(video_directory: Path, width: int, height: int, fps: float):
    """Create a timestamped MP4 file for one measurement session."""
    video_directory.mkdir(parents=True, exist_ok=True)
    filename = f"aruco_test_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    video_path = video_directory / filename
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video file: {video_path}")
    return writer, video_path


def draw_movement_chart(records, output_path: Path):
    """Create a PNG chart without requiring an extra plotting package."""
    series = defaultdict(list)
    for elapsed_seconds, marker_id, distance_mm in records:
        if distance_mm is not None:
            series[marker_id].append((elapsed_seconds, distance_mm))
    if not series:
        return False

    width, height = 1600, 900
    left, top, right, bottom = 135, 120, 70, 130
    chart_width, chart_height = width - left - right, height - top - bottom
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    all_times = [elapsed for points in series.values() for elapsed, _ in points]
    all_distances = [distance for points in series.values() for _, distance in points]
    max_time = max(max(all_times), 1.0)
    min_distance = min(0.0, min(all_distances))
    max_distance = max(0.0, max(all_distances))
    if max_distance == min_distance:
        max_distance += 1.0
    padding = (max_distance - min_distance) * 0.08
    min_distance -= padding
    max_distance += padding

    def point(elapsed, distance):
        x = left + int((elapsed / max_time) * chart_width)
        y = top + int(((max_distance - distance) / (max_distance - min_distance)) * chart_height)
        return x, y

    for tick in range(6):
        fraction = tick / 5
        x = left + int(fraction * chart_width)
        y = top + int(fraction * chart_height)
        cv2.line(image, (x, top), (x, top + chart_height), (225, 225, 225), 1)
        cv2.line(image, (left, y), (left + chart_width, y), (225, 225, 225), 1)
        time_value = fraction * max_time
        distance_value = max_distance - fraction * (max_distance - min_distance)
        cv2.putText(image, f"{time_value:.1f}", (x - 18, top + chart_height + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(image, f"{distance_value:.1f}", (20, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)

    cv2.rectangle(image, (left, top), (left + chart_width, top + chart_height), (60, 60, 60), 2)
    cv2.putText(image, "Movement distance vs elapsed time", (left, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.05, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(image, "Elapsed time (seconds)", (width // 2 - 130, height - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(image, "Distance (mm)", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1, cv2.LINE_AA)

    colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
              (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
              (188, 189, 34)]
    for index, marker_id in enumerate(sorted(series)):
        color = colors[index % len(colors)]
        points = [point(elapsed, distance) for elapsed, distance in series[marker_id]]
        if len(points) > 1:
            cv2.polylines(image, [np.array(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        for marker_point in points:
            cv2.circle(image, marker_point, 4, color, -1, cv2.LINE_AA)
        legend_x = left + (index % 5) * 185
        legend_y = 82 + (index // 5) * 25
        cv2.line(image, (legend_x, legend_y), (legend_x + 26, legend_y), color, 3, cv2.LINE_AA)
        cv2.putText(image, f"Target ID {marker_id}", (legend_x + 34, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    return cv2.imwrite(str(output_path), image)


def write_session_excel(output_path: Path, comparison_rows, marker_ids, raw_headers, raw_rows):
    """Write an editable Excel report with a movement chart and the raw data."""
    try:
        from openpyxl import Workbook
        from openpyxl.chart import LineChart, Reference
        from openpyxl.chart.series import SeriesLabel
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as error:
        print("ไม่ได้สร้าง Excel report: ติดตั้ง openpyxl ด้วย pip install -r requirements.txt")
        return False

    workbook = Workbook()
    summary_sheet = workbook.active
    summary_sheet.title = "กราฟและข้อมูล"
    comparison_headers = [
        "เวลาสะสม (วินาที)", "วันเวลาที่บันทึก",
        *[f"เป้า ID {marker_id} (มม.)" for marker_id in marker_ids],
    ]
    summary_sheet.append(comparison_headers)
    for elapsed_seconds, recorded_at, distances in comparison_rows:
        summary_sheet.append([
            elapsed_seconds, recorded_at,
            *[distances.get(marker_id) for marker_id in marker_ids],
        ])

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    for cell in summary_sheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")
    summary_sheet.freeze_panes = "C2"
    summary_sheet.auto_filter.ref = summary_sheet.dimensions
    summary_sheet.column_dimensions["A"].width = 20
    summary_sheet.column_dimensions["B"].width = 32
    for column in range(3, len(comparison_headers) + 1):
        summary_sheet.column_dimensions[chr(64 + column)].width = 18
    for row in summary_sheet.iter_rows(min_row=2, min_col=1, max_col=1):
        row[0].number_format = "0.000"
    for row in summary_sheet.iter_rows(min_row=2, min_col=3, max_col=len(comparison_headers)):
        for cell in row:
            cell.number_format = "0.000"

    if marker_ids and len(comparison_rows) > 0:
        chart = LineChart()
        chart.title = "ระยะการเคลื่อนที่เทียบกับเวลา"
        chart.style = 13
        chart.y_axis.title = "ระยะการเคลื่อนที่ (mm)"
        chart.x_axis.title = "เวลาสะสม (s)"
        chart.height = 14
        chart.width = 28
        data = Reference(summary_sheet, min_col=3, max_col=2 + len(marker_ids), min_row=1,
                         max_row=len(comparison_rows) + 1)
        categories = Reference(summary_sheet, min_col=1, min_row=2,
                               max_row=len(comparison_rows) + 1)
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(categories)
        summary_sheet.add_chart(chart, "A" + str(len(comparison_rows) + 4))

    raw_sheet = workbook.create_sheet("ข้อมูลดิบ")
    raw_sheet.append(raw_headers)
    for raw_row in raw_rows:
        raw_sheet.append(list(raw_row))
    for cell in raw_sheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    raw_sheet.freeze_panes = "A2"
    raw_sheet.auto_filter.ref = raw_sheet.dimensions
    for index, header in enumerate(raw_headers, start=1):
        raw_sheet.column_dimensions[get_column_letter(index)].width = min(max(len(header) + 3, 15), 28)

    speed_by_time = {}
    for raw_row in raw_rows:
        elapsed_seconds, speed = raw_row[12], raw_row[13]
        if elapsed_seconds is None or speed is None:
            continue
        key = round(float(elapsed_seconds), 3)
        point = speed_by_time.setdefault(key, (float(key), []))
        point[1].append(float(speed))
    speed_summary = [
        (elapsed, sum(speeds) / len(speeds), max(speeds))
        for elapsed, speeds in sorted(speed_by_time.values())
    ]
    if speed_summary:
        summary_start_column = len(raw_headers) + 2
        summary_end_column = summary_start_column + 2
        summary_headers = [
            "เวลาสะสม (วินาที)", "ความเร็วเฉลี่ย (มม./วินาที)",
            "ความเร็วสูงสุด (มม./วินาที)",
        ]
        for index, header in enumerate(summary_headers, start=summary_start_column):
            cell = raw_sheet.cell(1, index, header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
            raw_sheet.column_dimensions[get_column_letter(index)].width = 22
        for row_index, values in enumerate(speed_summary, start=2):
            for column_index, value in enumerate(values, start=summary_start_column):
                raw_sheet.cell(row_index, column_index, value).number_format = "0.000"
        speed_chart = LineChart()
        speed_chart.title = "ความเร็วเฉลี่ยและสูงสุดเทียบกับเวลา"
        speed_chart.y_axis.title = "ความเร็ว (มม./วินาที)"
        speed_chart.x_axis.title = "เวลาสะสม (วินาที)"
        speed_chart.height = 14
        speed_chart.width = 28
        speed_chart.add_data(
            Reference(raw_sheet, min_col=summary_start_column + 1, max_col=summary_end_column,
                      min_row=1, max_row=len(speed_summary) + 1),
            titles_from_data=True,
        )
        speed_chart.set_categories(
            Reference(raw_sheet, min_col=summary_start_column, min_row=2,
                      max_row=len(speed_summary) + 1),
        )
        speed_chart.series[0].tx = SeriesLabel(v=summary_headers[1])
        speed_chart.series[1].tx = SeriesLabel(v=summary_headers[2])
        speed_chart.series[0].graphicalProperties.line.solidFill = "1F4E78"
        speed_chart.series[1].graphicalProperties.line.solidFill = "C00000"
        raw_sheet.add_chart(speed_chart, f"{get_column_letter(summary_end_column + 2)}2")

    workbook.save(output_path)
    return True


def save_session_report(database, session_id: str, report_directory: Path):
    """Save a presentation-ready comparison CSV and graph for one test session."""
    rows = database.execute(
        "SELECT recorded_at, marker_id, elapsed_seconds, distance_mm "
        "FROM measurements WHERE session_id = ? ORDER BY elapsed_seconds, marker_id",
        (session_id,),
    ).fetchall()
    if not rows:
        return None

    output_directory = report_directory / f"session_{session_id}"
    output_directory.mkdir(parents=True, exist_ok=True)

    raw_headers = [
        "วันเวลาที่บันทึก", "รหัส ArUco", "ตำแหน่ง X (m)", "ตำแหน่ง Y (m)",
        "ตำแหน่ง Z (m)", "ไฟล์ภาพประกอบ", "จุดกึ่งกลาง X (pixel)",
        "จุดกึ่งกลาง Y (pixel)", "dX (mm)", "dY (m)",
        "dZ (มม.)", "ระยะการเคลื่อนที่ (mm)", "เวลาสะสม (s)",
        "ความเร็ว (mm/s))", "มุมเอียง (degre)",
    ]
    raw_rows = database.execute(
        """
        SELECT recorded_at, marker_id, x_m, y_m, z_m, image_file,
               center_x_px, center_y_px, dx_mm, dy_mm, dz_mm, distance_mm,
               elapsed_seconds, speed_mm_s, tilt_deg
        FROM measurements
        WHERE session_id = ?
        ORDER BY elapsed_seconds, marker_id
        """,
        (session_id,),
    ).fetchall()
    raw_csv_path = output_directory / "raw_data.csv"
    with raw_csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(raw_headers)
        for row in raw_rows:
            row = list(row)
            if row[5]:
                row[5] = f"snapshots/{row[5]}"
            writer.writerow(row)

    marker_ids = sorted({marker_id for _, marker_id, _, _ in rows})
    points = {}
    for recorded_at, marker_id, elapsed_seconds, distance_mm in rows:
        key = round(elapsed_seconds, 3)
        point = points.setdefault(key, (elapsed_seconds, recorded_at, {}))
        point[2][marker_id] = distance_mm

    csv_path = output_directory / "movement_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "เวลาสะสม (วินาที)", "วันเวลาที่บันทึก",
            *[f"เป้า ID {marker_id} (มม.)" for marker_id in marker_ids],
        ])
        comparison_rows = sorted(points.values())
        for elapsed_seconds, recorded_at, distances in comparison_rows:
            writer.writerow([
                elapsed_seconds, recorded_at,
                *[distances.get(marker_id) for marker_id in marker_ids],
            ])

    excel_path = output_directory / "movement_report.xlsx"
    excel_created = write_session_excel(
        excel_path, comparison_rows, marker_ids, raw_headers, raw_rows,
    )
    print(f"Saved session comparison CSV: {csv_path}")
    print(f"Saved session raw data CSV: {raw_csv_path}")
    if excel_created:
        print(f"Saved editable Excel report: {excel_path}")
    return output_directory


def main():
    parser = argparse.ArgumentParser(description="ArUco pose and center tracker")
    parser.add_argument("--camera", default="0", help="หมายเลขเว็บแคมเมื่อใช้ --local-camera")
    parser.add_argument("--local-camera", action="store_true", help="ใช้เว็บแคมแทนกล้อง IP")
    parser.add_argument("--hikvision-host", default=os.getenv("HIKVISION_HOST", "192.168.1.102"),
                        help="IP/hostname ของกล้อง Hikvision")
    parser.add_argument("--hikvision-user", default=os.getenv("HIKVISION_USER", "admin"),
                        help="ชื่อผู้ใช้ Hikvision")
    parser.add_argument("--hikvision-password", default=os.getenv("HIKVISION_PASSWORD"),
                        help="รหัสผ่าน Hikvision (แนะนำให้ตั้ง HIKVISION_PASSWORD แทน)")
    parser.add_argument("--hikvision-channel", type=int, default=101,
                        help="101=สตรีมหลัก, 102=สตรีมย่อย ของ channel 1")
    parser.add_argument("--rtsp-transport", choices=("tcp", "udp"), default="tcp",
                        help="รูปแบบส่งข้อมูล RTSP (ค่าเริ่มต้น tcp)")
    parser.add_argument("--marker-length", type=float, default=0.032,
                        help="ความยาวด้านของ ArUco หน่วยเมตร (ค่าเริ่มต้น 0.032 = 3.2 ซม.)")
    parser.add_argument("--horizontal-fov", type=float, default=103.0,
                        help="horizontal FOV ของกล้อง (องศา; ค่าเริ่มต้น 103)")
    parser.add_argument("--vertical-fov", type=float, default=56.0,
                        help="vertical FOV ของกล้อง (องศา; ค่าเริ่มต้น 56)")
    parser.add_argument("--dictionary", default="DICT_4X4_50",
                        help="เช่น DICT_4X4_50, DICT_5X5_100")
    parser.add_argument("--max-tilt", type=float, default=50.0,
                        help="มุมเอียงสูงสุด (องศา) ที่อนุญาตให้ตั้งจุดอ้างอิง (ค่าเริ่มต้น 50)")
    parser.add_argument("--pose-filter-window", type=int, default=7,
                        help="จำนวนเฟรมสำหรับ median filter ของตำแหน่ง/Tilt (ค่าเริ่มต้น 7)")
    parser.add_argument("--calibration", default="camera_calibration.npz",
                        help="ไฟล์ผลลัพธ์จาก camera_calibration.py")
    parser.add_argument("--width", type=int, default=1920, help="ความกว้างภาพที่ต้องการจากกล้อง")
    parser.add_argument("--height", type=int, default=1080, help="ความสูงภาพที่ต้องการจากกล้อง")
    parser.add_argument("--database", default="movement_data.db", help="ไฟล์ SQLite สำหรับเก็บผล")
    parser.add_argument("--log-interval", type=float, default=1.0,
                        help="บันทึกแต่ละเป้าทุกกี่วินาที (ค่าเริ่มต้น 1.0)")
    parser.add_argument("--image-directory", default="snapshots",
                        help="โฟลเดอร์เก็บภาพที่จับคู่กับข้อมูล CSV (ค่าเริ่มต้น snapshots)")
    parser.add_argument("--no-images", action="store_true", help="ไม่บันทึกภาพประกอบ CSV")
    parser.add_argument("--video-directory", default="recordings",
                        help="โฟลเดอร์เก็บวิดีโอช่วงทดสอบ (ค่าเริ่มต้น recordings)")
    parser.add_argument("--video-fps", type=float, default=20.0,
                        help="FPS ของไฟล์วิดีโอที่บันทึก (ค่าเริ่มต้น 20)")
    parser.add_argument("--no-video", action="store_true", help="ไม่บันทึกวิดีโอ")
    parser.add_argument("--session-directory", default="sessions",
                        help="โฟลเดอร์สรุป CSV และกราฟของแต่ละรอบทดสอบ")
    args = parser.parse_args()

    if not hasattr(cv2.aruco, args.dictionary):
        raise ValueError(f"ไม่พบ ArUco dictionary: {args.dictionary}")
    if args.pose_filter_window < 1:
        parser.error("--pose-filter-window ต้องมีค่าอย่างน้อย 1")
    dictionary_id = getattr(cv2.aruco, args.dictionary)
    aruco_dict, parameters, detector = get_detector(dictionary_id)

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

    cap = open_camera(source, args.width, args.height, args.rtsp_transport)
    if not cap.isOpened():
        raise RuntimeError(
            "เปิดกล้องไม่ได้: ตรวจสอบ IP/RTSP, username/password และเปิด RTSP ในกล้อง"
        )

    camera_matrix = None
    calibration_matrix = None
    calibration_size = None
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)  # fallback when no calibration file exists
    calibration_file = Path(args.calibration)
    if calibration_file.is_file():
        calibration = np.load(calibration_file)
        calibration_matrix = calibration["camera_matrix"].astype(np.float64)
        dist_coeffs = calibration["dist_coeffs"].astype(np.float64)
        calibration_size = (int(calibration["image_width"]), int(calibration["image_height"]))
        print(f"Using camera calibration: {calibration_file}")
    else:
        print("No calibration file found: using approximate FOV; distance is not measurement-grade.")
    center_histories = defaultdict(lambda: deque(maxlen=30))
    position_histories = defaultdict(lambda: deque(maxlen=args.pose_filter_window))
    tilt_histories = defaultdict(lambda: deque(maxlen=args.pose_filter_window))
    database = open_database(args.database)
    reference_poses = {
        marker_id: np.array([x, y, z], dtype=np.float64)
        for marker_id, x, y, z in database.execute(
            "SELECT marker_id, baseline_x_m, baseline_y_m, baseline_z_m "
            "FROM markers WHERE baseline_x_m IS NOT NULL"
        )
    }
    last_logged_at = defaultdict(lambda: -float("inf"))
    last_logged_pose = {}
    tracking_active = False
    logging_started_at = None
    video_writer = None
    video_path = None
    active_session_id = None
    active_snapshot_directory = None
    active_video_directory = None
    failed_reads = 0
    print(f"Ready to save measurements to SQLite: {args.database} (every {args.log_interval:g} s)")
    if reference_poses:
        print(f"Loaded baselines for marker IDs: {sorted(reference_poses)}")

    print("เริ่มตรวจจับ ArUco แล้ว — กด Q หรือ ESC เพื่อปิด")
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
                print("เชื่อมต่อสตรีมกล้องใหม่ไม่สำเร็จ")
                break
            continue
        failed_reads = 0

        height, width = frame.shape[:2]
        if camera_matrix is None:
            if calibration_matrix is not None:
                camera_matrix = calibration_matrix.copy()
                scale_x = width / calibration_size[0]
                scale_y = height / calibration_size[1]
                camera_matrix[0, 0] *= scale_x
                camera_matrix[0, 2] *= scale_x
                camera_matrix[1, 1] *= scale_y
                camera_matrix[1, 2] *= scale_y
            else:
                camera_matrix = make_camera_matrix(
                    width, height, args.horizontal_fov, args.vertical_fov
                )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect_markers(gray, aruco_dict, parameters, detector)
        visible_poses = {}
        monotonic_now = time.monotonic()
        pending_image_filename = None

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
                center_histories[int(marker_id)].append(tuple(center))
                smooth_center = np.mean(center_histories[int(marker_id)], axis=0).astype(int)

                # tvec เป็นพิกัดของเป้าเทียบกับกล้อง: +X ขวา, +Y ลง, +Z ออกจากกล้อง.
                # Median filter cuts occasional RTSP/pose-estimation spikes without
                # being pulled strongly by a single outlier frame.
                raw_tvec = tvec.ravel().copy()
                raw_tilt = marker_tilt_degrees(rvec)
                position_histories[int(marker_id)].append(raw_tvec)
                tilt_histories[int(marker_id)].append(raw_tilt)
                filtered_tvec = np.median(
                    np.asarray(position_histories[int(marker_id)]), axis=0
                )
                tilt = float(np.median(tilt_histories[int(marker_id)]))
                x, y, z = filtered_tvec
                visible_poses[int(marker_id)] = (filtered_tvec.copy(), tilt)
                aligned = tilt <= args.max_tilt
                alignment_text = f"Tilt={tilt:.1f} deg  " + ("ALIGNED" if aligned else "TILT TARGET")
                alignment_color = (0, 255, 0) if aligned else (0, 0, 255)
                text = f"ID {marker_id} | X={x:.3f}m Y={y:.3f}m Z={z:.3f}m"
                cv2.putText(frame, text, (10, 32 + 28 * int(marker_id % 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, alignment_text, (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, alignment_color, 2, cv2.LINE_AA)

                baseline = reference_poses.get(int(marker_id))
                dx = dy = dz = distance = None
                if baseline is not None:
                    dx, dy, dz = filtered_tvec - baseline
                    distance = float(np.linalg.norm([dx, dy, dz]))
                    delta_text = (
                        f"Movement: dX={dx * 100:.1f}cm dY={dy * 100:.1f}cm "
                        f"dZ={dz * 100:.1f}cm  D={distance * 100:.1f}cm"
                    )
                    cv2.putText(frame, delta_text, (10, 112),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)

                if (tracking_active and args.log_interval > 0
                        and monotonic_now - last_logged_at[int(marker_id)] >= args.log_interval):
                    now_text = current_time_text()
                    elapsed_seconds = monotonic_now - logging_started_at
                    image_file = None
                    if not args.no_images:
                        # All markers logged from this frame share one overall snapshot.
                        if pending_image_filename is None:
                            pending_image_filename = f"snapshot_t{elapsed_seconds:010.3f}s.png"
                        image_file = pending_image_filename
                    previous = last_logged_pose.get(int(marker_id))
                    speed_mm_s = None
                    if previous is not None:
                        previous_tvec, previous_elapsed = previous
                        elapsed_difference = elapsed_seconds - previous_elapsed
                        if elapsed_difference > 0:
                            movement_mm = np.linalg.norm(filtered_tvec - previous_tvec) * 1000
                            speed_mm_s = float(movement_mm / elapsed_difference)
                    database.execute(
                        "INSERT INTO markers(marker_id, first_seen_at) VALUES (?, ?) "
                        "ON CONFLICT(marker_id) DO NOTHING",
                        (int(marker_id), now_text),
                    )
                    database.execute(
                        """
                        INSERT INTO measurements(
                            recorded_at, session_id, marker_id, x_m, y_m, z_m,
                            image_file,
                            center_x_px, center_y_px,
                            dx_mm, dy_mm, dz_mm, distance_mm,
                            elapsed_seconds, speed_mm_s, tilt_deg
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            now_text, active_session_id, int(marker_id), float(x), float(y), float(z),
                            image_file,
                            int(center[0]), int(center[1]),
                            None if dx is None else float(dx * 1000),
                            None if dy is None else float(dy * 1000),
                            None if dz is None else float(dz * 1000),
                            None if distance is None else float(distance * 1000),
                            float(elapsed_seconds), speed_mm_s,
                            float(tilt),
                        ),
                    )
                    database.commit()
                    last_logged_at[int(marker_id)] = monotonic_now
                    last_logged_pose[int(marker_id)] = (filtered_tvec.copy(), elapsed_seconds)
                cv2.circle(frame, tuple(center), 6, (0, 0, 255), -1)
                cv2.circle(frame, tuple(smooth_center), 9, (255, 0, 255), 2)
                cv2.putText(frame, f"center: ({smooth_center[0]}, {smooth_center[1]})",
                            (smooth_center[0] + 10, smooth_center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, args.marker_length * 0.5)

            # วาดเส้นทางของ center ที่ track ไว้ (สีเหลือง) แยกตาม marker ID
            for history in center_histories.values():
                if len(history) > 1:
                    cv2.polylines(frame, [np.array(history, dtype=np.int32)], False, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "No ArUco marker detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.drawMarker(frame, (width // 2, height // 2), (255, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=22, thickness=1)
        status = "RECORDING" if tracking_active else "READY - press SPACE to set zero and start"
        status_color = (0, 255, 0) if tracking_active else (0, 255, 255)
        cv2.putText(frame, status, (10, height - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "Keep marker square to camera | SPACE: start/reset | R: stop & clear | Q: quit",
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if pending_image_filename is not None:
            active_snapshot_directory.mkdir(parents=True, exist_ok=True)
            image_path = active_snapshot_directory / pending_image_filename
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"Cannot save image: {image_path}")
            print(f"Saved snapshot: {image_path}")
        if tracking_active and video_writer is not None:
            video_writer.write(frame)
        cv2.imshow("ArUco Pose & Center Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            tracking_active = False
            if active_session_id is not None:
                save_session_report(database, active_session_id, Path(args.session_directory))
                active_session_id = None
            if video_writer is not None:
                video_writer.release()
                print(f"Saved video: {video_path}")
                video_writer = None
                video_path = None
            logging_started_at = None
            last_logged_at.clear()
            last_logged_pose.clear()
            center_histories.clear()
            position_histories.clear()
            tilt_histories.clear()
            reference_poses.clear()
            database.execute(
                "UPDATE markers SET baseline_x_m = NULL, baseline_y_m = NULL, "
                "baseline_z_m = NULL, baseline_set_at = NULL"
            )
            database.commit()
            print("Stopped tracking and cleared all movement baselines")
        elif key == ord(" "):
            if not visible_poses:
                print("Cannot set zero: no marker detected")
            else:
                eligible_poses = {
                    marker_id: pose
                    for marker_id, pose in visible_poses.items()
                    if pose[1] <= args.max_tilt
                }
                if not eligible_poses:
                    print(
                        "Cannot set zero: every detected target exceeds the "
                        f"tilt limit of {args.max_tilt:.1f} degrees"
                    )
                    continue
                if video_writer is not None:
                    video_writer.release()
                    print(f"Saved video: {video_path}")
                if active_session_id is not None:
                    save_session_report(database, active_session_id, Path(args.session_directory))
                reference_poses.clear()
                database.execute(
                    "UPDATE markers SET baseline_x_m = NULL, baseline_y_m = NULL, "
                    "baseline_z_m = NULL, baseline_set_at = NULL"
                )
                # A new SPACE press always begins a fresh timed measurement session.
                tracking_active = True
                logging_started_at = time.monotonic()
                active_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                session_path = Path(args.session_directory) / f"session_{active_session_id}"
                active_snapshot_directory = session_path / Path(args.image_directory).name
                active_video_directory = session_path / Path(args.video_directory).name
                last_logged_at.clear()
                last_logged_pose.clear()
                center_histories.clear()
                if not args.no_video:
                    video_writer, video_path = open_video_writer(
                        active_video_directory, width, height, args.video_fps
                    )
                    print(f"Recording video: {video_path}")
                now_text = current_time_text()
                for marker_id, (reference_tvec, _) in eligible_poses.items():
                    reference_poses[marker_id] = reference_tvec
                    database.execute(
                        "INSERT INTO markers(marker_id, first_seen_at, baseline_x_m, baseline_y_m, baseline_z_m, baseline_set_at) "
                        "VALUES (?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(marker_id) DO UPDATE SET "
                        "baseline_x_m = excluded.baseline_x_m, baseline_y_m = excluded.baseline_y_m, "
                        "baseline_z_m = excluded.baseline_z_m, baseline_set_at = excluded.baseline_set_at",
                        (marker_id, now_text, *reference_tvec.tolist(), now_text),
                    )
                database.commit()
                skipped = sorted(set(visible_poses) - set(eligible_poses))
                print(
                    "Set zero for marker IDs "
                    f"{sorted(eligible_poses)}; elapsed time reset to 0 and recording started"
                )
                if skipped:
                    print(f"Skipped targets above tilt limit: {skipped}")

    if active_session_id is not None:
        save_session_report(database, active_session_id, Path(args.session_directory))
    if video_writer is not None:
        video_writer.release()
        print(f"Saved video: {video_path}")
    cap.release()
    cv2.destroyAllWindows()
    database.close()


if __name__ == "__main__":
    main()
