"""Export ArUco measurement data from SQLite to an Excel-friendly CSV file.

Examples:
    python export_measurements.py
    python export_measurements.py --marker-id 0 --output marker_0.csv
"""

import argparse
import csv
import sqlite3
from pathlib import Path


THAI_HEADERS = {
    "recorded_at": "วันเวลาที่บันทึก",
    "marker_id": "รหัส ArUco",
    "x_m": "ตำแหน่ง X (เมตร)",
    "y_m": "ตำแหน่ง Y (เมตร)",
    "z_m": "ตำแหน่ง Z (เมตร)",
    "image_file": "ไฟล์ภาพประกอบ",
    "center_x_px": "จุดกึ่งกลาง X (พิกเซล)",
    "center_y_px": "จุดกึ่งกลาง Y (พิกเซล)",
    "dx_mm": "การเปลี่ยน X (มม.)",
    "dy_mm": "การเปลี่ยน Y (มม.)",
    "dz_mm": "การเปลี่ยน Z (มม.)",
    "distance_mm": "ระยะการเคลื่อนที่ (มม.)",
    "elapsed_seconds": "เวลาสะสม (วินาที)",
    "speed_mm_s": "ความเร็ว (มม./วินาที)",
    "tilt_deg": "มุมเอียง (องศา)",
}


def parse_target_ids(value: str):
    try:
        target_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("target IDs ต้องเป็นตัวเลขคั่นด้วย comma") from error
    if not target_ids:
        raise argparse.ArgumentTypeError("ระบุ target ID อย่างน้อยหนึ่งค่า")
    return target_ids


def latest_session(records):
    """Return the newest run, identified by the elapsed time resetting to zero."""
    current_session = []
    previous_elapsed = None
    for record in sorted(records, key=lambda item: item["recorded_at"]):
        elapsed = record["elapsed_seconds"]
        if previous_elapsed is not None and elapsed < previous_elapsed - 0.1:
            current_session = []
        current_session.append(record)
        previous_elapsed = elapsed
    return current_session


def write_comparison_csv(path: Path, records, target_ids):
    """Write one row per measurement time and one distance column per target."""
    session = latest_session(records)
    points = {}
    for record in session:
        key = round(record["elapsed_seconds"], 3)
        point = points.setdefault(
            key,
            {
                "elapsed_seconds": record["elapsed_seconds"],
                "recorded_at": record["recorded_at"],
                "distances": {},
            },
        )
        point["distances"][record["marker_id"]] = record["distance_mm"]

    headers = [
        "เวลาสะสม (วินาที)",
        "วันเวลาที่บันทึก",
        *[f"เป้า ID {marker_id} (มม.)" for marker_id in target_ids],
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        valid_points = [
            point for point in points.values()
            if any(point["distances"].get(marker_id) is not None for marker_id in target_ids)
        ]
        for point in sorted(valid_points, key=lambda item: item["elapsed_seconds"]):
            writer.writerow(
                [
                    point["elapsed_seconds"],
                    point["recorded_at"],
                    *[point["distances"].get(marker_id) for marker_id in target_ids],
                ]
            )
    return len(valid_points)


def main():
    parser = argparse.ArgumentParser(description="Export SQLite measurements to CSV")
    parser.add_argument("--database", default="movement_data.db")
    parser.add_argument("--output", default="movement_export.csv")
    parser.add_argument("--marker-id", type=int, help="export one marker ID only")
    parser.add_argument("--comparison-output", default="movement_comparison.csv",
                        help="CSV ตารางเปรียบเทียบรอบล่าสุด (ค่าเริ่มต้น movement_comparison.csv)")
    parser.add_argument("--target-ids", type=parse_target_ids, default=list(range(9)),
                        help="ID เป้าสำหรับตารางเปรียบเทียบ เช่น 0,1,2,3,4,5,6,7,8")
    parser.add_argument("--no-comparison", action="store_true",
                        help="ไม่สร้าง CSV ตารางเปรียบเทียบ")
    args = parser.parse_args()

    database_path = Path(args.database)
    if not database_path.is_file():
        raise FileNotFoundError(f"Database not found: {database_path}")

    connection = sqlite3.connect(database_path)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(measurements)")}
    pixel_columns = (
        "center_x_px, center_y_px,"
        if {"center_x_px", "center_y_px"}.issubset(columns)
        else "NULL AS center_x_px, NULL AS center_y_px,"
    )
    query = f"""
        SELECT recorded_at, marker_id, x_m, y_m, z_m,
               {'image_file,' if 'image_file' in columns else 'NULL AS image_file,'}
               {pixel_columns}
               dx_mm, dy_mm, dz_mm, distance_mm,
               elapsed_seconds, speed_mm_s, tilt_deg
        FROM measurements
    """
    parameters = []
    if args.marker_id is not None:
        query += " WHERE marker_id = ?"
        parameters.append(args.marker_id)
    query += " ORDER BY recorded_at, marker_id"

    cursor = connection.execute(query, parameters)
    raw_headers = [column[0] for column in cursor.description]
    headers = [THAI_HEADERS.get(column, column) for column in raw_headers]
    rows = cursor.fetchall()
    connection.close()

    # utf-8-sig adds a BOM so Excel on Windows reads column names correctly.
    with open(args.output, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"Exported {len(rows)} records to {args.output}")

    if not args.no_comparison:
        all_records_query = """
            SELECT recorded_at, marker_id, elapsed_seconds, distance_mm
            FROM measurements
            ORDER BY recorded_at, marker_id
        """
        comparison_connection = sqlite3.connect(database_path)
        comparison_rows = comparison_connection.execute(all_records_query).fetchall()
        comparison_connection.close()
        comparison_records = [
            {
                "recorded_at": recorded_at,
                "marker_id": marker_id,
                "elapsed_seconds": elapsed_seconds,
                "distance_mm": distance_mm,
            }
            for recorded_at, marker_id, elapsed_seconds, distance_mm in comparison_rows
            if elapsed_seconds is not None
        ]
        comparison_count = write_comparison_csv(
            Path(args.comparison_output), comparison_records, args.target_ids
        )
        print(
            f"Exported {comparison_count} time points to {args.comparison_output} "
            f"for target IDs {args.target_ids}"
        )


if __name__ == "__main__":
    main()
