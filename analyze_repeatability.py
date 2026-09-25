"""Compare repeated ArUco test sessions at one test angle.

Example (three repeated tests at 45 degrees):
    python analyze_repeatability.py --angle 45 --marker-id 1 --sessions \
        sessions/session_20260909_101500 sessions/session_20260909_102000 \
        sessions/session_20260909_103000

To compare OpenCV against a reference measurement, create a CSV with columns
``time_s,distance_mm`` and add ``--reference visio_45deg.csv``.
"""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


TIME_HEADER = "เวลาสะสม (วินาที)"
TARGET_HEADER = re.compile(r"เป้า ID (\d+) \(มม\.\)")


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("ต้องมากกว่า 0")
    return number


def find_session_csv(session_path: Path) -> Path:
    csv_path = session_path / "movement_comparison.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์: {csv_path}")
    return csv_path


def read_session(session_path: Path, marker_id: int, interval: float) -> dict[float, float]:
    """Read one marker distance series and align timestamps to the log interval."""
    csv_path = find_session_csv(session_path)
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"ไฟล์ว่าง: {csv_path}")
        target_column = next(
            (name for name in reader.fieldnames if TARGET_HEADER.fullmatch(name or "")
             and int(TARGET_HEADER.fullmatch(name).group(1)) == marker_id),
            None,
        )
        if target_column is None:
            raise ValueError(f"ไม่พบเป้า ID {marker_id} ใน {csv_path}")

        values = {}
        for row in reader:
            raw_time = row.get(TIME_HEADER)
            raw_distance = row.get(target_column)
            if not raw_time or not raw_distance:
                continue
            elapsed = float(raw_time)
            aligned_time = round(elapsed / interval) * interval
            # A duplicate can occur if the camera emitted two points in one interval.
            # Keep the point closest to the intended sampling instant.
            old_value = values.get(aligned_time)
            if old_value is None or abs(elapsed - aligned_time) < abs(values[aligned_time][0] - aligned_time):
                values[aligned_time] = (elapsed, float(raw_distance))
    return {aligned: distance for aligned, (_, distance) in values.items()}


def read_reference(path: Path, interval: float) -> dict[float, float]:
    """Read time/distance reference CSV. The first two numeric columns are used."""
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError("ไฟล์อ้างอิงต้องมีหัวตารางและข้อมูลอย่างน้อย 1 แถว")
    values = {}
    for row in rows[1:]:
        numeric = []
        for value in row:
            try:
                numeric.append(float(value))
            except (TypeError, ValueError):
                continue
        if len(numeric) >= 2:
            aligned_time = round(numeric[0] / interval) * interval
            values[aligned_time] = numeric[1]
    if not values:
        raise ValueError("ไม่พบตัวเลขเวลาและระยะในไฟล์อ้างอิง")
    return values


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def make_svg(path: Path, time_rows, run_labels: list[str], angle: float, marker_id: int) -> None:
    """Create a lightweight graph without adding a plotting dependency."""
    all_values = [value for row in time_rows for value in row["runs"] if value is not None]
    if not all_values:
        return
    times = [row["time"] for row in time_rows]
    min_time, max_time = min(times), max(times)
    min_value, max_value = min(all_values), max(all_values)
    if max_time == min_time:
        max_time += 1
    padding = max((max_value - min_value) * 0.08, 1.0)
    min_value -= padding
    max_value += padding

    width, height = 1200, 720
    left, right, top, bottom = 100, 55, 95, 100
    chart_width, chart_height = width - left - right, height - top - bottom

    def point(time_value, distance):
        x = left + (time_value - min_time) / (max_time - min_time) * chart_width
        y = top + (max_value - distance) / (max_value - min_value) * chart_height
        return x, y

    colors = ["#95a5a6", "#e67e22", "#8e44ad", "#16a085", "#c0392b"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="38" font-family="Arial" font-size="24" font-weight="bold">Repeatability: {angle:g} degrees, Target ID {marker_id}</text>',
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#444"/>',
    ]
    for step in range(6):
        fraction = step / 5
        x = left + fraction * chart_width
        y = top + fraction * chart_height
        time_value = min_time + fraction * (max_time - min_time)
        distance = max_value - fraction * (max_value - min_value)
        parts.extend([
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + chart_height}" stroke="#e5e5e5"/>',
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#e5e5e5"/>',
            f'<text x="{x - 12:.1f}" y="{top + chart_height + 30}" font-family="Arial" font-size="14">{time_value:.0f}</text>',
            f'<text x="20" y="{y + 5:.1f}" font-family="Arial" font-size="14">{distance:.1f}</text>',
        ])
    for index, label in enumerate(run_labels):
        points = [point(row["time"], row["runs"][index]) for row in time_rows if row["runs"][index] is not None]
        if points:
            coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{colors[index % len(colors)]}" stroke-width="2" stroke-dasharray="7 5"/>')
    mean_points = [point(row["time"], row["mean"]) for row in time_rows if row["mean"] is not None]
    if mean_points:
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in mean_points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="#1565c0" stroke-width="4"/>')
    for index, label in enumerate(run_labels):
        y = 67
        x = left + index * 165
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 24}" y2="{y}" stroke="{colors[index % len(colors)]}" stroke-width="3"/>')
        parts.append(f'<text x="{x + 30}" y="{y + 5}" font-family="Arial" font-size="14">{label}</text>')
    parts.extend([
        f'<line x1="{left + len(run_labels) * 165}" y1="67" x2="{left + len(run_labels) * 165 + 24}" y2="67" stroke="#1565c0" stroke-width="4"/>',
        f'<text x="{left + len(run_labels) * 165 + 30}" y="72" font-family="Arial" font-size="14">Mean</text>',
        f'<text x="{width / 2 - 85}" y="{height - 28}" font-family="Arial" font-size="16">Elapsed time (seconds)</text>',
        f'<text x="25" y="80" font-family="Arial" font-size="16">Distance (mm)</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="สรุปความคลาดเคลื่อนระหว่างชุดทดสอบ ArUco")
    parser.add_argument("--sessions", nargs="+", required=True,
                        help="โฟลเดอร์ session อย่างน้อย 2 รอบ")
    parser.add_argument("--marker-id", type=int, required=True, help="ID เป้าที่ต้องการเปรียบเทียบ")
    parser.add_argument("--angle", type=float, required=True, help="มุมทดสอบ (องศา)")
    parser.add_argument("--interval", type=positive_float, default=1.0,
                        help="ช่วงเวลาที่บันทึกข้อมูล (วินาที, ค่าเริ่มต้น 1)")
    parser.add_argument("--reference", type=Path,
                        help="CSV ค่าจริง: คอลัมน์เวลาและระยะ (มม.) เพื่อคำนวณ RMSE/R²")
    parser.add_argument("--output-directory", type=Path, default=Path("repeatability_results"),
                        help="โฟลเดอร์เก็บผลวิเคราะห์")
    args = parser.parse_args()
    if len(args.sessions) < 2:
        parser.error("ระบุ session อย่างน้อย 2 รอบ")

    session_paths = [Path(item) for item in args.sessions]
    series = [read_session(path, args.marker_id, args.interval) for path in session_paths]
    run_labels = [f"รอบ {index + 1}" for index in range(len(series))]
    all_times = sorted(set().union(*(run.keys() for run in series)))
    time_rows = []
    pooled_deviations = []
    for time_value in all_times:
        runs = [run.get(time_value) for run in series]
        valid = [value for value in runs if value is not None]
        average = mean(valid) if valid else None
        sd = stdev(valid) if len(valid) >= 2 else None
        if average is not None:
            pooled_deviations.extend(value - average for value in valid)
        time_rows.append({
            "time": time_value,
            "runs": runs,
            "n": len(valid),
            "mean": average,
            "sd": sd,
            "minimum": min(valid) if valid else None,
            "maximum": max(valid) if valid else None,
            "range": max(valid) - min(valid) if valid else None,
            "cv": (sd / average * 100) if sd is not None and average not in (None, 0) else None,
        })

    output_dir = args.output_directory / f"angle_{args.angle:g}_target_{args.marker_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    time_csv = output_dir / "repeatability_by_time.csv"
    time_headers = ["เวลา (วินาที)", *[f"ระยะ {label} (มม.)" for label in run_labels],
                    "จำนวนรอบที่มีข้อมูล", "ค่าเฉลี่ย (มม.)", "SD (มม.)", "ต่ำสุด (มม.)",
                    "สูงสุด (มม.)", "ช่วงต่าง (มม.)", "CV (%)"]
    write_csv(time_csv, time_headers, [
        [row["time"], *row["runs"], row["n"], row["mean"], row["sd"], row["minimum"],
         row["maximum"], row["range"], row["cv"]]
        for row in time_rows
    ])

    repeatability_rmse = math.sqrt(sum(value ** 2 for value in pooled_deviations) / len(pooled_deviations)) if pooled_deviations else None
    per_time_sd = [row["sd"] for row in time_rows if row["sd"] is not None]
    summary_rows = [
        ["มุมทดสอบ (องศา)", args.angle],
        ["เป้า ID", args.marker_id],
        ["จำนวนรอบทดสอบ", len(series)],
        ["ช่วงเวลาที่นำมาเทียบ", len(time_rows)],
        ["SD เฉลี่ยระหว่างรอบ (มม.)", mean(per_time_sd) if per_time_sd else None],
        ["Repeatability RMSE (มม.)", repeatability_rmse],
        ["RMSE เทียบค่าจริง (มม.)", None],
        ["R² เทียบค่าจริง", None],
    ]

    if args.reference:
        reference = read_reference(args.reference, args.interval)
        comparison_rows = []
        residuals = []
        observed = []
        expected = []
        for row in time_rows:
            ref_value = reference.get(row["time"])
            if row["mean"] is None or ref_value is None:
                continue
            residual = row["mean"] - ref_value
            comparison_rows.append([row["time"], row["mean"], ref_value, residual])
            residuals.append(residual)
            observed.append(row["mean"])
            expected.append(ref_value)
        if comparison_rows:
            reference_rmse = math.sqrt(sum(value ** 2 for value in residuals) / len(residuals))
            expected_mean = mean(expected)
            total_sum_squares = sum((value - expected_mean) ** 2 for value in expected)
            residual_sum_squares = sum(value ** 2 for value in residuals)
            r_squared = 1 - residual_sum_squares / total_sum_squares if total_sum_squares else None
            summary_rows[-2][1] = reference_rmse
            summary_rows[-1][1] = r_squared
            write_csv(output_dir / "reference_comparison.csv",
                      ["เวลา (วินาที)", "OpenCV เฉลี่ย (มม.)", "ค่าจริง (มม.)", "ความต่าง (มม.)"],
                      comparison_rows)

    summary_csv = output_dir / "repeatability_summary.csv"
    write_csv(summary_csv, ["รายการ", "ค่า"], summary_rows)
    graph_path = output_dir / "repeatability_graph.svg"
    make_svg(graph_path, time_rows, run_labels, args.angle, args.marker_id)

    print(f"Saved: {time_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {graph_path}")
    if args.reference:
        print(f"Saved: {output_dir / 'reference_comparison.csv'}")
    if not args.reference:
        print("หมายเหตุ: ยังไม่มี RMSE/R-squared เทียบค่าจริง เพราะไม่ได้ระบุ --reference")


if __name__ == "__main__":
    main()
