"""增量抽取、冲突消解与分层更新。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from tkg_model.build_tkg import CLEANED
from tkg_model.graph_store import CONFIG_PATH, DB_PATH, connect, dump_json, init_db, insert_entities, insert_quads
from tkg_model.holidays import parse_iso, time_features
from tkg_model.schema import SUCCESSION_CONFIDENCE, SUCCESSION_SECONDS
from tkg_pipeline.config import OUTPUT_DIR

REQUIRED = ["case_number", "crime_type_id", "grid_id", "time_id", "iso8601", "unix_timestamp", "city_code", "grid_x", "grid_y"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _batch_id() -> str:
    return "inc-" + datetime.now().strftime("%Y%m%d%H%M%S")


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {"schedule": "manual"}


def save_config(schedule: str) -> None:
    payload = load_config()
    payload["schedule"] = schedule
    dump_json(CONFIG_PATH, payload)


def is_due(schedule: str) -> bool:
    if schedule == "manual":
        return True
    row = None
    if DB_PATH.exists():
        conn = connect()
        row = conn.execute("SELECT finished_at FROM update_log ORDER BY finished_at DESC LIMIT 1").fetchone()
        conn.close()
    if row is None or row["finished_at"] is None:
        return True
    last = datetime.fromisoformat(row["finished_at"])
    gap = {"day": timedelta(days=1), "week": timedelta(days=7), "month": timedelta(days=30)}[schedule]
    return datetime.now(timezone.utc) - last >= gap


def make_demo(path) -> None:
    sample = pd.read_csv(CLEANED, nrows=8)
    duplicate = sample.head(5).copy()
    incoming = sample.tail(3).copy()
    incoming["case_number"] = ["INC-DEMO-1", "INC-DEMO-2", "INC-DEMO-3"]

    def shift(value: str) -> str:
        return (parse_iso(value) + timedelta(hours=5)).isoformat()

    incoming["iso8601"] = incoming["iso8601"].map(shift)
    for column in ("time_start_iso", "time_end_iso"):
        if column in incoming.columns:
            incoming[column] = incoming[column].map(lambda value: shift(value) if pd.notna(value) and str(value).strip() else value)
    incoming["unix_timestamp"] = incoming["iso8601"].map(lambda value: int(parse_iso(value).timestamp()))
    if "time_start_iso" in incoming.columns:
        incoming["time_start_unix"] = incoming["time_start_iso"].map(lambda value: int(parse_iso(value).timestamp()) if pd.notna(value) and str(value).strip() else value)
        incoming["time_end_unix"] = incoming["time_end_iso"].map(lambda value: int(parse_iso(value).timestamp()) if pd.notna(value) and str(value).strip() else value)
    if "time_grain" in incoming.columns:
        fine = incoming["time_grain"].isin(["second", "minute", "hour"])
        incoming.loc[fine, "time_id"] = incoming.loc[fine, "iso8601"].map(lambda value: time_features(parse_iso(value))["hour_id"])
    else:
        incoming["time_id"] = incoming["iso8601"].map(lambda value: time_features(parse_iso(value))["hour_id"])
    pd.concat([duplicate, incoming], ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")


def _case_exists(conn, case_number: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM quads WHERE case_number = ? AND relation_subtype = '同一网格关联' LIMIT 1",
        (case_number,),
    ).fetchone()
    return row is not None


def _field(rec, name: str, default: float) -> float:
    if not hasattr(rec, name):
        return default
    value = getattr(rec, name)
    if value is None or pd.isna(value):
        return default
    return float(value)


def _ensure_coarse_time(conn, time_id: str, stamp: str, batch_id: str) -> int:
    """旬、日、周、月实体不从补全钟点再生成小时节点。"""
    if conn.execute("SELECT 1 FROM entities WHERE entity_id = ?", (time_id,)).fetchone():
        return 0
    moment = parse_iso(stamp)
    features = time_features(moment)
    level = {"TD": "日", "TN": "旬", "TM": "月", "TW": "周"}.get(time_id[:2], "时间")
    attrs = {"granularity": level, "event_count": 0}
    insert_entities(
        conn,
        [(
            time_id,
            "时间单元",
            time_id[2:],
            stamp[:10],
            level,
            f"{level}级时间单元 {stamp[:10]}+08:00",
            json.dumps(attrs, ensure_ascii=False),
            batch_id,
            1,
        )],
    )
    created = 1
    if not time_id.startswith("TD"):
        return created
    for entity_id, code, name, parent_level, duration in (
        (features["week_id"], features["week_id"][2:], f"{features['iso_year']}-W{features['iso_week']:02d}", "周", 24 * 7),
        (features["month_id"], features["month_id"][2:], moment.strftime("%Y-%m"), "月", None),
    ):
        if conn.execute("SELECT 1 FROM entities WHERE entity_id = ?", (entity_id,)).fetchone():
            continue
        insert_entities(
            conn,
            [(
                entity_id,
                "时间单元",
                code,
                name,
                parent_level,
                f"{parent_level}级时间单元 {name}+08:00",
                json.dumps({"granularity": parent_level, "duration_hours": duration, "event_count": 0}, ensure_ascii=False),
                batch_id,
                1,
            )],
        )
        created += 1
    return created


def _ensure_time_entities(conn, stamp: str, batch_id: str) -> int:
    moment = parse_iso(stamp)
    features = time_features(moment)
    created = 0
    units = [
        (features["hour_id"], features["hour_id"][1:], moment.strftime("%Y-%m-%dT%H"), "小时", 1, True),
        (features["day_id"], features["day_id"][2:], moment.strftime("%Y-%m-%d"), "日", 24, True),
        (features["week_id"], features["week_id"][2:], f"{features['iso_year']}-W{features['iso_week']:02d}", "周", 24 * 7, False),
        (features["month_id"], features["month_id"][2:], moment.strftime("%Y-%m"), "月", None, False),
    ]
    fresh = []
    for entity_id, code, name, level, duration, use_holiday in units:
        if conn.execute("SELECT 1 FROM entities WHERE entity_id = ?", (entity_id,)).fetchone():
            continue
        holiday = features["holiday_name"] if use_holiday else ""
        attrs = {
            "granularity": level,
            "period": features["period"] if level == "小时" else "",
            "is_weekend": features["is_weekend"] if use_holiday else False,
            "is_holiday": bool(holiday),
            "holiday_name": holiday,
            "duration_hours": duration,
            "event_count": 0,
        }
        fresh.append(
            (
                entity_id,
                "时间单元",
                code,
                name,
                level,
                f"{level}级时间单元 {name}+08:00",
                json.dumps(attrs, ensure_ascii=False),
                batch_id,
                1,
            )
        )
        created += 1
    if fresh:
        insert_entities(conn, fresh)
    links = (
        (features["hour_id"], features["day_id"]),
        (features["day_id"], features["month_id"]),
        (features["day_id"], features["week_id"]),
    )
    link_rows = []
    for child, parent in links:
        if conn.execute(
            "SELECT 1 FROM quads WHERE entity1_id = ? AND relation_subtype = '时间粒度隶属' AND entity2_id = ?",
            (child, parent),
        ).fetchone():
            continue
        link_rows.append(
            (
                child,
                "时空关联",
                "时间粒度隶属",
                parent,
                stamp,
                stamp,
                stamp,
                1.0,
                1.0,
                "rule",
                "",
                "",
                int(moment.timestamp()),
                0.0,
                batch_id,
                1,
            )
        )
    if link_rows:
        insert_quads(conn, link_rows)
    return created


def _bump(conn, entity_id: str, batch_id: str, changed_at: str) -> None:
    row = conn.execute("SELECT attrs_json FROM entities WHERE entity_id = ?", (entity_id,)).fetchone()
    if row is None:
        return
    attrs = json.loads(row["attrs_json"] or "{}")
    old = int(attrs.get("event_count", 0))
    attrs["event_count"] = old + 1
    if entity_id.startswith("GG"):
        attrs["crime_density_per_km2"] = round(attrs["event_count"] / 0.25, 4)
    conn.execute(
        "UPDATE entities SET attrs_json = ? WHERE entity_id = ?",
        (json.dumps(attrs, ensure_ascii=False), entity_id),
    )
    conn.execute(
        """
        INSERT INTO attribute_history (batch_id, entity_id, attribute, old_value, new_value, changed_at)
        VALUES (?, ?, 'event_count', ?, ?, ?)
        """,
        (batch_id, entity_id, str(old), str(old + 1), changed_at),
    )


def _successors(conn, grid_id: str, unix_ts: int, case_number: str, crime_type_id: str, stamp: str, batch_id: str) -> list[tuple]:
    window_start = unix_ts - SUCCESSION_SECONDS
    window_end = unix_ts + SUCCESSION_SECONDS
    found = conn.execute(
        """
        SELECT case_number, entity1_id, iso8601_timestamp, unix_ts
        FROM quads
        WHERE relation_subtype = '同一网格关联' AND entity2_id = ?
          AND unix_ts BETWEEN ? AND ? AND case_number != ?
        ORDER BY unix_ts
        """,
        (grid_id, window_start, window_end, case_number),
    ).fetchall()
    earlier = [row for row in found if row["unix_ts"] < unix_ts]
    later = [row for row in found if row["unix_ts"] > unix_ts]
    chosen = []
    if earlier:
        chosen.append(earlier[-1])
    if later:
        chosen.append(later[0])
    rows = []
    for other in chosen:
        if other["unix_ts"] < unix_ts:
            left, right = other, {"entity1_id": crime_type_id, "iso8601_timestamp": stamp, "unix_ts": unix_ts, "case_number": case_number}
        else:
            left = {"entity1_id": crime_type_id, "iso8601_timestamp": stamp, "unix_ts": unix_ts, "case_number": case_number}
            right = other
        delta = int(right["unix_ts"]) - int(left["unix_ts"])
        strength = max(0.01, 1.0 - delta / SUCCESSION_SECONDS)
        rows.append(
            (
                left["entity1_id"],
                "时空关联",
                "短时序相继关联",
                right["entity1_id"],
                right["iso8601_timestamp"],
                left["iso8601_timestamp"],
                right["iso8601_timestamp"],
                SUCCESSION_CONFIDENCE,
                round(strength, 4),
                "rule",
                right["case_number"],
                left["case_number"],
                int(right["unix_ts"]),
                round(delta / 3600.0, 4),
                batch_id,
                1,
            )
        )
    return rows


def apply_increment(csv_path, mode: str = "manual") -> dict:
    if not DB_PATH.exists():
        raise SystemExit("还没有图谱库。请先运行 py -3 run_tkg_build.py")
    frame = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED if column not in frame.columns]
    if missing:
        raise SystemExit(f"增量文件缺少字段：{', '.join(missing)}。请先按清洗流水线做成标准化记录。")
    conn = connect()
    init_db(conn)
    batch_id = _batch_id()
    started = _now()
    new_entities = 0
    new_quads = 0
    skipped = 0
    conflicts = 0
    for rec in frame.itertuples(index=False):
        case_number = str(rec.case_number)
        if _case_exists(conn, case_number):
            skipped += 1
            continue
        crime_id = str(rec.crime_type_id)
        grid_id = str(rec.grid_id)
        if conn.execute("SELECT 1 FROM entities WHERE entity_id = ?", (crime_id,)).fetchone() is None:
            conflicts += 1
            conn.execute(
                """
                INSERT INTO conflicts (
                    batch_id, conflict_type, entity1_id, relation_subtype, entity2_id,
                    iso8601_timestamp, case_number, existing_value, incoming_value, resolution, created_at
                ) VALUES (?, 'unknown_crime_type', ?, '', '', ?, ?, '', ?, 'rejected', ?)
                """,
                (batch_id, crime_id, rec.iso8601, case_number, crime_id, started),
            )
            continue
        if conn.execute("SELECT 1 FROM entities WHERE entity_id = ?", (grid_id,)).fetchone() is None:
            conflicts += 1
            conn.execute(
                """
                INSERT INTO conflicts (
                    batch_id, conflict_type, entity1_id, relation_subtype, entity2_id,
                    iso8601_timestamp, case_number, existing_value, incoming_value, resolution, created_at
                ) VALUES (?, 'unknown_grid', ?, '', ?, ?, ?, '', ?, 'rejected', ?)
                """,
                (batch_id, crime_id, grid_id, rec.iso8601, case_number, grid_id, started),
            )
            continue
        stamp = str(rec.iso8601)
        time_id = str(rec.time_id)
        if time_id.startswith(("TD", "TN", "TM", "TW")):
            new_entities += _ensure_coarse_time(conn, time_id, stamp, batch_id)
        else:
            new_entities += _ensure_time_entities(conn, stamp, batch_id)
        unix_ts = int(rec.unix_timestamp)
        time_conf = _field(rec, "time_align_confidence", 1.0)
        spatial_conf = _field(rec, "spatial_align_confidence", 1.0)
        valid_from = getattr(rec, "time_start_iso", "")
        valid_to = getattr(rec, "time_end_iso", "")
        if valid_from is None or pd.isna(valid_from) or str(valid_from).strip() in {"", "nan", "None"}:
            valid_from = stamp
        if valid_to is None or pd.isna(valid_to) or str(valid_to).strip() in {"", "nan", "None"}:
            valid_to = stamp
        valid_from, valid_to = str(valid_from), str(valid_to)
        quad_rows = [
            (
                crime_id,
                "时空关联",
                "同一网格关联",
                grid_id,
                stamp,
                valid_from,
                valid_to,
                spatial_conf,
                spatial_conf,
                "align",
                case_number,
                "",
                unix_ts,
                0.0,
                batch_id,
                1,
            ),
            (
                crime_id,
                "时空关联",
                "发生于时刻",
                time_id,
                stamp,
                valid_from,
                valid_to,
                time_conf,
                time_conf,
                "align",
                case_number,
                "",
                unix_ts,
                0.0,
                batch_id,
                1,
            ),
        ]
        if time_id in {"", "nan", "None", "<NA>"}:
            quad_rows = quad_rows[:1]
        grain = str(getattr(rec, "time_grain", "hour") or "hour")
        if grain in {"second", "minute", "hour", ""}:
            quad_rows.extend(_successors(conn, grid_id, unix_ts, case_number, crime_id, stamp, batch_id))
        new_quads += insert_quads(conn, quad_rows)
        changed = started
        for entity_id in (crime_id, grid_id, str(rec.time_id)):
            _bump(conn, entity_id, batch_id, changed)
        admin_code = str(getattr(rec, "admin_code", "")).replace(".0", "").strip()
        county_id = "GD" + admin_code.zfill(6) if admin_code.isdigit() else ""
        if county_id and conn.execute("SELECT 1 FROM entities WHERE entity_id = ?", (county_id,)).fetchone():
            _bump(conn, county_id, batch_id, changed)
    finished = _now()
    conn.execute(
        """
        INSERT INTO update_log (
            batch_id, started_at, finished_at, mode, source_file,
            new_entities, new_quads, duplicate_skipped, conflicts, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (batch_id, started, finished, mode, str(csv_path), new_entities, new_quads, skipped, conflicts, "增量融合完成"),
    )
    conn.commit()
    conn.close()
    save_config(load_config().get("schedule", mode if mode in {"day", "week", "month", "manual"} else "manual"))
    report = {
        "batch_id": batch_id,
        "source_file": str(csv_path),
        "new_entities": new_entities,
        "new_quads": new_quads,
        "duplicate_skipped": skipped,
        "conflicts": conflicts,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="增量更新犯罪时序知识图谱")
    parser.add_argument("--input", help="标准化新案件 CSV")
    parser.add_argument("--demo", action="store_true", help="用少量样例演示去重与新增")
    parser.add_argument("--schedule", choices=["day", "week", "month", "manual"], help="记录自动更新粒度")
    parser.add_argument("--if-due", action="store_true", help="未到更新周期时不执行")
    args = parser.parse_args()
    if args.schedule:
        save_config(args.schedule)
    schedule = args.schedule or load_config().get("schedule", "manual")
    if args.if_due and not is_due(schedule):
        print(f"未到{schedule}更新周期", flush=True)
        return
    if args.demo:
        path = OUTPUT_DIR / "tkg" / "demo_increment.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        make_demo(path)
        apply_increment(path, mode="manual")
        return
    if args.input:
        apply_increment(args.input, mode=schedule)
        return
    if args.schedule:
        print(f"已将更新频率设为 {args.schedule}", flush=True)
        return
    parser.print_help()
