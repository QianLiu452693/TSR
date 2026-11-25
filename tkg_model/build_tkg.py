"""从标准化犯罪记录构建犯罪场景 TKG，并导出 Neo4j 可导入文件。"""

from __future__ import annotations

import bisect
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from pyproj import Transformer

from tkg_model.graph_store import NEO4J_DIR, TKG_DIR, dump_json, insert_entities, insert_quads, reset_database
from tkg_model.holidays import holiday_name, parse_iso, time_features
from tkg_pipeline.align import TZ, certain_successor_indices, certainly_successive, contained_bins, time_parent_links, to_iso
from tkg_model.schema import (
    ARTICLES,
    BASE_BATCH,
    CAUSAL_COUNTY_MIN_COUNT,
    CAUSAL_GRID_MIN_COUNT,
    CAUSAL_GRID_MIN_LIFT,
    CAUSAL_MIN_LIFT,
    CAUSAL_TYPE_MIN_COUNT,
    CHAPTER_NAME,
    INFER_MIN_CONFIDENCE,
    RELATION_SPECS,
    RULE_CONFIDENCE,
    SUBTYPES,
    SUCCESSION_CONFIDENCE,
    SUCCESSION_SECONDS,
)
from tkg_pipeline.config import OUTPUT_DIR
from tkg_pipeline.crime_types import CrimeTypeDictionary

CLEANED = OUTPUT_DIR / "cleaned_crime_records.csv"
GRID_META = OUTPUT_DIR / "city_grid_meta.csv"
USECOLS = [
    "case_number",
    "crime_type_code",
    "crime_type_name",
    "crime_type_id",
    "latitude",
    "longitude",
    "proj_x",
    "proj_y",
    "province_code",
    "province_name",
    "city_code",
    "city_name",
    "admin_code",
    "admin_name",
    "grid_code",
    "grid_id",
    "grid_x",
    "grid_y",
    "iso8601",
    "unix_timestamp",
    "time_id",
]


def _log(message: str) -> None:
    print(message, flush=True)


def _self_check() -> None:
    from datetime import date

    assert holiday_name(date(2016, 2, 8)) == "春节"
    assert holiday_name(date(2018, 2, 16)) == "春节"
    assert holiday_name(date(2016, 10, 1)) == "国庆节"
    features = time_features(parse_iso("2016-07-21T15:00:00+08:00"))
    assert features["hour_id"] == "T2016072115"
    assert features["period"] == "下午"
    assert features["day_id"] == "TD20160721"


def _quad(
    entity1: str,
    relation: str,
    subtype: str,
    entity2: str,
    stamp: str,
    valid_from: str,
    valid_to: str,
    confidence: float,
    strength: float,
    source: str,
    case_number: str = "",
    other_case: str = "",
    unix_ts: int | None = None,
    window_hours: float = 0.0,
) -> tuple:
    return (
        entity1,
        relation,
        subtype,
        entity2,
        stamp,
        valid_from,
        valid_to,
        round(float(confidence), 4),
        round(float(strength), 4),
        source,
        case_number or "",
        other_case or "",
        None if unix_ts is None else int(unix_ts),
        round(float(window_hours), 4),
        BASE_BATCH,
        0,
    )


def _entity(entity_id: str, entity_class: str, code: str, name: str, level: str, description: str, attrs: dict) -> tuple:
    return (
        entity_id,
        entity_class,
        code,
        name,
        level,
        description,
        json.dumps(attrs, ensure_ascii=False),
        BASE_BATCH,
        0,
    )


def _lift_confidence(lift: float) -> float:
    if lift <= 1:
        return 0.0
    return float(min(0.99, 1.0 - 1.0 / lift))


def _from_unix(value: int) -> datetime:
    # Windows 的 fromtimestamp 对过早的时间戳会抛 OSError。
    return (datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=int(value))).astimezone(TZ)


def load_events() -> pd.DataFrame:
    if not CLEANED.exists():
        raise SystemExit("缺少 output/cleaned_crime_records.csv，请先运行 py -3 -m tkg_pipeline.pipeline")
    _log("读取标准化犯罪记录")
    header = pd.read_csv(CLEANED, nrows=0).columns.tolist()
    optional = ["time_grain", "time_align_confidence", "time_start_unix", "time_end_unix", "spatial_align_confidence", "time_entity_level"]
    usecols = [column for column in USECOLS + optional if column in header]
    frame = pd.read_csv(CLEANED, usecols=usecols)
    frame["crime_type_code"] = frame["crime_type_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    frame["grid_id"] = frame["grid_id"].astype("string")
    frame.loc[frame["grid_id"].str.strip().isin(["", "<NA>", "nan", "None"]), "grid_id"] = pd.NA
    for column in ["province_code", "city_code", "admin_code"]:
        frame[column] = frame[column].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    frame["grid_code"] = frame["grid_code"].astype(str).str.replace(r"\.0$", "", regex=True)
    frame["grid_x"] = pd.to_numeric(frame["grid_x"], errors="coerce").astype("Int64")
    frame["grid_y"] = pd.to_numeric(frame["grid_y"], errors="coerce").astype("Int64")
    frame["unix_timestamp"] = pd.to_numeric(frame["unix_timestamp"], errors="coerce")
    frame = frame.dropna(
        subset=["crime_type_id", "grid_id", "iso8601", "admin_code", "city_code", "province_code", "unix_timestamp", "grid_x", "grid_y"]
    ).copy()
    frame["unix_timestamp"] = frame["unix_timestamp"].astype(np.int64)
    frame["time_id"] = frame["time_id"].fillna("").astype(str).replace({"nan": "", "<NA>": "", "None": ""})
    if "time_start_unix" not in frame.columns:
        frame["time_start_unix"] = frame["unix_timestamp"]
        frame["time_end_unix"] = frame["unix_timestamp"]
        frame["time_grain"] = "hour"
        frame["time_align_confidence"] = 1.0
        frame["spatial_align_confidence"] = 1.0
    else:
        frame["time_start_unix"] = pd.to_numeric(frame["time_start_unix"], errors="coerce")
        frame["time_end_unix"] = pd.to_numeric(frame["time_end_unix"], errors="coerce")
        missing_interval = frame["time_start_unix"].isna() | frame["time_end_unix"].isna()
        frame.loc[missing_interval, "time_start_unix"] = frame.loc[missing_interval, "unix_timestamp"]
        frame.loc[missing_interval, "time_end_unix"] = frame.loc[missing_interval, "unix_timestamp"]
        if "time_align_confidence" not in frame.columns:
            frame["time_align_confidence"] = 1.0
        if "spatial_align_confidence" not in frame.columns:
            frame["spatial_align_confidence"] = 1.0
        frame["time_align_confidence"] = pd.to_numeric(frame["time_align_confidence"], errors="coerce").fillna(1.0)
        frame["spatial_align_confidence"] = pd.to_numeric(frame["spatial_align_confidence"], errors="coerce").fillna(1.0)
    frame["time_start_unix"] = frame["time_start_unix"].astype(np.int64)
    frame["time_end_unix"] = frame["time_end_unix"].astype(np.int64)
    frame["county_id"] = "GD" + frame["admin_code"]
    frame["city_id"] = "GC" + frame["city_code"]
    frame["province_id"] = "GP" + frame["province_code"]
    _log(f"事件 {len(frame)} 条，其中无单一时间实体 {int(frame['time_id'].eq('').sum())}")
    return frame


def _grid_rectangles(events: pd.DataFrame) -> pd.DataFrame:
    grids = (
        events.groupby(["grid_id", "grid_code", "city_code", "grid_x", "grid_y"], sort=False)
        .agg(
            event_count=("case_number", "size"),
            lat_min=("latitude", "min"),
            lat_max=("latitude", "max"),
            lon_min=("longitude", "min"),
            lon_max=("longitude", "max"),
            city_name=("city_name", "first"),
        )
        .reset_index()
    )
    meta = pd.read_csv(GRID_META, dtype={"city_code": str})
    meta["city_code"] = meta["city_code"].str.replace(r"\.0$", "", regex=True).str.zfill(6)
    grids = grids.merge(meta, on="city_code", how="left", suffixes=("", "_meta"))
    origin_x = grids["origin_x"].to_numpy(dtype=float)
    origin_y = grids["origin_y"].to_numpy(dtype=float)
    gx = grids["grid_x"].to_numpy(dtype=float)
    gy = grids["grid_y"].to_numpy(dtype=float)
    known = np.isfinite(origin_x) & np.isfinite(origin_y)
    grids["proj_xmin"] = np.where(known, origin_x + gx * 500.0, np.nan)
    grids["proj_xmax"] = np.where(known, grids["proj_xmin"] + 500.0, np.nan)
    grids["proj_ymin"] = np.where(known, origin_y + gy * 500.0, np.nan)
    grids["proj_ymax"] = np.where(known, grids["proj_ymin"] + 500.0, np.nan)
    grids["gk_lon0"] = grids["gk_lon0"]
    grids["crime_density"] = grids["event_count"] / 0.25
    return grids


def _inverse_bounds(grids: pd.DataFrame) -> pd.DataFrame:
    grids = grids.copy()
    grids["grid_lon_min"] = np.nan
    grids["grid_lon_max"] = np.nan
    grids["grid_lat_min"] = np.nan
    grids["grid_lat_max"] = np.nan
    usable = grids["proj_xmin"].notna() & grids["gk_lon0"].notna()
    for lon0, index in grids.loc[usable].groupby("gk_lon0").groups.items():
        block = grids.loc[index]
        proj = (
            f"+proj=tmerc +lat_0=0 +lon_0={int(lon0)} +k=1 +x_0=500000 +y_0=0 "
            "+ellps=GRS80 +units=m +no_defs"
        )
        inverse = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
        xmin = block["proj_xmin"].to_numpy()
        xmax = block["proj_xmax"].to_numpy()
        ymin = block["proj_ymin"].to_numpy()
        ymax = block["proj_ymax"].to_numpy()
        xs = np.column_stack([xmin, xmax, xmin, xmax]).ravel()
        ys = np.column_stack([ymin, ymin, ymax, ymax]).ravel()
        lons, lats = inverse.transform(xs, ys)
        lons = lons.reshape(-1, 4)
        lats = lats.reshape(-1, 4)
        grids.loc[block.index, "grid_lon_min"] = pd.Series(lons.min(axis=1), index=block.index)
        grids.loc[block.index, "grid_lon_max"] = pd.Series(lons.max(axis=1), index=block.index)
        grids.loc[block.index, "grid_lat_min"] = pd.Series(lats.min(axis=1), index=block.index)
        grids.loc[block.index, "grid_lat_max"] = pd.Series(lats.max(axis=1), index=block.index)
    return grids


def build_entities(events: pd.DataFrame, grids: pd.DataFrame, dictionary: CrimeTypeDictionary) -> list[tuple]:
    _log("构建犯罪类型、地理区域、时间单元实体")
    rows: list[tuple] = []
    present = set(events["crime_type_code"].astype(str))
    type_counts = events["crime_type_id"].value_counts().to_dict()
    for code, spec in dictionary.by_code.items():
        if code not in present:
            continue
        chapter = CHAPTER_NAME.get(spec["chapter"], spec["chapter"])
        entity_id = f"C{code}"
        rows.append(
            _entity(
                entity_id,
                "犯罪类型",
                code,
                spec["name"],
                chapter,
                f"{chapter}；{ARTICLES.get(code, '')}；构成要件：{'、'.join(spec['elements'])}",
                {
                    "chapter": chapter,
                    "article": ARTICLES.get(code, ""),
                    "elements": spec["elements"],
                    "aliases": spec["aliases"],
                    "subtypes": SUBTYPES.get(code, []),
                    "event_count": int(type_counts.get(entity_id, 0)),
                },
            )
        )

    def add_admin(code_col: str, name_col: str, prefix: str, level: str) -> None:
        table = (
            events.groupby(code_col, sort=False)
            .agg(
                name=(name_col, "first"),
                event_count=("case_number", "size"),
                lat_min=("latitude", "min"),
                lat_max=("latitude", "max"),
                lon_min=("longitude", "min"),
                lon_max=("longitude", "max"),
            )
            .reset_index()
        )
        for rec in table.itertuples(index=False):
            code = str(getattr(rec, code_col))
            rows.append(
                _entity(
                    prefix + code,
                    "地理空间",
                    code,
                    str(rec.name),
                    level,
                    f"{level} {rec.name}（{code}）",
                    {
                        "admin_code": code,
                        "event_count": int(rec.event_count),
                        "lat_min": round(float(rec.lat_min), 6),
                        "lat_max": round(float(rec.lat_max), 6),
                        "lon_min": round(float(rec.lon_min), 6),
                        "lon_max": round(float(rec.lon_max), 6),
                    },
                )
            )

    add_admin("province_code", "province_name", "GP", "省级")
    add_admin("city_code", "city_name", "GC", "地级")
    add_admin("admin_code", "admin_name", "GD", "县级")
    for rec in grids.itertuples(index=False):
        attrs = {
            "grid_code": rec.grid_code,
            "city_code": rec.city_code,
            "event_count": int(rec.event_count),
            "crime_density_per_km2": round(float(rec.crime_density), 4),
            "lat_min": round(float(rec.lat_min), 6),
            "lat_max": round(float(rec.lat_max), 6),
            "lon_min": round(float(rec.lon_min), 6),
            "lon_max": round(float(rec.lon_max), 6),
        }
        if pd.notna(rec.proj_xmin):
            attrs.update(
                {
                    "proj_xmin": round(float(rec.proj_xmin), 2),
                    "proj_xmax": round(float(rec.proj_xmax), 2),
                    "proj_ymin": round(float(rec.proj_ymin), 2),
                    "proj_ymax": round(float(rec.proj_ymax), 2),
                    "gk_lon0": int(rec.gk_lon0),
                    "grid_lon_min": None if pd.isna(rec.grid_lon_min) else round(float(rec.grid_lon_min), 6),
                    "grid_lon_max": None if pd.isna(rec.grid_lon_max) else round(float(rec.grid_lon_max), 6),
                    "grid_lat_min": None if pd.isna(rec.grid_lat_min) else round(float(rec.grid_lat_min), 6),
                    "grid_lat_max": None if pd.isna(rec.grid_lat_max) else round(float(rec.grid_lat_max), 6),
                }
            )
        rows.append(
            _entity(
                rec.grid_id,
                "地理空间",
                rec.grid_code,
                f"{rec.city_name}{str(rec.grid_code)[-6:]}网格",
                "城市网格",
                f"500米网格 {rec.grid_code}",
                attrs,
            )
        )

    seen_time: set[str] = set()
    timed = events.loc[events["time_id"].astype(str).str.len().gt(0), ["time_id", "time_start_unix", "time_end_unix"]].drop_duplicates("time_id")
    for rec in timed.itertuples(index=False):
        start = _from_unix(int(rec.time_start_unix))
        end = _from_unix(int(rec.time_end_unix))
        for entity_id, level, left, right in contained_bins(start, end):
            if entity_id in seen_time:
                continue
            seen_time.add(entity_id)
            features = time_features(left)
            holiday_level = level in {"小时", "日"}
            code = entity_id[2:] if entity_id[:2] in {"TD", "TN", "TM", "TW"} else entity_id[1:]
            if level == "小时":
                name = left.strftime("%Y-%m-%dT%H")
            elif level == "日":
                name = left.strftime("%Y-%m-%d")
            elif level == "周":
                name = f"{features['iso_year']}-W{features['iso_week']:02d}"
            elif level == "旬":
                name = left.strftime("%Y-%m") + ("上旬" if left.day <= 10 else "中旬" if left.day <= 20 else "下旬")
            else:
                name = left.strftime("%Y-%m")
            duration = None if level == "月" else round((right - left).total_seconds() / 3600.0, 2)
            rows.append(
                _entity(
                    entity_id,
                    "时间单元",
                    code,
                    name,
                    level,
                    f"{level}级时间单元 {name}+08:00",
                    {
                        "granularity": level,
                        "period": features["period"] if level == "小时" else "",
                        "is_weekend": features["is_weekend"] if holiday_level else False,
                        "is_holiday": features["is_holiday"] if holiday_level else False,
                        "holiday_name": features["holiday_name"] if holiday_level else "",
                        "duration_hours": duration,
                    },
                )
            )
    _log(f"实体 {len(rows)} 个")
    return rows


def structural_edges(events: pd.DataFrame, dictionary: CrimeTypeDictionary, t_min: str, t_max: str, unix_min: int, unix_max: int) -> list[tuple]:
    rows: list[tuple] = []
    window = max((unix_max - unix_min) / 3600.0, 0.0)
    grid_info = events.drop_duplicates("grid_id")[
        ["grid_id", "county_id", "city_id", "province_id", "city_code", "grid_x", "grid_y"]
    ]
    for rec in grid_info.itertuples(index=False):
        rows.extend(
            [
                _quad(rec.grid_id, "时空关联", "空间层级隶属", rec.county_id, t_min, t_min, t_max, RULE_CONFIDENCE, 1.0, "rule", unix_ts=unix_min, window_hours=window),
                _quad(rec.county_id, "时空关联", "空间层级隶属", rec.city_id, t_min, t_min, t_max, RULE_CONFIDENCE, 1.0, "rule", unix_ts=unix_min, window_hours=window),
                _quad(rec.city_id, "时空关联", "空间层级隶属", rec.province_id, t_min, t_min, t_max, RULE_CONFIDENCE, 1.0, "rule", unix_ts=unix_min, window_hours=window),
            ]
        )
    occupied = {
        (str(rec.city_code), int(rec.grid_x), int(rec.grid_y)): rec.grid_id for rec in grid_info.itertuples(index=False)
    }
    for (city, x_index, y_index), grid_id in occupied.items():
        for dx, dy in ((1, 0), (0, 1)):
            other = occupied.get((city, x_index + dx, y_index + dy))
            if other:
                left, right = (grid_id, other) if grid_id < other else (other, grid_id)
                rows.append(
                    _quad(left, "时空关联", "相邻区域关联", right, t_min, t_min, t_max, RULE_CONFIDENCE, 1.0, "rule", unix_ts=unix_min, window_hours=window)
                )
    seen_link: set[tuple[str, str]] = set()
    timed = events.loc[events["time_id"].astype(str).str.len().gt(0), ["time_start_unix", "time_end_unix"]].drop_duplicates()
    for rec in timed.itertuples(index=False):
        links = time_parent_links(_from_unix(int(rec.time_start_unix)), _from_unix(int(rec.time_end_unix)))
        for child, parent in links:
            if (child, parent) in seen_link:
                continue
            seen_link.add((child, parent))
            rows.append(
                _quad(child, "时空关联", "时间粒度隶属", parent, t_min, t_min, t_max, RULE_CONFIDENCE, 1.0, "rule", unix_ts=unix_min, window_hours=window)
            )
    subtype_name = {"类型相似": "类型相似", "类型互斥": "类型互斥", "类型衍生": "类型转化"}
    for pair in dictionary.jaccard_pairs():
        rows.append(
            _quad(
                "C" + pair["code_a"],
                "类型关联",
                subtype_name[pair["relation_type"]],
                "C" + pair["code_b"],
                t_min,
                t_min,
                t_max,
                pair["confidence"],
                pair["confidence"],
                "rule",
                unix_ts=unix_min,
                window_hours=window,
            )
        )
    return rows


def event_edges(conn, events: pd.DataFrame) -> Counter:
    _log("规则抽取同一网格、发生于时刻、24小时相继关系")
    rows: list[tuple] = []
    written = 0

    def flush() -> None:
        nonlocal written
        if not rows:
            return
        written += insert_quads(conn, rows)
        rows.clear()

    buckets: dict[tuple, list[tuple]] = defaultdict(list)
    for rec in events.itertuples(index=False):
        stamp = str(rec.iso8601)
        unix_ts = int(rec.unix_timestamp)
        start_ts = int(rec.time_start_unix)
        end_ts = int(rec.time_end_unix)
        start_iso = to_iso(_from_unix(start_ts))
        end_iso = to_iso(_from_unix(end_ts))
        spatial_conf = float(rec.spatial_align_confidence)
        time_conf = float(rec.time_align_confidence)
        rows.append(
            _quad(
                rec.crime_type_id,
                "时空关联",
                "同一网格关联",
                rec.grid_id,
                stamp,
                start_iso,
                end_iso,
                spatial_conf,
                spatial_conf,
                "align",
                case_number=str(rec.case_number),
                unix_ts=unix_ts,
            )
        )
        if rec.time_id:
            rows.append(
                _quad(
                    rec.crime_type_id,
                    "时空关联",
                    "发生于时刻",
                    rec.time_id,
                    stamp,
                    start_iso,
                    end_iso,
                    time_conf,
                    time_conf,
                    "align",
                    case_number=str(rec.case_number),
                    unix_ts=unix_ts,
                )
            )
        if len(rows) >= 100000:
            flush()
        buckets[(str(rec.city_code), int(rec.grid_x), int(rec.grid_y))].append(
            (start_ts, end_ts, rec.crime_type_id, stamp, str(rec.case_number), time_conf)
        )
    for items in buckets.values():
        items.sort()
    transitions: Counter = Counter()
    point_candidates = 0
    for items in buckets.values():
        if len(items) >= 2:
            starts = np.fromiter((item[0] for item in items), dtype=np.int64, count=len(items))
            point_candidates += int(((starts[1:] - starts[:-1] > 0) & (starts[1:] - starts[:-1] <= SUCCESSION_SECONDS)).sum())
        else:
            starts = None
        ends = np.fromiter((item[1] for item in items), dtype=np.int64, count=len(items)) if items else np.empty(0, dtype=np.int64)
        if starts is None:
            starts = np.fromiter((item[0] for item in items), dtype=np.int64, count=len(items)) if items else np.empty(0, dtype=np.int64)
        ii, jj = certain_successor_indices(starts, ends, SUCCESSION_SECONDS)
        for left, right in zip(ii, jj):
            _append_succession(rows, transitions, items[int(left)], items[int(right)])
            if len(rows) >= 100000:
                flush()
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for key, items in buckets.items():
        city, x_index, y_index = key
        neighbor_lists = [buckets.get((city, x_index + dx, y_index + dy)) for dx, dy in directions]
        neighbor_lists = [other for other in neighbor_lists if other]
        if not neighbor_lists:
            continue
        neighbor_starts = [[item[0] for item in other] for other in neighbor_lists]
        for current in items:
            best = None
            best_delta = SUCCESSION_SECONDS + 1
            limit = current[0] + SUCCESSION_SECONDS
            for other, starts in zip(neighbor_lists, neighbor_starts):
                lo = bisect.bisect_left(starts, max(current[1], current[0] + 1))
                while lo < len(other) and other[lo][0] <= limit:
                    nxt = other[lo]
                    if certainly_successive(current[0], current[1], nxt[0], nxt[1], SUCCESSION_SECONDS):
                        delta = nxt[0] - current[0]
                        if 0 < delta < best_delta:
                            best = nxt
                            best_delta = delta
                        break
                    lo += 1
            if best is not None:
                _append_succession(rows, transitions, current, best)
                if len(rows) >= 100000:
                    flush()
    flush()
    _log(f"事件级关系写入 {written} 条，区间相继 {sum(transitions.values())} 次，点时间候选 {point_candidates} 次")
    return transitions


def _append_succession(rows: list[tuple], transitions: Counter, current: tuple, nxt: tuple) -> None:
    delta = max(int(nxt[0] - current[0]), 1)
    strength = max(0.01, 1.0 - delta / SUCCESSION_SECONDS)
    confidence = min(SUCCESSION_CONFIDENCE, float(current[5]), float(nxt[5]))
    rows.append(
        _quad(
            current[2],
            "时空关联",
            "短时序相继关联",
            nxt[2],
            nxt[3],
            current[3],
            nxt[3],
            confidence,
            strength,
            "align",
            case_number=nxt[4],
            other_case=current[4],
            unix_ts=int(nxt[0]),
            window_hours=delta / 3600.0,
        )
    )
    transitions[(current[2], nxt[2])] += 1


def causal_edges(events: pd.DataFrame, transitions: Counter) -> list[tuple]:
    _log("按共现提升度抽取因果关联")
    rows: list[tuple] = []
    base = events["crime_type_id"].value_counts(normalize=True)
    for geo_col, min_count, min_lift in (
        ("county_id", CAUSAL_COUNTY_MIN_COUNT, CAUSAL_MIN_LIFT),
        ("grid_id", CAUSAL_GRID_MIN_COUNT, CAUSAL_GRID_MIN_LIFT),
    ):
        counts = events.groupby([geo_col, "crime_type_id"], sort=False).size()
        totals = events.groupby(geo_col, sort=False).size()
        spans = events.groupby([geo_col, "crime_type_id"], sort=False)["unix_timestamp"].agg(["min", "max"])
        stamps = events.groupby([geo_col, "crime_type_id"], sort=False)["iso8601"].agg(["min", "max"])
        for key, count in counts.items():
            total = int(totals.loc[key[0]])
            rate = float(base.get(key[1], 0.0))
            if int(count) < min_count or total < min_count or rate <= 0:
                continue
            lift = (count / total) / rate
            if lift < min_lift:
                continue
            confidence = _lift_confidence(lift)
            start = str(stamps.loc[key, "min"])
            end = str(stamps.loc[key, "max"])
            rows.append(
                _quad(
                    key[0],
                    "因果关联",
                    "地理致罪",
                    key[1],
                    end,
                    start,
                    end,
                    confidence,
                    confidence,
                    "rule",
                    unix_ts=int(spans.loc[key, "max"]),
                    window_hours=(int(spans.loc[key, "max"]) - int(spans.loc[key, "min"])) / 3600.0,
                )
            )
    outgoing = Counter()
    for (left, _right), count in transitions.items():
        outgoing[left] += count
    follower = Counter()
    for (_left, right), count in transitions.items():
        follower[right] += count
    total_follow = sum(follower.values()) or 1
    for (left, right), count in transitions.items():
        if left == right or count < CAUSAL_TYPE_MIN_COUNT:
            continue
        reverse = transitions.get((right, left), 0)
        if count < reverse * 1.2:
            continue
        base_rate = follower[right] / total_follow
        conditional = count / outgoing[left]
        if base_rate <= 0:
            continue
        lift = conditional / base_rate
        if lift < CAUSAL_MIN_LIFT:
            continue
        confidence = _lift_confidence(lift)
        rows.append(
            _quad(left, "因果关联", "类型衍生因果", right, events["iso8601"].iloc[len(events) // 2], events["iso8601"].min(), events["iso8601"].max(), confidence, confidence, "rule", unix_ts=int(events["unix_timestamp"].iloc[len(events) // 2]), window_hours=24.0)
        )
    _log(f"规则因果边 {len(rows)} 条")
    return rows


def smooth_grid_mix(events: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, int], np.ndarray]:
    """两层均值聚合：每个网格吸收相邻网格的罪名分布。"""
    type_ids = sorted(events["crime_type_id"].unique())
    type_index = {type_id: i for i, type_id in enumerate(type_ids)}
    grid_ids = events["grid_id"].drop_duplicates().tolist()
    grid_index = {grid_id: i for i, grid_id in enumerate(grid_ids)}
    mix = np.zeros((len(grid_ids), len(type_ids)), dtype=np.float64)
    counts = events.groupby(["grid_id", "crime_type_id"], sort=False).size()
    for (grid_id, type_id), count in counts.items():
        mix[grid_index[grid_id], type_index[type_id]] += count
    global_rate = mix.sum(axis=0)
    global_rate = global_rate / max(global_rate.sum(), 1.0)
    base = mix / np.maximum(mix.sum(axis=1, keepdims=True), 1e-12)
    neighbors = []
    occupied = {
        (str(city), int(x_index), int(y_index)): grid_index[grid_id]
        for city, x_index, y_index, grid_id in events[["city_code", "grid_x", "grid_y", "grid_id"]].drop_duplicates().itertuples(index=False)
    }
    for (city, x_index, y_index), left in occupied.items():
        for dx, dy in ((1, 0), (0, 1)):
            right = occupied.get((city, x_index + dx, y_index + dy))
            if right is not None:
                neighbors.append((left, right))

    def pass_once(features: np.ndarray) -> np.ndarray:
        acc = np.zeros_like(features)
        degree = np.zeros(len(features))
        for left, right in neighbors:
            acc[left] += features[right]
            acc[right] += features[left]
            degree[left] += 1
            degree[right] += 1
        mixed = features.copy()
        touched = degree > 0
        mixed[touched] = 0.5 * features[touched] + 0.5 * (acc[touched] / degree[touched, None])
        mixed /= np.maximum(mixed.sum(axis=1, keepdims=True), 1e-12)
        return mixed

    smoothed = pass_once(pass_once(base))
    county_grids: dict[str, list[int]] = defaultdict(list)
    for county_id, grid_id in events[["county_id", "grid_id"]].drop_duplicates().itertuples(index=False):
        county_grids[county_id].append(grid_index[grid_id])
    county_feat = {county_id: smoothed[indexes].mean(axis=0) for county_id, indexes in county_grids.items()}
    grid_feat = {grid_id: smoothed[index] for grid_id, index in grid_index.items()}
    return {**county_feat, **grid_feat}, type_index, global_rate


def rescore_causal(conn, events: pd.DataFrame, transitions: Counter) -> dict:
    _log("用两层网格邻域聚合重估因果边，并补充漏检边")
    features, type_index, global_rate = smooth_grid_mix(events)
    existing = {
        (row["entity1_id"], row["entity2_id"])
        for row in conn.execute("SELECT entity1_id, entity2_id FROM quads WHERE relation_subtype = '地理致罪'")
    }
    updated = 0
    for row in conn.execute("SELECT quad_id, entity1_id, entity2_id, confidence FROM quads WHERE relation_subtype = '地理致罪'"):
        vector = features.get(row["entity1_id"])
        type_at = type_index.get(row["entity2_id"])
        if vector is None or type_at is None:
            continue
        base = float(global_rate[type_at])
        lift = float(vector[type_at]) / base if base > 0 else 0.0
        blended = 0.5 * float(row["confidence"]) + 0.5 * _lift_confidence(lift)
        conn.execute(
            "UPDATE quads SET confidence = ?, strength = ?, source = 'rule+ml' WHERE quad_id = ?",
            (round(blended, 4), round(blended, 4), row["quad_id"]),
        )
        updated += 1
    mined = 0
    counts = events.groupby(["county_id", "crime_type_id"]).size()
    stamps = events.groupby(["county_id", "crime_type_id"])["iso8601"].agg(["min", "max"])
    spans = events.groupby(["county_id", "crime_type_id"])["unix_timestamp"].agg(["min", "max"])
    new_rows = []
    for (county_id, type_id), count in counts.items():
        if (county_id, type_id) in existing or int(count) < CAUSAL_COUNTY_MIN_COUNT:
            continue
        vector = features.get(county_id)
        type_at = type_index.get(type_id)
        if vector is None or type_at is None:
            continue
        base = float(global_rate[type_at])
        if base <= 0:
            continue
        lift = float(vector[type_at]) / base
        if lift < 2.0:
            continue
        confidence = _lift_confidence(lift)
        if confidence < INFER_MIN_CONFIDENCE:
            continue
        start = str(stamps.loc[(county_id, type_id), "min"])
        end = str(stamps.loc[(county_id, type_id), "max"])
        new_rows.append(
            _quad(
                county_id,
                "因果关联",
                "地理致罪",
                type_id,
                end,
                start,
                end,
                confidence,
                confidence,
                "ml",
                unix_ts=int(spans.loc[(county_id, type_id), "max"]),
                window_hours=(int(spans.loc[(county_id, type_id), "max"]) - int(spans.loc[(county_id, type_id), "min"])) / 3600.0,
            )
        )
        mined += 1
    outgoing = Counter()
    for (left, _right), count in transitions.items():
        outgoing[left] += count
    for row in conn.execute("SELECT quad_id, entity1_id, entity2_id, confidence FROM quads WHERE relation_subtype = '类型衍生因果'"):
        count = transitions.get((row["entity1_id"], row["entity2_id"]), 0)
        denom = outgoing.get(row["entity1_id"], 0)
        observed = count / denom if denom else 0.0
        blended = 0.5 * float(row["confidence"]) + 0.5 * min(0.99, observed)
        conn.execute(
            "UPDATE quads SET confidence = ?, strength = ?, source = 'rule+ml' WHERE quad_id = ?",
            (round(blended, 4), round(min(0.99, observed), 4), row["quad_id"]),
        )
        updated += 1
    if new_rows:
        insert_quads(conn, new_rows)
    else:
        conn.commit()
    pruned = conn.execute(
        "DELETE FROM quads WHERE relation_type = '因果关联' AND source IN ('ml', 'rule+ml') AND confidence < ?",
        (INFER_MIN_CONFIDENCE,),
    ).rowcount
    orphans = conn.execute(
        """
        DELETE FROM entities
        WHERE entity_id NOT IN (SELECT entity1_id FROM quads UNION SELECT entity2_id FROM quads)
        """
    ).rowcount
    conn.commit()
    _log(f"重估 {updated} 条，机器学习补充 {mined} 条，修剪低置信因果边 {pruned} 条，孤立实体 {orphans} 个")
    return {"rescored": updated, "mined": mined, "pruned_edges": pruned, "pruned_orphans": orphans}


def export_neo4j(conn) -> None:
    _log("导出 Neo4j 节点、关系与索引脚本")
    NEO4J_DIR.mkdir(parents=True, exist_ok=True)
    crime_path = NEO4J_DIR / "nodes_crime_type.csv"
    geo_path = NEO4J_DIR / "nodes_geo.csv"
    time_path = NEO4J_DIR / "nodes_time.csv"
    with crime_path.open("w", encoding="utf-8", newline="") as crime_file, geo_path.open(
        "w", encoding="utf-8", newline=""
    ) as geo_file, time_path.open("w", encoding="utf-8", newline="") as time_file:
        crime_writer = csv.writer(crime_file)
        geo_writer = csv.writer(geo_file)
        time_writer = csv.writer(time_file)
        crime_writer.writerow(["entity_id", "name", "standard_code", "level", "article", "elements", "subtypes", "event_count", "description"])
        geo_writer.writerow(
            [
                "entity_id",
                "name",
                "standard_code",
                "level",
                "event_count",
                "crime_density_per_km2",
                "lat_min",
                "lat_max",
                "lon_min",
                "lon_max",
                "proj_xmin",
                "proj_xmax",
                "proj_ymin",
                "proj_ymax",
                "description",
            ]
        )
        time_writer.writerow(
            ["entity_id", "name", "standard_code", "granularity", "period", "is_weekend", "is_holiday", "holiday_name", "duration_hours", "description"]
        )
        for row in conn.execute("SELECT * FROM entities"):
            attrs = json.loads(row["attrs_json"] or "{}")
            if row["entity_class"] == "犯罪类型":
                crime_writer.writerow(
                    [
                        row["entity_id"],
                        row["name"],
                        row["standard_code"],
                        row["level"],
                        attrs.get("article", ""),
                        "、".join(attrs.get("elements") or []),
                        "、".join(attrs.get("subtypes") or []),
                        attrs.get("event_count", 0),
                        row["description"],
                    ]
                )
            elif row["entity_class"] == "地理空间":
                geo_writer.writerow(
                    [
                        row["entity_id"],
                        row["name"],
                        row["standard_code"],
                        row["level"],
                        attrs.get("event_count", 0),
                        attrs.get("crime_density_per_km2", ""),
                        attrs.get("lat_min", ""),
                        attrs.get("lat_max", ""),
                        attrs.get("lon_min", ""),
                        attrs.get("lon_max", ""),
                        attrs.get("proj_xmin", ""),
                        attrs.get("proj_xmax", ""),
                        attrs.get("proj_ymin", ""),
                        attrs.get("proj_ymax", ""),
                        row["description"],
                    ]
                )
            else:
                time_writer.writerow(
                    [
                        row["entity_id"],
                        row["name"],
                        row["standard_code"],
                        attrs.get("granularity", row["level"]),
                        attrs.get("period", ""),
                        "true" if attrs.get("is_weekend") else "false",
                        "true" if attrs.get("is_holiday") else "false",
                        attrs.get("holiday_name", ""),
                        "" if attrs.get("duration_hours") is None else attrs.get("duration_hours"),
                        row["description"],
                    ]
                )
    rel_header = [
        "start_id",
        "end_id",
        "iso8601_timestamp",
        "valid_from",
        "valid_to",
        "confidence",
        "strength",
        "window_hours",
        "source",
        "case_number",
        "other_case",
        "relation_type",
        "relation_subtype",
    ]
    present = [row[0] for row in conn.execute("SELECT DISTINCT relation_subtype FROM quads")]
    for subtype in present:
        rel_type = RELATION_SPECS[( _relation_of(conn, subtype), subtype)]
        path = NEO4J_DIR / f"rels_{rel_type}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(rel_header)
            cursor = conn.execute(
                """
                SELECT entity1_id, entity2_id, iso8601_timestamp, valid_from, valid_to,
                       confidence, strength, window_hours, source, case_number, other_case,
                       relation_type, relation_subtype
                FROM quads WHERE relation_subtype = ?
                """,
                (subtype,),
            )
            while True:
                chunk = cursor.fetchmany(50000)
                if not chunk:
                    break
                writer.writerows(chunk)
    cypher = ["// 将本目录 csv 复制到 Neo4j import 后执行：cypher-shell -f import.cypher", ""]
    cypher.append("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.entity_id IS UNIQUE;")
    cypher.append("CREATE INDEX entity_class IF NOT EXISTS FOR (n:Entity) ON (n.level);")
    cypher.append("CREATE INDEX crime_code IF NOT EXISTS FOR (n:CrimeType) ON (n.standard_code);")
    cypher.append("CREATE INDEX geo_level IF NOT EXISTS FOR (n:GeoRegion) ON (n.level);")
    cypher.append("CREATE INDEX time_grain IF NOT EXISTS FOR (n:TimeUnit) ON (n.granularity);")
    cypher.append(_load_nodes("nodes_crime_type.csv", "CrimeType", ["name", "standard_code", "level", "article", "elements", "subtypes", "description"], ["event_count"]))
    cypher.append(
        _load_nodes(
            "nodes_geo.csv",
            "GeoRegion",
            ["name", "standard_code", "level", "description"],
            ["event_count"],
            ["crime_density_per_km2", "lat_min", "lat_max", "lon_min", "lon_max", "proj_xmin", "proj_xmax", "proj_ymin", "proj_ymax"],
        )
    )
    cypher.append(
        _load_nodes(
            "nodes_time.csv",
            "TimeUnit",
            ["name", "standard_code", "granularity", "period", "holiday_name", "description"],
            [],
            ["duration_hours"],
            bool_fields=["is_weekend", "is_holiday"],
        )
    )
    for subtype in present:
        relation = _relation_of(conn, subtype)
        rel_type = RELATION_SPECS[(relation, subtype)]
        cypher.append(
            f"""
LOAD CSV WITH HEADERS FROM 'file:///rels_{rel_type}.csv' AS row
CALL {{
  WITH row
  MATCH (a:Entity {{entity_id: row.start_id}})
  MATCH (b:Entity {{entity_id: row.end_id}})
  CREATE (a)-[:{rel_type} {{
    relation_type: row.relation_type,
    relation_subtype: row.relation_subtype,
    iso8601_timestamp: row.iso8601_timestamp,
    valid_from: row.valid_from,
    valid_to: row.valid_to,
    confidence: toFloat(row.confidence),
    strength: toFloat(row.strength),
    window_hours: toFloat(row.window_hours),
    source: row.source,
    case_number: row.case_number,
    other_case: row.other_case
  }}]->(b)
}} IN TRANSACTIONS OF 5000 ROWS;
""".strip()
        )
        cypher.append(f"CREATE INDEX {rel_type.lower()}_time IF NOT EXISTS FOR ()-[r:{rel_type}]-() ON (r.iso8601_timestamp);")
    (NEO4J_DIR / "import.cypher").write_text("\n".join(cypher) + "\n", encoding="utf-8")


def _relation_of(conn, subtype: str) -> str:
    row = conn.execute("SELECT relation_type FROM quads WHERE relation_subtype = ? LIMIT 1", (subtype,)).fetchone()
    return row["relation_type"]


def _load_nodes(filename: str, label: str, texts: list[str], integers: list[str], floats: list[str] | None = None, bool_fields: list[str] | None = None) -> str:
    floats = floats or []
    bool_fields = bool_fields or []
    fields = ["entity_id: row.entity_id"]
    for name in texts:
        fields.append(f"{name}: row.{name}")
    for name in integers:
        fields.append(f"{name}: toInteger(row.{name})")
    for name in floats:
        fields.append(f"{name}: CASE row.{name} WHEN '' THEN null ELSE toFloat(row.{name}) END")
    for name in bool_fields:
        fields.append(f"{name}: row.{name} = 'true'")
    body = ",\n    ".join(fields)
    return f"""
LOAD CSV WITH HEADERS FROM 'file:///{filename}' AS row
CALL {{
  WITH row
  CREATE (:Entity:{label} {{
    {body}
  }})
}} IN TRANSACTIONS OF 5000 ROWS;
""".strip()


def summarize(conn) -> dict:
    entities = {row["entity_class"]: row["n"] for row in conn.execute("SELECT entity_class, COUNT(*) AS n FROM entities GROUP BY 1")}
    levels = {row["level"]: row["n"] for row in conn.execute("SELECT level, COUNT(*) AS n FROM entities GROUP BY 1")}
    relations = {
        f"{row['relation_type']}/{row['relation_subtype']}": row["n"]
        for row in conn.execute("SELECT relation_type, relation_subtype, COUNT(*) AS n FROM quads GROUP BY 1, 2 ORDER BY n DESC")
    }
    sources = {row["source"]: row["n"] for row in conn.execute("SELECT source, COUNT(*) AS n FROM quads GROUP BY 1")}
    return {
        "entities": int(sum(entities.values())),
        "entity_class_counts": entities,
        "entity_level_counts": levels,
        "quadruples": int(conn.execute("SELECT COUNT(*) FROM quads").fetchone()[0]),
        "relation_counts": relations,
        "source_counts": sources,
    }


def main() -> None:
    started = datetime.now(timezone.utc)
    _self_check()
    events = load_events()
    grids = _inverse_bounds(_grid_rectangles(events))
    dictionary = CrimeTypeDictionary()
    t_min = str(events["iso8601"].min())
    t_max = str(events["iso8601"].max())
    unix_min = int(events["unix_timestamp"].min())
    unix_max = int(events["unix_timestamp"].max())
    conn = reset_database()
    insert_entities(conn, build_entities(events, grids, dictionary))
    conn.commit()
    structural = structural_edges(events, dictionary, t_min, t_max, unix_min, unix_max)
    _log(f"写入结构关系 {len(structural)} 条")
    insert_quads(conn, structural)
    transitions = event_edges(conn, events)
    insert_quads(conn, causal_edges(events, transitions))
    ml_stats = rescore_causal(conn, events, transitions)
    export_neo4j(conn)
    report = {
        "built_at": started.isoformat(),
        "model": {
            "entity_classes": ["犯罪类型", "地理空间", "时间单元"],
            "time_granularity": ["小时", "日", "周", "旬", "月"],
            "alignment": "时间实体取能包住不确定区间的最细粒度；24小时相继只保留区间上必定成立的边；超出12位网格编码的点不并入边界格",
            "relation_types": ["时空关联", "因果关联", "类型关联"],
            "ml": "两层网格邻域均值聚合，用于重估地理致罪边并补充漏检边",
            "prune": f"因果边在重估后置信度低于 {INFER_MIN_CONFIDENCE} 的予以删除",
            "population_density": "源数据没有人口分母，网格上保存的是每平方公里案件密度，不填人口数",
        },
        **summarize(conn),
        **ml_stats,
        "neo4j_dir": str(NEO4J_DIR),
        "sqlite": str(TKG_DIR / "crime_tkg.sqlite"),
    }
    dump_json(TKG_DIR / "tkg_report.json", report)
    _log(json.dumps(report, ensure_ascii=False, indent=2))
    conn.close()


if __name__ == "__main__":
    main()
