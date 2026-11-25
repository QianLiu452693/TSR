"""清洗、时空对齐、实体关系抽取，输出可导入 Neo4j 的 TKG 四元组。"""

from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd

from tkg_pipeline.admin_geo import assign_admin, load_county_gdf
from tkg_pipeline.align import align_time, certain_successor_indices, score_spatial, to_iso, to_unix
from tkg_pipeline.config import (
    CAUSAL_COUNTY_MIN_COUNT,
    CAUSAL_GRID_MIN_COUNT,
    CAUSAL_GRID_MIN_LIFT,
    CAUSAL_MIN_LIFT,
    CAUSAL_TYPE_MIN_COUNT,
    DEDUP_DISTANCE_M,
    DEDUP_JACCARD,
    DEDUP_TIME_SECONDS,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    LOW_CONFIDENCE,
    OUTPUT_DIR,
    RANDOM_SEED,
    REVIEW_FRACTION,
    RULE_CONFIDENCE,
    SOURCE_COLUMNS,
    SOURCE_CSV,
    SUCCESSION_CONFIDENCE,
    SUCCESSION_SECONDS,
    TIME_END,
    TIME_START,
    TYPE_ACCEPT_SCORE,
)
from tkg_pipeline.crime_types import GENERIC_CASE_TYPES, CrimeTypeDictionary
from tkg_pipeline.spatial import attach_grids, dbscan_keep_mask
from tkg_pipeline.time_norm import parse_raw_time

ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00$")


def _log(message: str) -> None:
    print(message, flush=True)


def load_records() -> pd.DataFrame:
    _log(f"读取 {SOURCE_CSV.name}（不载入裁判文书全文）")
    parts = []
    for chunk in pd.read_csv(
        SOURCE_CSV,
        usecols=SOURCE_COLUMNS,
        chunksize=200_000,
        dtype="string",
        low_memory=False,
    ):
        parts.append(chunk)
        _log(f"  已读 {sum(len(p) for p in parts)} 行")
    df = pd.concat(parts, ignore_index=True)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df


def _case_key(series: pd.Series) -> pd.Series:
    key = series.fillna("").astype(str).str.strip()
    key = key.str.replace("（", "(", regex=False).str.replace("）", ")", regex=False)
    key = key.str.replace(" ", "", regex=False).str.replace("\u3000", "", regex=False)
    key = key.replace({"nan": "", "None": "", "无": "", "NaN": ""})
    return key


def _completeness(df: pd.DataFrame) -> pd.Series:
    details_len = df["details"].fillna("").astype(str).str.len().clip(upper=400)
    case_type = df["case_type"].fillna("").astype(str).str.strip()
    specific = ~case_type.isin(GENERIC_CASE_TYPES) & ~case_type.str.startswith("刑事")
    formatted = df["formatted_datetime"].fillna("").astype(str).str.strip().ne("")
    incident = df["incident_time"].fillna("").astype(str).str.strip().ne("")
    location = df["incident_location"].fillna("").astype(str).str.strip().ne("")
    coords = df["latitude"].notna() & df["longitude"].notna()
    return details_len + specific.astype(int) * 80 + formatted.astype(int) * 30 + incident.astype(int) * 20 + location.astype(int) * 10 + coords.astype(int) * 10


def _char_jaccard(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["case_key"] = _case_key(df["case_number"])
    df["complete"] = _completeness(df)
    before = len(df)
    numbered = df[df["case_key"].ne("")].sort_values("complete", ascending=False)
    numbered = numbered.drop_duplicates("case_key", keep="first")
    unnumbered = df[df["case_key"].eq("")].copy()
    id_removed = before - len(numbered) - len(unnumbered)
    fuzzy_removed = 0
    if len(unnumbered):
        unnumbered["bucket_x"] = (unnumbered["longitude"] * 200).round()
        unnumbered["bucket_y"] = (unnumbered["latitude"] * 200).round()
        unnumbered["day_key"] = unnumbered["incident_time"].fillna("").astype(str).str.slice(0, 12)
        unnumbered["kw"] = (
            unnumbered["case_type"].fillna("").astype(str) + unnumbered["incident_location"].fillna("").astype(str).str.slice(0, 16)
        )
        keep_idx = []
        for _, grp in unnumbered.groupby(["bucket_x", "bucket_y", "day_key"], dropna=False, sort=False):
            grp = grp.sort_values("complete", ascending=False)
            chosen: list[tuple[float, float, float, str]] = []
            for row in grp.itertuples():
                duplicate = False
                for prev_lat, prev_lon, prev_score_unused, prev_kw in chosen:
                    if pd.isna(row.latitude) or pd.isna(row.longitude) or pd.isna(prev_lat):
                        distance = 0.0
                    else:
                        distance = _equirectangular_m(row.latitude, row.longitude, prev_lat, prev_lon)
                    if distance <= DEDUP_DISTANCE_M and _char_jaccard(str(row.kw), prev_kw) >= DEDUP_JACCARD:
                        duplicate = True
                        break
                if duplicate:
                    fuzzy_removed += 1
                    continue
                chosen.append((row.latitude, row.longitude, row.complete, str(row.kw)))
                keep_idx.append(row.Index)
        unnumbered = unnumbered.loc[keep_idx]
    out = pd.concat([numbered, unnumbered], ignore_index=True)
    stats = {
        "input_rows": int(before),
        "duplicate_case_id_removed": int(id_removed),
        "duplicate_fuzzy_removed": int(fuzzy_removed),
        "rows_after_dedup": int(len(out)),
    }
    _log(f"去重：案件编号剔除 {id_removed}，无编号三重匹配剔除 {fuzzy_removed}，剩余 {len(out)}")
    return out, stats


def _equirectangular_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat0 = math.radians((lat1 + lat2) / 2.0)
    dx = math.radians(lon1 - lon2) * math.cos(lat0) * 6_371_000.0
    dy = math.radians(lat1 - lat2) * 6_371_000.0
    return math.hypot(dx, dy)


def normalize_time(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    _log("时间解析、补全，并按不确定区间对齐到时间实体")
    start = datetime.fromisoformat(TIME_START)
    end = datetime.fromisoformat(TIME_END)
    rows: list[dict] = []
    keep: list[bool] = []
    outside = 0
    failed = 0
    incidents = df["incident_time"].tolist()
    formatted = df["formatted_datetime"].tolist()
    for i, (incident, formatted_value) in enumerate(zip(incidents, formatted)):
        aligned = align_time(incident, formatted_value)
        if aligned is None or aligned.anchor < start or aligned.anchor > end:
            if aligned is None:
                failed += 1
            else:
                outside += 1
            rows.append({})
            keep.append(False)
        else:
            keep.append(True)
            rows.append(
                {
                    "iso8601": to_iso(aligned.anchor),
                    "unix_timestamp": to_unix(aligned.anchor),
                    "time_precision": aligned.grain,
                    "time_grain": aligned.grain,
                    "time_imputed": int(aligned.imputed),
                    "time_align_confidence": aligned.confidence,
                    "time_source": aligned.source,
                    "time_start_iso": to_iso(aligned.start),
                    "time_end_iso": to_iso(aligned.end),
                    "time_start_unix": to_unix(aligned.start),
                    "time_end_unix": to_unix(aligned.end),
                    "time_entity_id": aligned.entity_id,
                    "time_entity_level": aligned.entity_level,
                }
            )
        if i and i % 200_000 == 0:
            _log(f"  时间已处理 {i}")
    aligned_frame = pd.DataFrame(rows, index=df.index)
    before = len(df)
    df = pd.concat([df, aligned_frame], axis=1)
    df = df.loc[keep].reset_index(drop=True)
    source_counts = df["time_source"].value_counts().astype(int).to_dict() if len(df) else {}
    stats = {
        "time_unparsed_removed": int(failed),
        "time_outside_window_removed": int(outside),
        "rows_after_time": int(len(df)),
        "rows_entering_time": int(before),
        "time_source_counts": source_counts,
        "time_without_entity": int(df["time_entity_id"].fillna("").eq("").sum()) if len(df) else 0,
    }
    _log(
        f"时间对齐：无法解析 {failed}，超出 2000–2019 {outside}，剩余 {len(df)}，"
        f"无单一时间实体 {stats['time_without_entity']}"
    )
    return df, stats


def filter_coordinates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mask = (
        df["latitude"].between(LAT_MIN, LAT_MAX)
        & df["longitude"].between(LON_MIN, LON_MAX)
        & ~((df["latitude"] == 0) & (df["longitude"] == 0))
    )
    removed = int((~mask).sum())
    df = df.loc[mask].reset_index(drop=True)
    _log(f"疆域规则剔除 {removed}，剩余 {len(df)}")
    return df, {"coordinate_out_of_china_removed": removed, "rows_after_coordinates": int(len(df))}


def classify_crimes(df: pd.DataFrame, dictionary: CrimeTypeDictionary) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    _log("犯罪类型标准化与语义补全")
    case_types = df["case_type"].fillna("").astype(str).tolist()
    details = df["details"].fillna("").astype(str).tolist()
    codes, names, scores, methods = [], [], [], []
    for i, (case_type, detail) in enumerate(zip(case_types, details)):
        hit = dictionary.classify(case_type, detail)
        if hit is None:
            codes.append(None)
            names.append(None)
            scores.append(0.0)
            methods.append("unmatched")
        else:
            codes.append(hit.code)
            names.append(hit.name)
            scores.append(round(float(hit.score), 4))
            methods.append(hit.method)
        if i and i % 200_000 == 0:
            _log(f"  罪名已处理 {i}")
    df = df.copy()
    df["crime_type_code"] = codes
    df["crime_type_name"] = names
    df["type_match_score"] = scores
    df["type_match_method"] = methods
    review_mask = df["type_match_method"].isin(["unmatched", "tfidf"]) & (df["type_match_score"] < TYPE_ACCEPT_SCORE)
    review_mask = review_mask | (df["crime_type_code"].isna() & ~df["type_match_method"].isin(["blacklist", "civil"]))
    review = df.loc[review_mask, ["case_number", "case_type", "crime_type_name", "type_match_score", "type_match_method"]].copy()
    review["review_reason"] = "罪名匹配相似度低于阈值或未能匹配，需人工复核"
    out_of_scope = df["crime_type_code"].eq("OUT") | df["type_match_method"].isin(["blacklist", "civil"])
    accepted = (~out_of_scope) & (~review_mask) & df["crime_type_code"].notna()
    stats = {
        "crime_out_of_scope_removed": int(out_of_scope.sum()),
        "crime_low_score_to_review": int(review_mask.sum()),
        "rows_after_crime_type": int(accepted.sum()),
        "crime_type_method_counts": df.loc[accepted, "type_match_method"].value_counts().astype(int).to_dict(),
    }
    df = df.loc[accepted].drop(columns=["details"]).reset_index(drop=True)
    df["crime_type_id"] = "C" + df["crime_type_code"].astype(str)
    _log(
        f"罪名过滤：无关类型 {stats['crime_out_of_scope_removed']}，低相似度待复核 {stats['crime_low_score_to_review']}，剩余 {len(df)}"
    )
    return df, review, stats


def drop_spatial_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    _log("DBSCAN 识别地理孤立点")
    keep = dbscan_keep_mask(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    removed = int((~keep).sum())
    df = df.loc[keep].reset_index(drop=True)
    _log(f"孤立点剔除 {removed}，剩余 {len(df)}")
    return df, {"dbscan_outliers_removed": removed, "rows_after_dbscan": int(len(df))}


def _entity_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).drop_duplicates("entity_id")
    return frame.sort_values("entity_id").reset_index(drop=True)


def build_entities(df: pd.DataFrame, dictionary: CrimeTypeDictionary) -> pd.DataFrame:
    rows: list[dict] = []
    present = set(df["crime_type_code"].astype(str))
    for code, spec in dictionary.by_code.items():
        if code not in present:
            continue
        rows.append(
            {
                "entity_id": f"C{code}",
                "entity_class": "犯罪类型",
                "standard_code": code,
                "name": spec["name"],
                "level": spec["chapter"],
                "description": f"{spec['chapter']} / {spec['name']}；构成要件：{'、'.join(spec['elements'])}",
            }
        )
    geo_levels = [
        ("province_code", "province_name", "GP", "省级"),
        ("city_code", "city_name", "GC", "地级"),
        ("admin_code", "admin_name", "GD", "县级"),
    ]
    for code_col, name_col, prefix, level in geo_levels:
        table = df[[code_col, name_col]].dropna().drop_duplicates()
        for rec in table.itertuples(index=False):
            code = str(getattr(rec, code_col))
            name = str(getattr(rec, name_col))
            rows.append(
                {
                    "entity_id": prefix + code,
                    "entity_class": "地理空间",
                    "standard_code": code,
                    "name": name,
                    "level": level,
                    "description": f"{level}地理单元 {name}（{code}）",
                }
            )
    grids = df.loc[df["grid_id"].notna(), ["grid_id", "grid_code", "city_name"]].drop_duplicates()
    for rec in grids.itertuples(index=False):
        rows.append(
            {
                "entity_id": rec.grid_id,
                "entity_class": "地理空间",
                "standard_code": rec.grid_code,
                "name": f"{rec.city_name}{rec.grid_code[-6:]}网格",
                "level": "城市网格",
                "description": f"500米网格 {rec.grid_code}，编码规则为地级代码6位+X索引3位+Y索引3位",
            }
        )
    times = df[["time_entity_id", "time_entity_level", "iso8601", "time_grain"]].copy()
    times = times[times["time_entity_id"].fillna("").astype(str).str.len().gt(0)].drop_duplicates("time_entity_id")
    for rec in times.itertuples(index=False):
        rows.append(
            {
                "entity_id": rec.time_entity_id,
                "entity_class": "时间单元",
                "standard_code": str(rec.time_entity_id)[1:],
                "name": str(rec.iso8601)[:13] if rec.time_entity_level == "小时" else str(rec.iso8601)[:10],
                "level": rec.time_entity_level,
                "description": f"{rec.time_entity_level}级时间单元，由{rec.time_grain}粒度的区间对齐得到",
            }
        )
    return _entity_frame(rows)


def _quads(entity1, relation, entity2, stamp, confidence) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "entity1_id": entity1,
            "relation_type": relation,
            "entity2_id": entity2,
            "iso8601_timestamp": stamp,
            "confidence": confidence,
        }
    )
    return frame


def build_quadruples(df: pd.DataFrame, dictionary: CrimeTypeDictionary) -> pd.DataFrame:
    _log("抽取实体关系并生成四元组")
    work = df.copy()
    work["county_id"] = "GD" + work["admin_code"].astype(str)
    work["city_id"] = "GC" + work["city_code"].astype(str)
    work["province_id"] = "GP" + work["province_code"].astype(str)
    work["time_id"] = work["time_entity_id"].fillna("").astype(str)
    reference_time = work["iso8601"].sort_values().iloc[len(work) // 2]
    has_time = work["time_id"].str.len().gt(0)
    has_grid = work["grid_id"].notna() & work["grid_id"].astype(str).str.len().gt(0)
    placed = work.loc[has_grid]
    timed = work.loc[has_time]

    occurrence_place = _quads(placed["crime_type_id"], "时空关联", placed["grid_id"], placed["iso8601"], placed["spatial_align_confidence"])
    occurrence_time = _quads(timed["crime_type_id"], "时空关联", timed["time_id"], timed["iso8601"], timed["time_align_confidence"])

    ordered = placed.sort_values(["grid_code", "time_start_unix"])
    same_grid = ordered["grid_code"].eq(ordered["grid_code"].shift(-1))
    point_delta = ordered["unix_timestamp"].shift(-1) - ordered["unix_timestamp"]
    naive_succession = int((same_grid & point_delta.gt(0) & point_delta.le(SUCCESSION_SECONDS)).sum())
    left_ids: list[str] = []
    right_ids: list[str] = []
    stamps: list[str] = []
    confs: list[float] = []
    for _, grp in ordered.groupby("grid_code", sort=False):
        if len(grp) < 2:
            continue
        ii, jj = certain_successor_indices(
            grp["time_start_unix"].to_numpy(dtype=np.int64),
            grp["time_end_unix"].to_numpy(dtype=np.int64),
            SUCCESSION_SECONDS,
        )
        if len(ii) == 0:
            continue
        types = grp["crime_type_id"].to_numpy()
        score = grp["time_align_confidence"].to_numpy(dtype=float)
        left_ids.extend(types[ii].tolist())
        right_ids.extend(types[jj].tolist())
        stamps.extend(grp["iso8601"].to_numpy()[jj].tolist())
        confs.extend(np.minimum(np.minimum(score[ii], score[jj]), SUCCESSION_CONFIDENCE).round(4).tolist())
    succession = _quads(left_ids, "时空关联", right_ids, stamps, confs)
    successive = pd.DataFrame({"crime_type_id": left_ids, "next_type": right_ids})

    grid_first = placed.groupby("grid_id", sort=False)["iso8601"].min()
    hierarchy_rows = []
    grid_info = placed.drop_duplicates("grid_id")
    for rec in grid_info.itertuples(index=False):
        stamp = grid_first.loc[rec.grid_id]
        hierarchy_rows.extend(
            [
                (rec.grid_id, "空间隶属", rec.county_id, stamp, RULE_CONFIDENCE),
                (rec.county_id, "空间隶属", rec.city_id, stamp, RULE_CONFIDENCE),
                (rec.city_id, "空间隶属", rec.province_id, stamp, RULE_CONFIDENCE),
            ]
        )
    hierarchy = pd.DataFrame(hierarchy_rows, columns=["entity1_id", "relation_type", "entity2_id", "iso8601_timestamp", "confidence"])

    occupied = placed.dropna(subset=["grid_x", "grid_y"]).drop_duplicates(["city_code", "grid_x", "grid_y"])[
        ["city_code", "grid_x", "grid_y", "grid_id"]
    ]
    lookup = {
        (str(r.city_code), int(r.grid_x), int(r.grid_y)): r.grid_id for r in occupied.itertuples(index=False)
    }
    adjacent_rows = []
    for (city, x_index, y_index), grid_id in lookup.items():
        for dx, dy in ((1, 0), (0, 1)):
            other = lookup.get((city, x_index + dx, y_index + dy))
            if other:
                adjacent_rows.append((grid_id, "空间相邻", other, reference_time, RULE_CONFIDENCE))
    adjacent = pd.DataFrame(adjacent_rows, columns=["entity1_id", "relation_type", "entity2_id", "iso8601_timestamp", "confidence"])

    type_rows = []
    for pair in dictionary.jaccard_pairs():
        type_rows.append(
            (
                "C" + pair["code_a"],
                pair["relation_type"],
                "C" + pair["code_b"],
                reference_time,
                pair["confidence"],
            )
        )
    type_rel = pd.DataFrame(type_rows, columns=["entity1_id", "relation_type", "entity2_id", "iso8601_timestamp", "confidence"])

    causal_geo = _causal_place(work, reference_time)
    causal_type = _causal_types(successive, reference_time)

    quads = pd.concat(
        [occurrence_place, occurrence_time, succession, hierarchy, adjacent, type_rel, causal_geo, causal_type],
        ignore_index=True,
    )
    quads["confidence"] = pd.to_numeric(quads["confidence"], errors="coerce").round(4)
    quads = quads.drop_duplicates(
        ["entity1_id", "relation_type", "entity2_id", "iso8601_timestamp", "confidence"]
    ).reset_index(drop=True)
    quads.attrs["succession_point_candidates"] = naive_succession
    quads.attrs["succession_interval_kept"] = int(len(succession))
    _log(
        f"四元组 {len(quads)} 条；点时间相继 {naive_succession} 条，区间判定保留 {len(succession)} 条"
    )
    return quads


def _lift_confidence(lift: float) -> float:
    if lift <= 1.0:
        return 0.0
    return float(min(0.99, 1.0 - 1.0 / lift))


def _causal_place(work: pd.DataFrame, reference_time: str) -> pd.DataFrame:
    rows = []
    base = work["crime_type_id"].value_counts(normalize=True)
    for geo_col, min_count, min_lift in (
        ("county_id", CAUSAL_COUNTY_MIN_COUNT, CAUSAL_MIN_LIFT),
        ("grid_id", CAUSAL_GRID_MIN_COUNT, CAUSAL_GRID_MIN_LIFT),
    ):
        counts = work.groupby([geo_col, "crime_type_id"], sort=False).size()
        totals = work.groupby(geo_col, sort=False).size()
        for (geo_id, crime_id), count in counts.items():
            total = int(totals.loc[geo_id])
            if int(count) < min_count or total < min_count:
                continue
            rate = float(base.get(crime_id, 0.0))
            if rate <= 0:
                continue
            lift = (count / total) / rate
            if lift < min_lift:
                continue
            rows.append((geo_id, "因果关联", crime_id, reference_time, round(_lift_confidence(lift), 4)))
    if not rows:
        return pd.DataFrame(columns=["entity1_id", "relation_type", "entity2_id", "iso8601_timestamp", "confidence"])
    return pd.DataFrame(rows, columns=["entity1_id", "relation_type", "entity2_id", "iso8601_timestamp", "confidence"])


def _causal_types(successive: pd.DataFrame, reference_time: str) -> pd.DataFrame:
    if successive.empty:
        return pd.DataFrame(columns=["entity1_id", "relation_type", "entity2_id", "iso8601_timestamp", "confidence"])
    pair_counts = successive.groupby(["crime_type_id", "next_type"], sort=False).size()
    from_counts = successive.groupby("crime_type_id", sort=False).size()
    base = successive["next_type"].value_counts(normalize=True)
    rows = []
    for (left, right), count in pair_counts.items():
        if left == right or int(count) < CAUSAL_TYPE_MIN_COUNT:
            continue
        reverse = int(pair_counts.get((right, left), 0))
        if int(count) < reverse * 1.2:
            continue
        rate = float(base.get(right, 0.0))
        denom = int(from_counts.loc[left])
        if rate <= 0 or denom == 0:
            continue
        lift = (count / denom) / rate
        if lift < CAUSAL_MIN_LIFT:
            continue
        rows.append((left, "因果关联", right, reference_time, round(_lift_confidence(lift), 4)))
    if not rows:
        return pd.DataFrame(columns=["entity1_id", "relation_type", "entity2_id", "iso8601_timestamp", "confidence"])
    return pd.DataFrame(rows, columns=["entity1_id", "relation_type", "entity2_id", "iso8601_timestamp", "confidence"])


def validate_quadruples(quads: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    complete = (
        quads["entity1_id"].notna()
        & quads["entity2_id"].notna()
        & quads["relation_type"].notna()
        & quads["iso8601_timestamp"].map(lambda value: bool(ISO_RE.match(str(value))))
        & quads["confidence"].between(0, 1)
    )
    incomplete_removed = int((~complete).sum())
    quads = quads.loc[complete].copy()
    # 类型关系已由杰卡德阈值筛选，置信度保留计算结果；0.3 只截断因果推断中的弱边。
    structural = quads["relation_type"].isin(["类型相似", "类型衍生", "类型互斥", "时空关联", "空间隶属", "空间相邻"])
    low_mask = (quads["confidence"] < LOW_CONFIDENCE) & ~structural
    low_removed = int(low_mask.sum())
    quads = quads.loc[~low_mask].reset_index(drop=True)
    sample_n = max(1, int(round(len(quads) * REVIEW_FRACTION)))
    sample = quads.sample(n=min(sample_n, len(quads)), random_state=RANDOM_SEED).copy()

    def audit(row: pd.Series) -> str:
        left, right, relation = str(row.entity1_id), str(row.entity2_id), str(row.relation_type)
        if relation == "时空关联":
            ok = left.startswith("C") and (right.startswith("GG") or right.startswith("T") or right.startswith("C"))
        elif relation in {"类型相似", "类型衍生", "类型互斥", "因果关联"} and left.startswith("C") and right.startswith("C"):
            ok = True
        elif relation == "因果关联" and right.startswith("C") and left.startswith("G"):
            ok = True
        elif relation in {"空间隶属", "空间相邻"} and left.startswith("G") and right.startswith("G"):
            ok = True
        else:
            ok = False
        return "pass" if ok else "fail"

    sample["auto_check"] = sample.apply(audit, axis=1)
    failed = sample.index[sample["auto_check"].eq("fail")]
    quads = quads.drop(index=failed).reset_index(drop=True)
    stats = {
        "quadruples_incomplete_removed": incomplete_removed,
        "quadruples_low_confidence_removed": low_removed,
        "quadruples": int(len(quads)),
        "review_sample": int(len(sample)),
        "review_sample_failed": int((sample["auto_check"] == "fail").sum()),
        "relation_counts": quads["relation_type"].value_counts().astype(int).to_dict(),
    }
    _log(
        f"四元组校验后 {len(quads)} 条；1% 复核样本 {len(sample)}，自动逻辑未通过 {stats['review_sample_failed']}"
    )
    return quads, sample, stats


def cleaned_view(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "case_number",
        "crime_type_code",
        "crime_type_name",
        "crime_type_id",
        "type_match_method",
        "type_match_score",
        "latitude",
        "longitude",
        "proj_x",
        "proj_y",
        "gk_lon0",
        "province_code",
        "province_name",
        "city_code",
        "city_name",
        "admin_code",
        "admin_name",
        "admin_match",
        "grid_code",
        "grid_id",
        "grid_x",
        "grid_y",
        "iso8601",
        "unix_timestamp",
        "time_precision",
        "time_grain",
        "time_imputed",
        "time_align_confidence",
        "time_source",
        "time_start_iso",
        "time_end_iso",
        "time_start_unix",
        "time_end_unix",
        "time_entity_id",
        "time_entity_level",
        "spatial_align_confidence",
        "place_agreement",
        "grid_align_status",
        "admin_distance_m",
        "incident_time",
        "formatted_datetime",
        "incident_location",
        "incident_province",
        "incident_city",
        "incident_county",
        "city",
        "court_name",
    ]
    view = df[columns].rename(columns={"city": "court_city", "time_entity_id": "time_id"})
    return view


def self_check(dictionary: CrimeTypeDictionary) -> None:
    samples = {
        "2017年8月20日17时许": "2017-08-20T17:00:00+08:00",
        "2018年3月28日下午": "2018-03-28T15:00:00+08:00",
        "2018年7月10日下午1时30分许": "2018-07-10T13:30:00+08:00",
        "2018年2月16日16时40分": "2018-02-16T16:40:00+08:00",
        "2017-08-20 17:00:00": "2017-08-20T17:00:00+08:00",
    }
    for text, expected in samples.items():
        parsed = parse_raw_time(text, None)
        assert parsed is not None, text
        assert to_iso(parsed[0]) == expected, (text, to_iso(parsed[0]), expected)
    ten = parse_raw_time("2017年12月上旬某日下午", None)
    assert ten is not None and to_iso(ten[0]) == "2017-12-05T15:00:00+08:00"
    theft = dictionary.classify("盗窃罪", "")
    assert theft is not None and theft.code == "0101"
    generic = dictionary.classify("刑事", "被告人犯盗窃罪，判处有期徒刑六个月")
    assert generic is not None and generic.code == "0101"
    drug = dictionary.classify("贩卖毒品罪", "")
    assert drug is not None and drug.code == "OUT"
    civil = dictionary.classify("物业服务合同纠纷", "")
    assert civil is not None and civil.code == "OUT"
    _log("解析与罪名词典自检通过")


def main() -> None:
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dictionary = CrimeTypeDictionary()
    self_check(dictionary)

    counties = load_county_gdf()
    df = load_records()
    input_rows = int(len(df))
    df, dedup_stats = deduplicate(df)
    df, time_stats = normalize_time(df)
    df, coord_stats = filter_coordinates(df)
    df, review_types, type_stats = classify_crimes(df, dictionary)
    df, dbscan_stats = drop_spatial_outliers(df)
    df = assign_admin(df, counties)
    df = score_spatial(df)
    admin_missing = int(df["admin_code"].isna().sum())
    df = df[df["admin_code"].notna() & df["city_code"].notna()].reset_index(drop=True)
    for col in ["admin_code", "city_code", "province_code"]:
        df[col] = df[col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    df, grid_meta, grid_stats = attach_grids(df, counties)

    entities = build_entities(df, dictionary)
    quads = build_quadruples(df, dictionary)
    succession_point_candidates = int(quads.attrs.get("succession_point_candidates", 0))
    succession_interval_kept = int(quads.attrs.get("succession_interval_kept", 0))
    entered_validation = int(len(quads))
    quads, review_quads, quad_stats = validate_quadruples(quads)
    quad_stats["quadruples_entering_validation"] = entered_validation
    quad_stats["succession_point_candidates"] = succession_point_candidates
    quad_stats["succession_interval_kept"] = succession_interval_kept

    cleaned = cleaned_view(df)
    type_dict = pd.DataFrame(
        [
            {
                "crime_type_id": f"C{spec['code']}",
                "crime_type_code": spec["code"],
                "crime_type_name": spec["name"],
                "chapter": spec["chapter"],
                "aliases": "、".join(spec["aliases"]),
                "elements": "、".join(spec["elements"]),
            }
            for spec in dictionary.by_code.values()
        ]
    )

    cleaned_path = OUTPUT_DIR / "cleaned_crime_records.csv"
    entity_path = OUTPUT_DIR / "entities.csv"
    quad_path = OUTPUT_DIR / "tkg_quadruples.csv"
    review_type_path = OUTPUT_DIR / "manual_review_crime_type.csv"
    review_quad_path = OUTPUT_DIR / "manual_review_quadruples_1pct.csv"
    review_align_path = OUTPUT_DIR / "alignment_review.csv"
    grid_path = OUTPUT_DIR / "city_grid_meta.csv"
    dict_path = OUTPUT_DIR / "crime_type_dict.csv"
    report_path = OUTPUT_DIR / "quality_report.json"

    _log("写出数据集")
    cleaned.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
    entities.to_csv(entity_path, index=False, encoding="utf-8-sig")
    quads.to_csv(quad_path, index=False, encoding="utf-8-sig")
    review_types.to_csv(review_type_path, index=False, encoding="utf-8-sig")
    review_quads.to_csv(review_quad_path, index=False, encoding="utf-8-sig")
    align_mask = cleaned["time_source"].isin(["conflict", "narrative_span"]) | cleaned["place_agreement"].eq("conflict") | cleaned["grid_align_status"].eq("out_of_frame")
    align_review = cleaned.loc[align_mask]
    if len(align_review) > 500:
        align_review = align_review.sample(n=500, random_state=RANDOM_SEED)
    align_cols = [
        "case_number",
        "crime_type_name",
        "incident_time" if "incident_time" in align_review.columns else "iso8601",
        "iso8601",
        "time_grain",
        "time_source",
        "time_align_confidence",
        "time_start_iso",
        "time_end_iso",
        "time_id",
        "place_agreement",
        "spatial_align_confidence",
        "grid_align_status",
        "admin_name",
        "incident_county",
        "admin_distance_m",
    ]
    align_review[[col for col in align_cols if col in align_review.columns]].to_csv(review_align_path, index=False, encoding="utf-8-sig")
    grid_meta.to_csv(grid_path, index=False, encoding="utf-8-sig")
    type_dict.to_csv(dict_path, index=False, encoding="utf-8-sig")

    precision_counts = cleaned["time_precision"].value_counts().astype(int).to_dict()
    report = {
        "source": "Figshare 10.6084/m9.figshare.28106939",
        "input_rows": input_rows,
        "output_records": int(len(cleaned)),
        "entities": int(len(entities)),
        "entity_class_counts": entities["entity_class"].value_counts().astype(int).to_dict(),
        "admin_unmatched_removed": admin_missing,
        "admin_match_counts": cleaned["admin_match"].value_counts().astype(int).to_dict(),
        "time_precision_counts": precision_counts,
        "time_entity_level_counts": cleaned["time_entity_level"].fillna("无").value_counts().astype(int).to_dict(),
        "place_agreement_counts": cleaned["place_agreement"].value_counts().astype(int).to_dict(),
        "grid_align_status_counts": cleaned["grid_align_status"].value_counts().astype(int).to_dict(),
        "mean_time_align_confidence": round(float(cleaned["time_align_confidence"].mean()), 4),
        "mean_spatial_align_confidence": round(float(cleaned["spatial_align_confidence"].mean()), 4),
        "parameters": {
            "time_window": [TIME_START, TIME_END],
            "dbscan_eps_m": 3000,
            "dbscan_min_samples": 2,
            "grid_size_m": 500,
            "succession_hours": 24,
            "type_accept_score": TYPE_ACCEPT_SCORE,
            "low_confidence": LOW_CONFIDENCE,
        },
        **dedup_stats,
        **time_stats,
        **coord_stats,
        **type_stats,
        **dbscan_stats,
        **grid_stats,
        **quad_stats,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
