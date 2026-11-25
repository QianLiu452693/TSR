"""在清洗后的犯罪记录上检验 FIMUT，并给出一个可复核的热点对照。

用法：py -3 run_tkg_fimut.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd

from tkg_pipeline.align import TZ, to_unix
from tkg_pipeline.config import LOW_CONFIDENCE, OUTPUT_DIR
from tkg_pipeline.fimut import fimut_time_unix, grid_w2_meters, spatial_hierarchy, with_space
from datetime import datetime

CLEANED = OUTPUT_DIR / "cleaned_crime_records.csv"
META = OUTPUT_DIR / "city_grid_meta.csv"
REPORT_DIR = OUTPUT_DIR / "tkg" / "fimut"
CUTOFF = to_unix(datetime(2019, 1, 1, tzinfo=TZ))
DECAY_DAYS = (180, 365, 730)


def _log(message: str) -> None:
    print(message, flush=True)


def _code(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return "" if text.lower() in {"nan", "none"} else text


def _hits(scores: dict[str, dict[str, float]], tests: dict[str, list[str]], minimum_grids: int = 50, minimum_events: int = 30) -> dict:
    hit = 0
    used = 0
    chance = 0.0
    cities = 0
    for city, events in tests.items():
        grid_scores = scores.get(city, {})
        if len(grid_scores) < minimum_grids or len(events) < minimum_events:
            continue
        ranking = sorted(grid_scores, key=grid_scores.get, reverse=True)[:10]
        chosen = set(ranking)
        cities += 1
        used += len(events)
        hit += sum(grid in chosen for grid in events)
        chance += len(events) * (10 / len(grid_scores))
    return {
        "cities": cities,
        "test_events": used,
        "hits_at_10": round(hit / used, 4) if used else 0.0,
        "chance_hits_at_10": round(chance / used, 4) if used else 0.0,
    }


def _nll(histogram: np.ndarray, hours: np.ndarray) -> float:
    if hours.size == 0 or histogram.sum() <= 0:
        return 0.0
    probability = histogram + 1e-6
    probability = probability / probability.sum()
    return float(-np.log(probability[hours]).mean())


def main() -> None:
    if not CLEANED.exists():
        raise SystemExit("缺少 output/cleaned_crime_records.csv")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "crime_type_id",
        "city_code",
        "province_code",
        "admin_code",
        "grid_id",
        "grid_x",
        "grid_y",
        "proj_x",
        "proj_y",
        "time_start_unix",
        "time_end_unix",
        "time_id",
        "time_entity_level",
        "time_grain",
        "spatial_align_confidence",
    ]
    frame = pd.read_csv(CLEANED, usecols=columns, low_memory=False)
    meta = pd.read_csv(META, usecols=["city_code", "origin_x", "origin_y"])
    frame["city_code"] = pd.to_numeric(frame["city_code"], errors="coerce")
    meta["city_code"] = pd.to_numeric(meta["city_code"], errors="coerce")
    frame = frame.merge(meta, on="city_code", how="left")
    for column in ("time_start_unix", "time_end_unix", "proj_x", "proj_y", "grid_x", "grid_y", "spatial_align_confidence"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["time_start_unix", "time_end_unix"]).reset_index(drop=True)
    _log(f"FIMUT 对齐 {len(frame)} 条记录")

    cache: dict[tuple[float, float], object] = {}
    level_atoms: dict[str, int] = defaultdict(int)
    effective_time: dict[str, int] = defaultdict(int)
    effective_space = 0
    low_time = 0
    residual = 0.0
    residual_checks = 0
    w2_sum = 0.0
    w2_count = 0
    hard_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    soft_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    precise_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    hard_decay = [defaultdict(lambda: defaultdict(float)) for _ in DECAY_DAYS]
    soft_decay = [defaultdict(lambda: defaultdict(float)) for _ in DECAY_DAYS]
    tests: dict[str, list[str]] = defaultdict(list)
    train_soft_hour = np.zeros(24, dtype=np.float64)
    train_hard_hour = np.zeros(24, dtype=np.float64)
    test_hours: list[int] = []
    samples: list[dict] = []

    def time_plan(start_ts: float, end_ts: float):
        key = (float(start_ts), float(end_ts))
        found = cache.get(key)
        if found is None:
            found = fimut_time_unix(key[0], key[1])
            cache[key] = found
        return found

    for index, record in enumerate(frame.itertuples(index=False)):
        plan = time_plan(float(record.time_start_unix), float(record.time_end_unix))
        for level in {atom.level for atom in plan.atoms}:
            total = plan.level_total(level)
            level_atoms[level] += sum(atom.level == level for atom in plan.atoms)
            residual = max(residual, abs(total - plan.retained))
            residual_checks += 1
        if plan.retained < LOW_CONFIDENCE:
            low_time += 1
        else:
            for atom in plan.atoms:
                effective_time[atom.level] += 1

        grid_id = _code(record.grid_id)
        confidence = float(record.spatial_align_confidence) if pd.notna(record.spatial_align_confidence) else 0.0
        w2 = None
        if grid_id and pd.notna(record.proj_x) and pd.notna(record.origin_x) and pd.notna(record.grid_x):
            w2 = grid_w2_meters(
                float(record.proj_x),
                float(record.proj_y),
                float(record.origin_x),
                float(record.origin_y),
                int(record.grid_x),
                int(record.grid_y),
            )
            w2_sum += w2
            w2_count += 1
        links = spatial_hierarchy(grid_id, _code(record.province_code), _code(record.city_code), _code(record.admin_code), confidence)
        effective_space += sum(bool(entity_id) and link_confidence >= LOW_CONFIDENCE for entity_id, _, link_confidence in links)

        city = _code(record.city_code)
        train = float(record.time_start_unix) < CUTOFF
        precise = str(record.time_entity_level) == "小时"
        if city and grid_id and train:
            hard_scores[city][grid_id] += 1.0
            soft_scores[city][grid_id] += plan.retained * confidence
            if str(record.time_grain) in {"hour", "minute", "second", "day"}:
                precise_scores[city][grid_id] += 1.0
            age = max(CUTOFF - float(record.time_start_unix), 0.0)
            for tau_days, bucket in zip(DECAY_DAYS, hard_decay):
                bucket[city][grid_id] += math.exp(-age / (tau_days * 86400))
            for atom in plan.atoms:
                if atom.level == "月":
                    age_month = max(CUTOFF - to_unix(datetime.fromisoformat(atom.iso8601)), 0.0)
                    for tau_days, bucket in zip(DECAY_DAYS, soft_decay):
                        bucket[city][grid_id] += atom.mass * math.exp(-age_month / (tau_days * 86400))
                elif atom.level == "小时":
                    train_soft_hour[int(atom.entity_id[-2:])] += atom.mass
            if precise and _code(record.time_id).startswith("T") and len(_code(record.time_id)) == 11:
                train_hard_hour[int(_code(record.time_id)[-2:])] += 1.0
        elif city and grid_id and not train:
            tests[city].append(grid_id)
            if precise and _code(record.time_id).startswith("T") and len(_code(record.time_id)) == 11:
                test_hours.append(int(_code(record.time_id)[-2:]))

        if len(samples) < 6 and plan.finest_level == "小时" and len(plan.masses("小时")) >= 4:
            samples.extend(with_space(plan, links, w2).quadruples(_code(record.crime_type_id))[:6])
        if index and index % 200_000 == 0:
            _log(f"  已对齐 {index}")

    hours = np.array(test_hours, dtype=int)
    report = {
        "algorithm": "FIMUT",
        "statement": "最细层是 IMUT 的零位移重叠；更粗的小时、日、周、旬、月质量是这层质量的条件期望。网格由包含关系确定。",
        "records": int(len(frame)),
        "unique_intervals": int(len(cache)),
        "low_confidence": LOW_CONFIDENCE,
        "records_time_plan_below_cutoff": int(low_time),
        "max_level_mass_residual": residual,
        "residual_checks": int(residual_checks),
        "time_atoms_by_level": {key: int(value) for key, value in sorted(level_atoms.items())},
        "effective_time_quadruples_by_level": {key: int(value) for key, value in sorted(effective_time.items())},
        "effective_time_quadruples": int(sum(effective_time.values())),
        "effective_space_quadruples": int(effective_space),
        "effective_quadruples": int(sum(effective_time.values()) + effective_space),
        "grid_w2_mean_m": round(w2_sum / w2_count, 2) if w2_count else None,
        "grid_w2_center_reference_m": round(500 / (6 ** 0.5), 2),
        "grid_w2_observations": int(w2_count),
        "hotspot_holdout": "2019-01-01+08 之前为训练，之后为测试；只统计训练网格不少于 50、测试事件不少于 30 的城市",
        "hotspot_hard_count": _hits(hard_scores, tests),
        "hotspot_precise_only": _hits(precise_scores, tests),
        "hotspot_fimut_mass": _hits(soft_scores, tests),
        "hotspot_decay": {
            f"{days}_days": {
                "hard_anchor": _hits(hard_bucket, tests),
                "fimut_month_mass": _hits(soft_bucket, tests),
            }
            for days, hard_bucket, soft_bucket in zip(DECAY_DAYS, hard_decay, soft_decay)
        },
        "hour_of_day_test_events": int(hours.size),
        "hour_of_day_nll_fimut": round(_nll(train_soft_hour, hours), 4),
        "hour_of_day_nll_hard_hour_entity": round(_nll(train_hard_hour, hours), 4),
        "hour_of_day_nll_uniform": round(_nll(np.ones(24), hours), 4),
    }
    path = REPORT_DIR / "fimut_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    sample = pd.DataFrame(samples[:12])
    if not sample.empty:
        sample["confidence"] = sample["confidence"].round(4)
        sample.to_csv(REPORT_DIR / "fimut_sample.csv", index=False, encoding="utf-8-sig")
    _log(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
