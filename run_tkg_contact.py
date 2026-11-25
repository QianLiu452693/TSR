"""接触时刻衰减和四邻网格的近重复对照。

尺度固定为窗口长度，邻格权重固定为 0.5。用法：py -3 run_tkg_contact.py
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from run_tkg_eval import (
    BOOTSTRAP,
    DAY,
    LOOKBACK_DAYS,
    MIN_CITY,
    MIN_SEEN,
    WINDOWS,
    YEARS,
    _add,
    _blank,
    _clear,
    _cuts,
    _log,
    _paired,
    _ranking,
    _summarize,
)
from tkg_pipeline.config import OUTPUT_DIR
from tkg_pipeline.fimut import contact_score, tie_hit_probability

CLEANED = OUTPUT_DIR / "cleaned_crime_records.csv"
REPORT_DIR = OUTPUT_DIR / "tkg" / "fimut"
METHODS = ("overlap", "contact", "overlap_neighbor", "contact_neighbor")
NEIGHBOR = 0.5
OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _chance_mrr(n_seen: int, true_seen: bool) -> float:
    if n_seen <= 0:
        return 0.0
    if not true_seen:
        return 1.0 / (n_seen + 1)
    if n_seen <= 40:
        return float(sum(1.0 / rank for rank in range(1, n_seen + 1)) / n_seen)
    return (math.log(n_seen) + np.euler_gamma) / n_seen


def _neighbors(xs: np.ndarray, ys: np.ndarray, grids: np.ndarray, n_grids: int) -> list[list[int]]:
    coord = {}
    for x_value, y_value, grid in zip(xs, ys, grids):
        if not np.isfinite(x_value) or not np.isfinite(y_value):
            continue
        coord[(int(x_value), int(y_value))] = int(grid)
    adjacent = [[] for _ in range(n_grids)]
    for (x_value, y_value), grid in coord.items():
        for dx, dy in OFFSETS:
            other = coord.get((x_value + dx, y_value + dy))
            if other is not None and other != grid:
                adjacent[grid].append(other)
    return adjacent


def evaluate_city(part: pd.DataFrame, cuts: dict[int, tuple[float, float]]) -> dict:
    starts = part["time_start_unix"].to_numpy(dtype=np.float64)
    ends = part["time_end_unix"].to_numpy(dtype=np.float64)
    order = np.argsort(starts, kind="mergesort")
    starts = starts[order]
    ends = np.maximum(ends[order], starts)
    grids, _ = pd.factorize(part["grid_id"].to_numpy()[order], sort=False)
    xs = pd.to_numeric(part["grid_x"], errors="coerce").to_numpy()[order]
    ys = pd.to_numeric(part["grid_y"], errors="coerce").to_numpy()[order]
    n_grids = int(grids.max()) + 1 if len(grids) else 0
    adjacent = _neighbors(xs, ys, grids, n_grids)
    years = np.zeros(len(starts), dtype=np.int16)
    for year, (left, right) in cuts.items():
        years[(starts >= left) & (starts < right)] = year

    slots = len(METHODS) * len(WINDOWS)
    acc = np.zeros((slots, n_grids), dtype=np.float64)
    touched = [[] for _ in range(slots)]
    seen = np.zeros(n_grids, dtype=bool)
    n_seen = 0
    totals = {f"{year}|{window}|{method}": _blank() for year in YEARS for window in WINDOWS for method in ("chance", *METHODS)}
    repeat = {"n": 0.0, "same": 0.0, "base": 0.0}
    lookback = LOOKBACK_DAYS * DAY

    for index in range(len(starts)):
        grid = int(grids[index])
        year = int(years[index])
        if year in cuts and n_seen >= MIN_SEEN:
            query = float(starts[index])
            earliest = query - lookback
            cursor = index - 1
            while cursor >= 0 and starts[cursor] >= earliest:
                left = float(starts[cursor])
                right = float(ends[cursor])
                target = int(grids[cursor])
                if not seen[target]:
                    cursor -= 1
                    continue
                for window_index, window in enumerate(WINDOWS):
                    opening = query - window * DAY
                    if right <= opening or left >= query:
                        continue
                    weight = contact_score(left, right, query, window * DAY)
                    base = window_index
                    _add(acc[base], touched[base], target, 1.0)
                    _add(acc[3 + base], touched[3 + base], target, weight)
                    _add(acc[6 + base], touched[6 + base], target, 1.0)
                    _add(acc[9 + base], touched[9 + base], target, weight)
                    for other in adjacent[target]:
                        if seen[other]:
                            _add(acc[6 + base], touched[6 + base], other, NEIGHBOR)
                            _add(acc[9 + base], touched[9 + base], other, NEIGHBOR * weight)
                cursor -= 1

            true_seen = bool(seen[grid])
            for window_index, window in enumerate(WINDOWS):
                labels = (
                    (f"{year}|{window}|overlap", window_index),
                    (f"{year}|{window}|contact", 3 + window_index),
                    (f"{year}|{window}|overlap_neighbor", 6 + window_index),
                    (f"{year}|{window}|contact_neighbor", 9 + window_index),
                )
                for key, slot in labels:
                    hit1, hit5, hit10, mrr = _ranking(acc[slot], touched[slot], grid, true_seen, n_seen)
                    bucket = totals[key]
                    bucket["n"] += 1
                    bucket["new_grid"] += float(not true_seen)
                    bucket["hit1"] += hit1
                    bucket["hit5"] += hit5
                    bucket["hit10"] += hit10
                    bucket["mrr"] += mrr
                chance = totals[f"{year}|{window}|chance"]
                chance["n"] += 1
                chance["new_grid"] += float(not true_seen)
                if true_seen:
                    chance["hit1"] += tie_hit_probability(0, n_seen, 1)
                    chance["hit5"] += tie_hit_probability(0, n_seen, 5)
                    chance["hit10"] += tie_hit_probability(0, n_seen, 10)
                    chance["mrr"] += _chance_mrr(n_seen, True)
                else:
                    chance["mrr"] += _chance_mrr(n_seen, False)
            if true_seen and n_seen:
                active = sum(float(acc[0, item]) > 0 for item in touched[0])
                repeat["n"] += 1
                repeat["same"] += float(acc[0, grid] > 0)
                repeat["base"] += active / n_seen
            for slot in range(slots):
                _clear(acc[slot], touched[slot])
        if not seen[grid]:
            seen[grid] = True
            n_seen += 1
    totals["city"] = str(int(part["city_code"].iloc[0]))
    totals["grids"] = float(n_grids)
    totals["repeat7"] = repeat
    return totals


def main() -> None:
    if not CLEANED.exists():
        raise SystemExit("缺少 output/cleaned_crime_records.csv")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        CLEANED,
        usecols=["city_code", "grid_id", "grid_x", "grid_y", "time_start_unix", "time_end_unix"],
        low_memory=False,
    )
    for column in ("city_code", "time_start_unix", "time_end_unix", "grid_x", "grid_y"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["time_start_unix", "time_end_unix", "grid_id", "city_code"]).copy()
    cuts = _cuts()
    rows = []
    grouped = list(frame.groupby("city_code", sort=False))
    _log(f"接触衰减对照 {len(frame)} 条，{len(grouped)} 个城市")
    for number, (_, part) in enumerate(grouped, start=1):
        rows.append(evaluate_city(part, cuts))
        if number % 40 == 0:
            _log(f"  已完成 {number}/{len(grouped)} 个城市")
    flat_rows = []
    repeat_n = repeat_same = repeat_base = 0.0
    for totals in rows:
        flat = {"city": totals["city"], "grids": totals["grids"]}
        repeat_n += totals["repeat7"]["n"]
        repeat_same += totals["repeat7"]["same"]
        repeat_base += totals["repeat7"]["base"]
        for key, value in totals.items():
            if isinstance(value, dict) and "n" in value:
                for metric, number in value.items():
                    flat[f"{key}|{metric}"] = number
        flat_rows.append(flat)
    city_frame = pd.DataFrame(flat_rows)
    rng = np.random.default_rng(0)
    near = {}
    paired = {}
    for year in YEARS:
        near[str(year)] = {}
        paired[str(year)] = {}
        for window in WINDOWS:
            near[str(year)][f"{window}d"] = {
                method: _summarize(city_frame, f"{year}|{window}|{method}", rng) for method in ("chance", *METHODS)
            }
            paired[str(year)][f"{window}d"] = {
                "contact_minus_overlap": _paired(city_frame, f"{year}|{window}|contact", f"{year}|{window}|overlap", rng),
                "overlap_neighbor_minus_overlap": _paired(city_frame, f"{year}|{window}|overlap_neighbor", f"{year}|{window}|overlap", rng),
                "contact_neighbor_minus_overlap": _paired(city_frame, f"{year}|{window}|contact_neighbor", f"{year}|{window}|overlap", rng),
            }
    report = {
        "protocol": {
            "contact": "相交段中点到查询时刻的时间，按窗口长度作指数衰减",
            "neighbor_weight": NEIGHBOR,
            "neighbor_definition": "同一城市内网格索引上下左右四邻，且该邻格在查询前已经出现",
            "chance": "在已经出现的网格中随机排序；新网格的命中率为 0",
            "repeat7": "7 天窗口内，本网格有历史记录的比例，除以任一已出现网格有历史记录的平均比例",
        },
        "near_repeat": near,
        "paired_difference_against_overlap": paired,
        "same_grid_repeat_ratio_7d": {
            "events": int(repeat_n),
            "same_grid_rate": round(repeat_same / repeat_n, 4) if repeat_n else None,
            "random_grid_rate": round(repeat_base / repeat_n, 4) if repeat_n else None,
            "ratio": round(repeat_same / repeat_base, 3) if repeat_base else None,
        },
    }
    path = REPORT_DIR / "contact_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    city_frame.to_csv(REPORT_DIR / "contact_cities.csv", index=False, encoding="utf-8-sig")
    _log(json.dumps({"repeat": report["same_grid_repeat_ratio_7d"], "paired_2019_30d": paired["2019"]["30d"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
