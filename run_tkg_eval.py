"""近重复、时间轮廓和表示误差的对照实验。

用法：py -3 run_tkg_eval.py
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from tkg_pipeline.align import TZ, certainly_successive, to_unix
from tkg_pipeline.config import OUTPUT_DIR
from tkg_pipeline.fimut import fimut_time_unix, restricted_mass, tie_hit_probability
from tkg_pipeline.imut import adapted_succession_mass, imut_align_unix

CLEANED = OUTPUT_DIR / "cleaned_crime_records.csv"
META = OUTPUT_DIR / "city_grid_meta.csv"
REPORT_DIR = OUTPUT_DIR / "tkg" / "fimut"
YEARS = (2017, 2018, 2019)
WINDOWS = (7, 30, 90)
METHODS = ("hard_start", "hard_overlap", "fimut")
ETAS = (360.0, 720.0, 1440.0)
MIN_SEEN = 30
MIN_CITY = 30
LOOKBACK_DAYS = 90 + 180
BOOTSTRAP = 2000
DAY = 86400.0


def _log(message: str) -> None:
    print(message, flush=True)


def _cuts() -> dict[int, tuple[float, float]]:
    bounds = {year: to_unix(datetime(year, 1, 1, tzinfo=TZ)) for year in (2017, 2018, 2019, 2020)}
    return {year: (float(bounds[year]), float(bounds[year + 1])) for year in YEARS}


def _mrr(better: int, tied_with_self: int) -> float:
    if tied_with_self <= 0:
        return 0.0
    start = better + 1
    end = better + tied_with_self
    if tied_with_self <= 40:
        return float(sum(1.0 / rank for rank in range(start, end + 1)) / tied_with_self)
    return 1.0 / (better + (tied_with_self + 1) / 2.0)


def _ranking(acc: np.ndarray, touched: list[int], true_grid: int, true_seen: bool, n_seen: int) -> tuple[float, float, float, float]:
    if n_seen <= 0:
        return 0.0, 0.0, 0.0, 0.0
    if not true_seen:
        return 0.0, 0.0, 0.0, 1.0 / (n_seen + 1)
    true_score = float(acc[true_grid])
    if true_score > 1e-15:
        better = 0
        tied_others = 0
        for grid in touched:
            if grid == true_grid:
                continue
            value = float(acc[grid])
            if value > true_score + 1e-12:
                better += 1
            elif abs(value - true_score) <= 1e-12:
                tied_others += 1
        tied = tied_others + 1
    else:
        better = sum(float(acc[grid]) > 1e-15 for grid in touched)
        tied = n_seen - better
    return (
        tie_hit_probability(better, tied, 1),
        tie_hit_probability(better, tied, 5),
        tie_hit_probability(better, tied, 10),
        _mrr(better, tied),
    )


def _add(acc: np.ndarray, touched: list[int], grid: int, value: float) -> None:
    if value <= 0:
        return
    if acc[grid] == 0.0:
        touched.append(grid)
    acc[grid] += value


def _clear(acc: np.ndarray, touched: list[int]) -> None:
    for grid in touched:
        acc[grid] = 0.0
    touched.clear()


def _blank() -> dict[str, float]:
    return {"n": 0.0, "new_grid": 0.0, "hit1": 0.0, "hit5": 0.0, "hit10": 0.0, "mrr": 0.0}


def evaluate_city(city: str, part: pd.DataFrame, cuts: dict[int, tuple[float, float]], reservoir: list, rng: np.random.Generator, seen_pairs: list[int]) -> dict:
    starts = part["time_start_unix"].to_numpy(dtype=np.float64)
    ends = part["time_end_unix"].to_numpy(dtype=np.float64)
    order = np.argsort(starts, kind="mergesort")
    starts, ends = starts[order], np.maximum(ends[order], starts[order])
    grids, _ = pd.factorize(part["grid_id"].to_numpy()[order], sort=False)
    types, _ = pd.factorize(part["crime_type_id"].to_numpy()[order], sort=False)
    grains = part["time_grain"].to_numpy()[order]
    theft = part["crime_type_id"].to_numpy()[order] == "C0101"
    width = np.maximum(ends - starts, 1.0)
    retained = {eta: np.exp(-(width / 3600.0) / eta) for eta in ETAS}
    n_grids = int(grids.max()) + 1 if len(grids) else 0
    years = np.zeros(len(starts), dtype=np.int16)
    for year, (left, right) in cuts.items():
        years[(starts >= left) & (starts < right)] = year

    slots = 16
    acc = np.zeros((slots, n_grids), dtype=np.float64)
    touched = [[] for _ in range(slots)]
    seen = np.zeros(n_grids, dtype=bool)
    n_seen = 0
    totals = {f"{year}|{window}|{method}": _blank() for year in YEARS for window in WINDOWS for method in METHODS}
    for eta, name in ((360.0, "fimut_360"), (1440.0, "fimut_1440")):
        totals[f"2019|30|{name}"] = _blank()
    for method in METHODS:
        totals[f"2019|30|same|{method}"] = _blank()
    for grain in ("precise", "stated", "coarse"):
        for method in METHODS:
            totals[f"2019|30|{grain}|{method}"] = _blank()
    for group in ("theft", "other"):
        for method in METHODS:
            totals[f"2019|30|{group}|{method}"] = _blank()

    grain_name = {
        "hour": "precise",
        "minute": "precise",
        "second": "precise",
        "period": "stated",
        "day": "stated",
        "span": "coarse",
        "month": "coarse",
        "ten_day": "coarse",
    }
    lookback = LOOKBACK_DAYS * DAY
    last_on_grid = {}

    for index in range(len(starts)):
        grid = int(grids[index])
        previous = last_on_grid.get(grid)
        if previous is not None:
            seen_pairs[0] += 1
            pair = (float(starts[previous]), float(ends[previous]), float(starts[index]), float(ends[index]))
            if len(reservoir) < 12000:
                reservoir.append(pair)
            else:
                pick = int(rng.integers(0, seen_pairs[0]))
                if pick < 12000:
                    reservoir[pick] = pair
        last_on_grid[grid] = index

        year = int(years[index])
        if year in cuts and n_seen >= MIN_SEEN:
            query = float(starts[index])
            earliest = query - lookback
            cursor = index - 1
            while cursor >= 0 and starts[cursor] >= earliest:
                left = float(starts[cursor])
                right = float(ends[cursor])
                target = int(grids[cursor])
                same = int(types[cursor]) == int(types[index])
                span = float(width[cursor])
                for window_index, window in enumerate(WINDOWS):
                    opening = query - window * DAY
                    if right <= opening or left >= query:
                        continue
                    if opening <= left < query:
                        _add(acc[window_index], touched[window_index], target, 1.0)
                    _add(acc[3 + window_index], touched[3 + window_index], target, 1.0)
                    overlap = min(right, query) - max(left, opening)
                    _add(acc[6 + window_index], touched[6 + window_index], target, float(retained[720.0][cursor]) * overlap / span)
                    if year == 2019 and window == 30:
                        _add(acc[11], touched[11], target, float(retained[360.0][cursor]) * overlap / span)
                        _add(acc[12], touched[12], target, float(retained[1440.0][cursor]) * overlap / span)
                        if same:
                            if opening <= left < query:
                                _add(acc[13], touched[13], target, 1.0)
                            _add(acc[14], touched[14], target, 1.0)
                            _add(acc[15], touched[15], target, float(retained[720.0][cursor]) * overlap / span)
                cursor -= 1

            true_seen = bool(seen[grid])
            labels = []
            for window_index, window in enumerate(WINDOWS):
                for method_index, method in enumerate(METHODS):
                    slot = method_index * 3 + window_index
                    labels.append((f"{year}|{window}|{method}", slot))
            if year == 2019:
                labels.append(("2019|30|fimut_360", 11))
                labels.append(("2019|30|fimut_1440", 12))
                labels.append(("2019|30|same|hard_start", 13))
                labels.append(("2019|30|same|hard_overlap", 14))
                labels.append(("2019|30|same|fimut", 15))
                group = "theft" if bool(theft[index]) else "other"
                grain = grain_name.get(str(grains[index]), "")
                extra = []
                if grain:
                    extra.append(f"2019|30|{grain}|")
                extra.append(f"2019|30|{group}|")
            else:
                extra = []

            for key, slot in labels:
                hit1, hit5, hit10, mrr = _ranking(acc[slot], touched[slot], grid, true_seen, n_seen)
                bucket = totals[key]
                bucket["n"] += 1
                bucket["new_grid"] += float(not true_seen)
                bucket["hit1"] += hit1
                bucket["hit5"] += hit5
                bucket["hit10"] += hit10
                bucket["mrr"] += mrr
                if key.startswith("2019|30|") and key.count("|") == 2:
                    method = key.split("|")[-1]
                    for prefix in extra:
                        shadow = totals.get(prefix + method)
                        if shadow is None:
                            continue
                        shadow["n"] += 1
                        shadow["new_grid"] += float(not true_seen)
                        shadow["hit1"] += hit1
                        shadow["hit5"] += hit5
                        shadow["hit10"] += hit10
                        shadow["mrr"] += mrr
            for slot in range(slots):
                _clear(acc[slot], touched[slot])

        if not seen[grid]:
            seen[grid] = True
            n_seen += 1
    totals["city"] = city
    totals["grids"] = float(n_grids)
    return totals


def _ci(values: np.ndarray) -> list[float]:
    return [round(float(np.quantile(values, 0.025)), 4), round(float(np.quantile(values, 0.975)), 4)]


def _summarize(frame: pd.DataFrame, prefix: str, rng: np.random.Generator) -> dict | None:
    count = frame[f"{prefix}|n"].to_numpy(dtype=float)
    keep = count >= MIN_CITY
    if not np.any(keep):
        return None
    count = count[keep]
    hit1 = frame[f"{prefix}|hit1"].to_numpy(dtype=float)[keep]
    hit5 = frame[f"{prefix}|hit5"].to_numpy(dtype=float)[keep]
    hit10 = frame[f"{prefix}|hit10"].to_numpy(dtype=float)[keep]
    mrr = frame[f"{prefix}|mrr"].to_numpy(dtype=float)[keep]
    new_grid = frame[f"{prefix}|new_grid"].to_numpy(dtype=float)[keep]
    index = np.arange(len(count))
    micro10, macro10, micro_mrr = [], [], []
    for _ in range(BOOTSTRAP):
        draw = rng.choice(index, size=len(index), replace=True)
        micro10.append(hit10[draw].sum() / count[draw].sum())
        macro10.append((hit10[draw] / count[draw]).mean())
        micro_mrr.append(mrr[draw].sum() / count[draw].sum())
    micro10 = np.array(micro10)
    macro10 = np.array(macro10)
    micro_mrr = np.array(micro_mrr)
    return {
        "cities": int(len(count)),
        "events": int(count.sum()),
        "new_grid_rate": round(float(new_grid.sum() / count.sum()), 4),
        "hits_at_1": round(float(hit1.sum() / count.sum()), 4),
        "hits_at_5": round(float(hit5.sum() / count.sum()), 4),
        "hits_at_10": round(float(hit10.sum() / count.sum()), 4),
        "hits_at_10_ci": _ci(micro10),
        "macro_hits_at_10": round(float((hit10 / count).mean()), 4),
        "macro_hits_at_10_ci": _ci(macro10),
        "city_hits_at_10_p25": round(float(np.quantile(hit10 / count, 0.25)), 4),
        "city_hits_at_10_p50": round(float(np.quantile(hit10 / count, 0.50)), 4),
        "city_hits_at_10_p75": round(float(np.quantile(hit10 / count, 0.75)), 4),
        "mrr": round(float(mrr.sum() / count.sum()), 4),
        "mrr_ci": _ci(micro_mrr),
    }


def _paired(frame: pd.DataFrame, left: str, right: str, rng: np.random.Generator) -> dict | None:
    count = frame[f"{left}|n"].to_numpy(dtype=float)
    keep = count >= MIN_CITY
    if not np.any(keep):
        return None
    count = count[keep]
    diff = frame[f"{left}|hit10"].to_numpy(dtype=float)[keep] - frame[f"{right}|hit10"].to_numpy(dtype=float)[keep]
    index = np.arange(len(count))
    micros, macros = [], []
    for _ in range(BOOTSTRAP):
        draw = rng.choice(index, size=len(index), replace=True)
        micros.append(diff[draw].sum() / count[draw].sum())
        macros.append((diff[draw] / count[draw]).mean())
    point = float(diff.sum() / count.sum())
    return {
        "events": int(count.sum()),
        "micro_hits_at_10": round(point, 4),
        "micro_ci": _ci(np.array(micros)),
        "macro_hits_at_10": round(float((diff / count).mean()), 4),
        "macro_ci": _ci(np.array(macros)),
        "cities_fimut_higher": int(np.sum(diff > 1e-9)),
        "cities_hard_higher": int(np.sum(diff < -1e-9)),
        "cities_tie": int(np.sum(np.abs(diff) <= 1e-9)),
    }


def _hour_of(timestamps: np.ndarray) -> np.ndarray:
    return ((timestamps + 8 * 3600) // 3600).astype(np.int64) % 24


def _tv(left: np.ndarray, right: np.ndarray) -> float:
    left = left / left.sum()
    right = right / right.sum()
    return float(0.5 * np.abs(left - right).sum())


def _nll(histogram: np.ndarray, hours: np.ndarray) -> float:
    probability = histogram + 1e-6
    probability = probability / probability.sum()
    return float(-np.log(probability[hours]).mean())


def descriptive(frame: pd.DataFrame, cuts: dict[int, tuple[float, float]]) -> dict:
    width_hours = np.maximum(frame["time_end_unix"] - frame["time_start_unix"], 1.0) / 3600.0
    retention = {}
    for eta in ETAS:
        mass = np.exp(-width_hours / eta)
        table = {}
        for grain, index in frame.groupby("time_grain").groups.items():
            values = mass.loc[index]
            table[str(grain)] = {
                "records": int(len(values)),
                "mean_retained": round(float(values.mean()), 4),
                "fraction_below_0_30": round(float((values < 0.30).mean()), 4),
            }
        retention[str(int(eta))] = table

    merged = frame.dropna(subset=["grid_x", "origin_x", "proj_x"]).copy()
    xmin = merged["origin_x"] + merged["grid_x"] * 500.0
    ymin = merged["origin_y"] + merged["grid_y"] * 500.0

    def axis(point, low):
        length = 500.0
        return (point - low) ** 2 - (point - low) * length + length ** 2 / 3.0

    w2 = np.sqrt(np.maximum(axis(merged["proj_x"], xmin) + axis(merged["proj_y"], ymin), 0.0))
    w2_table = {
        "observations": int(len(w2)),
        "p10": round(float(np.quantile(w2, 0.10)), 2),
        "p50": round(float(np.quantile(w2, 0.50)), 2),
        "p90": round(float(np.quantile(w2, 0.90)), 2),
        "mean": round(float(w2.mean()), 2),
        "center_reference_m": round(500 / (6 ** 0.5), 2),
    }
    by_match = {}
    for name, index in merged.groupby("admin_match").groups.items():
        values = w2.loc[index]
        by_match[str(name)] = {"observations": int(len(values)), "p50": round(float(np.quantile(values, 0.50)), 2)}
    confidence = frame.groupby("city_code")["spatial_align_confidence"].agg(["mean", "std"])
    cv = (confidence["std"] / confidence["mean"]).replace([np.inf, -np.inf], np.nan).dropna()

    precise = frame["time_grain"].isin(["hour", "minute", "second"])
    period = frame["time_grain"].eq("period")
    years = np.zeros(len(frame), dtype=np.int16)
    starts = frame["time_start_unix"].to_numpy(dtype=float)
    for year, (left, right) in cuts.items():
        years[(starts >= left) & (starts < right)] = year
    precise_hist = {year: np.zeros(24) for year in YEARS}
    period_hist = {year: np.zeros(24) for year in YEARS}
    precise_hours = _hour_of(starts[precise.to_numpy()])
    precise_years = years[precise.to_numpy()]
    for year in YEARS:
        chosen = precise_hours[precise_years == year]
        if len(chosen):
            precise_hist[year] += np.bincount(chosen, minlength=24)
    period_starts = starts[period.to_numpy()]
    period_ends = frame.loc[period, "time_end_unix"].to_numpy(dtype=float)
    period_years = years[period.to_numpy()]
    for left, right, year in zip(period_starts, period_ends, period_years):
        if year not in period_hist or right <= left:
            continue
        hour = int(_hour_of(np.array([left]))[0])
        remaining = right - left
        while remaining > 0 and (right - left) <= 36 * 3600:
            step = min(3600.0, remaining)
            period_hist[year][hour] += step / max(right - left, 1.0)
            remaining -= step
            hour = (hour + 1) % 24
            if step <= 0:
                break
    profiles = {}
    for train, test in ((2017, 2018), (2018, 2019)):
        hours = precise_hours[precise_years == test]
        profiles[f"{train}_to_{test}"] = {
            "test_precise_events": int(len(hours)),
            "tv_precise_vs_next_precise": round(_tv(precise_hist[train], precise_hist[test]), 4) if precise_hist[test].sum() else None,
            "tv_period_vs_next_precise": round(_tv(period_hist[train], precise_hist[test]), 4) if period_hist[train].sum() and precise_hist[test].sum() else None,
            "nll_from_precise": round(_nll(precise_hist[train], hours), 4) if len(hours) and precise_hist[train].sum() else None,
            "nll_from_period": round(_nll(period_hist[train], hours), 4) if len(hours) and period_hist[train].sum() else None,
            "nll_uniform": round(_nll(np.ones(24), hours), 4) if len(hours) else None,
        }
    return {
        "retention_by_eta_hours": retention,
        "grid_w2_m": w2_table,
        "grid_w2_by_admin_match": by_match,
        "within_city_spatial_confidence_cv_p50": round(float(cv.median()), 4) if len(cv) else None,
        "hour_profile": profiles,
    }


def succession_report(pairs: list[tuple[float, float, float, float]]) -> dict:
    gaps = []
    hard = []
    soft = []
    for start_a, end_a, start_b, end_b in pairs:
        early = imut_align_unix(start_a, end_a)
        late = imut_align_unix(start_b, end_b)
        soft.append(adapted_succession_mass(early, late))
        hard.append(int(certainly_successive(int(start_a), int(end_a), int(start_b), int(end_b))))
    soft = np.array(soft, dtype=float)
    hard = np.array(hard, dtype=int)
    gap = np.array([start_b - end_a for _, end_a, start_b, _ in pairs], dtype=float)
    bins = (
        ("overlap", gap < 0),
        ("0_1h", (gap >= 0) & (gap <= 3600)),
        ("1_6h", (gap > 3600) & (gap <= 6 * 3600)),
        ("6_24h", (gap > 6 * 3600) & (gap <= DAY)),
        ("1_3d", (gap > DAY) & (gap <= 3 * DAY)),
        ("3_7d", (gap > 3 * DAY) & (gap <= 7 * DAY)),
        ("over_7d", gap > 7 * DAY),
    )
    by_gap = {}
    for name, mask in bins:
        by_gap[name] = {
            "pairs": int(mask.sum()),
            "hard_rate": round(float(hard[mask].mean()), 4) if mask.any() else None,
            "mean_adapted_mass": round(float(soft[mask].mean()), 4) if mask.any() else None,
        }
    edges = (0.0, 0.05, 0.2, 0.5, 0.8, 1.01)
    reliability = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (soft >= left) & (soft < right)
        reliability.append(
            {
                "mass_from": left,
                "mass_to": right,
                "pairs": int(mask.sum()),
                "hard_rate": round(float(hard[mask].mean()), 4) if mask.any() else None,
            }
        )
    return {"pairs": int(len(pairs)), "by_gap": by_gap, "reliability": reliability}


def filtration_audit(frame: pd.DataFrame, rng: np.random.Generator) -> dict:
    sample = frame.sample(n=min(800, len(frame)), random_state=int(rng.integers(0, 1_000_000)))
    residual = 0.0
    for start, end in zip(sample["time_start_unix"], sample["time_end_unix"]):
        plan = fimut_time_unix(float(start), float(end))
        for level in {atom.level for atom in plan.atoms}:
            residual = max(residual, abs(plan.level_total(level) - plan.retained))
    return {"plans": int(len(sample)), "max_level_mass_residual": residual}


def main() -> None:
    if not CLEANED.exists():
        raise SystemExit("缺少 output/cleaned_crime_records.csv")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    columns = [
        "city_code",
        "crime_type_id",
        "crime_type_name",
        "grid_id",
        "grid_x",
        "grid_y",
        "proj_x",
        "proj_y",
        "admin_match",
        "spatial_align_confidence",
        "time_start_unix",
        "time_end_unix",
        "time_grain",
    ]
    frame = pd.read_csv(CLEANED, usecols=columns, low_memory=False)
    meta = pd.read_csv(META, usecols=["city_code", "origin_x", "origin_y"])
    for column in ("city_code", "time_start_unix", "time_end_unix", "grid_x", "grid_y", "proj_x", "proj_y", "spatial_align_confidence"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    meta["city_code"] = pd.to_numeric(meta["city_code"], errors="coerce")
    frame = frame.merge(meta, on="city_code", how="left")
    frame = frame.dropna(subset=["time_start_unix", "time_end_unix", "grid_id", "city_code"]).copy()
    frame["crime_type_id"] = frame["crime_type_id"].fillna("").astype(str)
    frame["time_grain"] = frame["time_grain"].fillna("unknown").astype(str)
    cuts = _cuts()
    _log(f"描述统计 {len(frame)} 条有网格的记录")
    described = descriptive(frame, cuts)
    _log("滤子抽查")
    audit = filtration_audit(frame, rng)

    rows = []
    reservoir: list[tuple[float, float, float, float]] = []
    seen_pairs = [0]
    grouped = list(frame.groupby("city_code", sort=False))
    for number, (city, part) in enumerate(grouped, start=1):
        rows.append(evaluate_city(str(int(city)), part, cuts, reservoir, rng, seen_pairs))
        if number % 40 == 0:
            _log(f"  已完成 {number}/{len(grouped)} 个城市")
    flat_rows = []
    for totals in rows:
        flat = {"city": totals["city"], "grids": totals["grids"]}
        for key, value in totals.items():
            if isinstance(value, dict):
                for metric, number in value.items():
                    flat[f"{key}|{metric}"] = number
        flat_rows.append(flat)
    city_frame = pd.DataFrame(flat_rows)
    primary = ["city", "grids"]
    for column in city_frame.columns:
        if column.startswith("2019|30|") and "same" not in column and "precise" not in column and "stated" not in column and "coarse" not in column and "theft" not in column and "other" not in column and "360" not in column and "1440" not in column:
            primary.append(column)
    city_frame[primary].to_csv(REPORT_DIR / "eval_cities_2019_30d.csv", index=False, encoding="utf-8-sig")

    near = {}
    paired = {}
    for year in YEARS:
        near[str(year)] = {}
        paired[str(year)] = {}
        for window in WINDOWS:
            near[str(year)][f"{window}d"] = {method: _summarize(city_frame, f"{year}|{window}|{method}", rng) for method in METHODS}
            paired[str(year)][f"{window}d"] = {
                "fimut_minus_hard_start": _paired(city_frame, f"{year}|{window}|fimut", f"{year}|{window}|hard_start", rng),
                "fimut_minus_hard_overlap": _paired(city_frame, f"{year}|{window}|fimut", f"{year}|{window}|hard_overlap", rng),
            }
    extra = {
        "eta": {
            "360": _summarize(city_frame, "2019|30|fimut_360", rng),
            "720": _summarize(city_frame, "2019|30|fimut", rng),
            "1440": _summarize(city_frame, "2019|30|fimut_1440", rng),
        },
        "same_type": {method: _summarize(city_frame, f"2019|30|same|{method}", rng) for method in METHODS},
        "by_test_grain": {grain: {method: _summarize(city_frame, f"2019|30|{grain}|{method}", rng) for method in METHODS} for grain in ("precise", "stated", "coarse")},
        "by_crime_group": {group: {method: _summarize(city_frame, f"2019|30|{group}|{method}", rng) for method in METHODS} for group in ("theft", "other")},
    }
    _log("相继可靠性")
    succession = succession_report(reservoir)
    names = frame[["crime_type_id", "crime_type_name"]].drop_duplicates().set_index("crime_type_id")["crime_type_name"].to_dict()
    report = {
        "protocol": {
            "task": "同一城市内，用查询时刻之前的记录给网格打分，看当前案件落在不落在前 10 名",
            "test_years": list(YEARS),
            "windows_days": list(WINDOWS),
            "minimum_grids_seen": MIN_SEEN,
            "minimum_city_events": MIN_CITY,
            "lookback_days": LOOKBACK_DAYS,
            "tie_break": "并列名次随机拆开，Hits 取进入前 K 的概率",
            "bootstrap_cities": BOOTSTRAP,
            "hard_start": "窗口内按区间起点计 1",
            "hard_overlap": "区间与窗口相交就计 1",
            "fimut": "保留质量乘以落在窗口内的长度比例",
            "truncation": "只回看窗口再加 180 天；更早且更长的区间保留质量已低于 0.003",
        },
        "crime_type_names": {str(key): str(value) for key, value in names.items()},
        "near_repeat": near,
        "paired_difference": paired,
        "slices_2019_30d": extra,
        "succession": succession,
        "descriptive": described,
        "filtration_audit": audit,
    }
    path = REPORT_DIR / "eval_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(json.dumps({"near_repeat_2019_30d": near["2019"]["30d"], "paired_2019_30d": paired["2019"]["30d"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
