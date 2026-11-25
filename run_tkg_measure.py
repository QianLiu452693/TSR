"""潜在时刻恢复与同一网格上的区间 Knox 统计量。

用法：py -3 run_tkg_measure.py

窗口、平滑、年份和阈值在运行前写定。结果出来之后不改这些常数，
也不把近重复排序从「区间是否与查询窗口相交」改成倾斜分数。
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from tkg_model.holidays import PERIODS
from tkg_pipeline.align import TZ, to_unix
from tkg_pipeline.config import OUTPUT_DIR
from tkg_pipeline.fimut import fimut_time_tilted, hour_occupancy_batch, start_close_counts, uniform_close_probability_many
from tkg_pipeline.imut import _from_unix

CLEANED = OUTPUT_DIR / "cleaned_crime_records.csv"
REPORT_DIR = OUTPUT_DIR / "tkg" / "fimut"
YEARS = (2017, 2018, 2019)
PROFILE_OF = {2017: 2016, 2018: 2017, 2019: 2018}
PRECISE_GRAINS = ("hour", "minute", "second")
MAX_PRECISE_WIDTH = 2 * 3600.0
SMOOTHING = 0.5
MIN_CITY = 30
BOOTSTRAP = 2000
PIT_BINS = 10
TASKS = ("day", "stated_period", "asym_3_9")
ASYM_BEFORE = 3 * 3600.0
ASYM_AFTER = 9 * 3600.0
TAUS = (86400.0, 7 * 86400.0, 30 * 86400.0)
TAU_NAME = {86400.0: "1d", 7 * 86400.0: "7d", 30 * 86400.0: "30d"}
SHORT_WIDTH = 2 * 3600.0
LONG_WIDTH = 10 * 86400.0
MIN_KNOX_EVENTS = 20
MIN_KNOX_PAIRS = 30
MIN_SUB_PAIRS = 30
MIN_MACRO_EXPECTED = 5.0
AUDIT_PLANS = 400
OFFSET = 8 * 3600.0
PERIOD_START = np.array([item[0] for item in PERIODS], dtype=np.int64)
PERIOD_END = np.array([item[1] + 1 for item in PERIODS], dtype=np.int64)


def _log(message: str) -> None:
    print(message, flush=True)


def _ci(samples) -> list[float]:
    arr = np.asarray(samples, dtype=float)
    return [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return round(float(value), 6)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _cuts() -> dict[int, tuple[float, float]]:
    bounds = {year: float(to_unix(datetime(year, 1, 1, tzinfo=TZ))) for year in range(2016, 2021)}
    return {year: (bounds[year], bounds[year + 1]) for year in range(2016, 2020)}


def _load() -> pd.DataFrame:
    frame = pd.read_csv(
        CLEANED,
        usecols=["city_code", "grid_code", "time_grain", "time_start_unix", "time_end_unix"],
    )
    frame = frame.dropna(subset=["city_code", "time_start_unix", "time_end_unix", "time_grain"]).copy()
    frame = frame[frame["time_end_unix"] >= frame["time_start_unix"]]
    frame["time_start_unix"] = frame["time_start_unix"].astype(float)
    frame["time_end_unix"] = frame["time_end_unix"].astype(float)
    return frame


def _year_id(starts: np.ndarray, cuts: dict[int, tuple[float, float]]) -> np.ndarray:
    years = np.zeros(len(starts), dtype=np.int64)
    for year, (left, right) in cuts.items():
        years[(starts >= left) & (starts < right)] = year
    return years


def _profile(hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(hours.astype(np.int64), minlength=24).astype(float)
    return counts, counts + SMOOTHING


def _truth_hour(mid: np.ndarray) -> np.ndarray:
    return np.floor((mid + OFFSET) / 3600.0).astype(np.int64) % 24


def _windows(mid: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
    local = mid + OFFSET
    day0 = np.floor(local / 86400.0) * 86400.0
    if task == "day":
        return day0 - OFFSET, day0 + 86400.0 - OFFSET
    if task == "asym_3_9":
        return mid - ASYM_BEFORE, mid + ASYM_AFTER
    if task != "stated_period":
        raise KeyError(task)
    truth = _truth_hour(mid)
    slot = np.searchsorted(PERIOD_START, truth, side="right") - 1
    if np.any(slot < 0) or np.any(truth < PERIOD_START[slot]) or np.any(truth >= PERIOD_END[slot]):
        raise RuntimeError("时段边界没有覆盖全部小时")
    return day0 + PERIOD_START[slot] * 3600.0 - OFFSET, day0 + PERIOD_END[slot] * 3600.0 - OFFSET


def _losses(prob: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    picked = prob[np.arange(len(truth)), truth]
    brier = np.sum(prob * prob, axis=1) - 2.0 * picked + 1.0
    nll = -np.log(np.maximum(picked, 1e-300))
    return brier, nll, picked


def _dirac_brier(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.where(pred == truth, 0.0, 2.0)


def _circular_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    gap = np.abs(pred.astype(float) - truth.astype(float))
    return np.minimum(gap, 24.0 - gap)


def _distribution_mae(prob: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = 2.0 * np.pi * np.arange(24) / 24.0
    sin_mean = prob @ np.sin(theta)
    cos_mean = prob @ np.cos(theta)
    length = np.sqrt(sin_mean * sin_mean + cos_mean * cos_mean)
    angle = np.arctan2(sin_mean, cos_mean) % (2.0 * np.pi)
    return _circular_error(angle / (2.0 * np.pi) * 24.0, truth), length


def _pit(prob: np.ndarray, truth: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(prob, axis=1)
    reached = cumulative[np.arange(len(truth)), truth]
    mass = prob[np.arange(len(truth)), truth]
    return (reached - mass) + uniforms * mass


def _tv(histogram: np.ndarray) -> float:
    total = float(histogram.sum())
    if total <= 0:
        return float("nan")
    frequency = histogram / total
    return float(0.5 * np.abs(frequency - 1.0 / PIT_BINS).sum())


def _bootstrap_sum(stat: np.ndarray, count: np.ndarray, rng: np.random.Generator) -> dict:
    means = stat / count
    index = np.arange(len(count))
    micro, macro = [], []
    for _ in range(BOOTSTRAP):
        take = rng.choice(index, size=len(index), replace=True)
        micro.append(stat[take].sum() / count[take].sum())
        macro.append(means[take].mean())
    return {
        "micro": float(stat.sum() / count.sum()),
        "micro_ci": _ci(micro),
        "macro": float(means.mean()),
        "macro_ci": _ci(macro),
    }


def _bootstrap_diff(left: np.ndarray, right: np.ndarray, count: np.ndarray, rng: np.random.Generator) -> dict:
    diff = left - right
    summary = _bootstrap_sum(diff, count, rng)
    city = diff / count
    summary["cities_left_lower"] = int(np.sum(city < -1e-12))
    summary["cities_right_lower"] = int(np.sum(city > 1e-12))
    summary["cities_tie"] = int(np.sum(np.abs(city) <= 1e-12))
    return summary


def _eligible(table: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keep = table["count"] >= MIN_CITY
    return {key: value[keep] for key, value in table.items() if key != "code"}


def _city_sums(city: np.ndarray, columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    inverse, _uniques = pd.factorize(np.asarray(city), sort=False)
    count = np.bincount(inverse).astype(float)
    packed = {"count": count}
    for name, values in columns.items():
        column = np.asarray(values, dtype=float)
        if column.shape != inverse.shape:
            raise ValueError(f"{name} 的形状是 {column.shape}，城市索引是 {inverse.shape}")
        packed[name] = np.bincount(inverse, weights=column)
    return packed


def _pit_summary(pit: np.ndarray, city: np.ndarray, rng: np.random.Generator) -> dict | None:
    inverse, _uniques = pd.factorize(np.asarray(city), sort=False)
    count = np.bincount(inverse)
    keep = np.flatnonzero(count >= MIN_CITY)
    if len(keep) < 2:
        return None
    remap = -np.ones(len(count), dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    chosen = count[inverse] >= MIN_CITY
    compact = remap[inverse[chosen]]
    bins = np.minimum((pit[chosen] * PIT_BINS).astype(np.int64), PIT_BINS - 1)
    histogram = np.zeros((len(keep), PIT_BINS), dtype=np.int64)
    np.add.at(histogram, (compact, bins), 1)
    point = _tv(histogram.sum(axis=0))
    index = np.arange(len(keep))
    draws = np.empty(BOOTSTRAP)
    for draw in range(BOOTSTRAP):
        take = rng.choice(index, size=len(index), replace=True)
        draws[draw] = _tv(histogram[take].sum(axis=0))
    return {"micro": point, "micro_ci": _ci(draws)}


def _model_block(stat_sum: np.ndarray, count: np.ndarray, pooled: float, rng: np.random.Generator, *, nll: bool, extra: dict) -> dict:
    if len(count) < 2:
        block = {"events": int(count.sum()) if len(count) else 0, "cities": int(len(count)), "brier": None, "pooled_brier_including_small_cities": pooled}
    else:
        summary = _bootstrap_sum(stat_sum, count, rng)
        block = {
            "events": int(count.sum()),
            "cities": int(len(count)),
            "brier": summary["micro"],
            "brier_ci": summary["micro_ci"],
            "macro_brier": summary["macro"],
            "macro_brier_ci": summary["macro_ci"],
            "pooled_brier_including_small_cities": pooled,
        }
    if nll:
        block["nll"] = extra.get("nll")
        block["nll_ci"] = extra.get("nll_ci")
        block["macro_nll"] = extra.get("macro_nll")
        block["macro_nll_ci"] = extra.get("macro_nll_ci")
    else:
        block["nll"] = None
    for key in ("mode_hit", "circular_mae", "resultant_length", "min_truth_probability", "pit_tv", "pit_tv_ci"):
        block[key] = extra.get(key)
    return block


def _score_population(start, end, city, lam: np.ndarray, task: str, rng: np.random.Generator, pit_uniforms: np.ndarray) -> dict:
    mid = (start + end) / 2.0
    truth = _truth_hour(mid)
    obs_start, obs_end = _windows(mid, task)
    uniform = hour_occupancy_batch(obs_start, obs_end, None)
    tilt = hour_occupancy_batch(obs_start, obs_end, lam)
    shifted = hour_occupancy_batch(obs_start, obs_end, np.roll(lam, 12))
    climate = np.repeat((lam / lam.sum()).reshape(1, 24), len(truth), axis=0)
    if task == "day" and np.max(np.abs(tilt - climate)) > 1e-8:
        raise RuntimeError("整日窗口上，小时倾斜与不分区间的气候态应当相同")
    packed = {
        "uniform": uniform,
        "tilt": tilt,
        "tilt_shift12": shifted,
        "climatology": climate,
    }
    losses = {name: _losses(prob, truth) for name, prob in packed.items()}
    if np.any(losses["uniform"][2] <= 0):
        raise RuntimeError("均匀占据没有覆盖潜在小时，窗口构造有误")
    start_hour = _truth_hour(obs_start)
    mid_hour = _truth_hour((obs_start + obs_end) / 2.0)
    dirac = {
        "start": (_dirac_brier(start_hour, truth), start_hour),
        "mid": (_dirac_brier(mid_hour, truth), mid_hour),
    }
    columns = {f"{name}_brier": values[0] for name, values in losses.items()}
    columns.update({f"{name}_nll": values[1] for name, values in losses.items()})
    columns["tilt_hit"] = (np.argmax(tilt, axis=1) == truth).astype(float)
    columns["tilt_shift12_hit"] = (np.argmax(shifted, axis=1) == truth).astype(float)
    columns["start_brier"] = dirac["start"][0]
    columns["mid_brier"] = dirac["mid"][0]
    columns["start_hit"] = (start_hour == truth).astype(float)
    columns["mid_hit"] = (mid_hour == truth).astype(float)
    table = _eligible(_city_sums(city, columns))
    models = {}
    for name, (brier, nll, picked) in losses.items():
        nll_summary = _bootstrap_sum(table[f"{name}_nll"], table["count"], rng) if len(table["count"]) >= 2 else None
        mae, length = _distribution_mae(packed[name], truth)
        stable = float(length.mean()) >= 0.05
        pit_summary = _pit_summary(_pit(packed[name], truth, pit_uniforms), city, rng)
        models[name] = _model_block(
            table[f"{name}_brier"],
            table["count"],
            float(brier.mean()),
            rng,
            nll=True,
            extra={
                "nll": None if nll_summary is None else nll_summary["micro"],
                "nll_ci": None if nll_summary is None else nll_summary["micro_ci"],
                "macro_nll": None if nll_summary is None else nll_summary["macro"],
                "macro_nll_ci": None if nll_summary is None else nll_summary["macro_ci"],
                "mode_hit": float(table[f"{name}_hit"].sum() / table["count"].sum()) if name in {"tilt", "tilt_shift12"} and len(table["count"]) else None,
                "circular_mae": float(mae.mean()) if stable else None,
                "resultant_length": float(length.mean()),
                "min_truth_probability": float(picked.min()),
                "pit_tv": None if pit_summary is None else pit_summary["micro"],
                "pit_tv_ci": None if pit_summary is None else pit_summary["micro_ci"],
            },
        )
    for name, (brier, pred) in dirac.items():
        models[name] = _model_block(
            table[f"{name}_brier"],
            table["count"],
            float(brier.mean()),
            rng,
            nll=False,
            extra={
                "mode_hit": float(table[f"{name}_hit"].sum() / table["count"].sum()) if len(table["count"]) else None,
                "circular_mae": float(_circular_error(pred, truth).mean()),
                "resultant_length": 1.0,
                "min_truth_probability": None,
                "pit_tv": None,
                "pit_tv_ci": None,
            },
        )
    paired = {}
    if len(table["count"]) >= 2:
        comparisons = (
            ("tilt_minus_uniform", "tilt_brier", "uniform_brier"),
            ("tilt_minus_shift12", "tilt_brier", "tilt_shift12_brier"),
            ("tilt_minus_climatology", "tilt_brier", "climatology_brier"),
            ("uniform_minus_start", "uniform_brier", "start_brier"),
            ("uniform_minus_mid", "uniform_brier", "mid_brier"),
            ("tilt_minus_uniform_nll", "tilt_nll", "uniform_nll"),
            ("tilt_minus_shift12_nll", "tilt_nll", "tilt_shift12_nll"),
            ("tilt_minus_climatology_nll", "tilt_nll", "climatology_nll"),
        )
        for label, left, right in comparisons:
            paired[label] = _bootstrap_diff(table[left], table[right], table["count"], rng)
    return {"events_input": int(len(truth)), "models": models, "paired_brier_and_nll": paired}


def _entropy(counts: np.ndarray) -> float:
    probability = counts / counts.sum()
    probability = probability[probability > 0]
    return float(-(probability * np.log(probability)).sum())


def run_latent(frame: pd.DataFrame, cuts: dict[int, tuple[float, float]]) -> tuple[dict, dict[int, np.ndarray]]:
    width = frame["time_end_unix"].to_numpy(dtype=float) - frame["time_start_unix"].to_numpy(dtype=float)
    precise = frame["time_grain"].isin(PRECISE_GRAINS).to_numpy() & (width <= MAX_PRECISE_WIDTH) & (width >= 0)
    start = frame["time_start_unix"].to_numpy(dtype=float)[precise]
    end = frame["time_end_unix"].to_numpy(dtype=float)[precise]
    city = frame["city_code"].to_numpy()[precise]
    grain = frame["time_grain"].to_numpy()[precise]
    years = _year_id(start, cuts)
    hours = _truth_hour((start + end) / 2.0)
    profiles = {}
    descriptive = {}
    for year in range(2016, 2020):
        chosen = years == year
        counts, smoothed = _profile(hours[chosen])
        profiles[year] = smoothed
        descriptive[str(year)] = {
            "events": int(chosen.sum()),
            "counts": [int(value) for value in counts],
            "entropy": _entropy(counts) if counts.sum() else None,
            "uniform_entropy": float(np.log(24)),
        }
    transport = {}
    for left, right in ((2016, 2017), (2017, 2018), (2018, 2019)):
        previous = np.array(descriptive[str(left)]["counts"], dtype=float)
        current = np.array(descriptive[str(right)]["counts"], dtype=float)
        share_left = previous / previous.sum()
        share_right = current / current.sum()
        transport[f"{left}_{right}"] = float(0.5 * np.abs(share_left - share_right).sum())
    populations = {
        "all_precise": np.ones(len(start), dtype=bool),
        "minute_second": np.isin(grain, ("minute", "second")),
    }
    scored = {}
    for population, mask in populations.items():
        scored[population] = {}
        for year in YEARS:
            lam = profiles[PROFILE_OF[year]]
            chosen = mask & (years == year)
            _log(f"潜在时刻 {population} {year} 事件 {int(chosen.sum())}")
            scored[population][str(year)] = {}
            for task in TASKS:
                pit_uniforms = np.random.default_rng(0).random(int(chosen.sum()))
                rng = np.random.default_rng(1)
                block = _score_population(start[chosen], end[chosen], city[chosen], lam, task, rng, pit_uniforms)
                scored[population][str(year)][task] = block
                tilt = block["models"]["tilt"]["brier"]
                uniform = block["models"]["uniform"]["brier"]
                _log(f"  {task} Brier 倾斜 {tilt} 均匀 {uniform}")
    return {"profiles": descriptive, "profile_tv": transport, "scores": scored}, profiles


def run_audit(frame: pd.DataFrame, cuts: dict[int, tuple[float, float]], profiles: dict[int, np.ndarray]) -> dict:
    width = frame["time_end_unix"].to_numpy(dtype=float) - frame["time_start_unix"].to_numpy(dtype=float)
    precise = frame["time_grain"].isin(PRECISE_GRAINS).to_numpy() & (width <= MAX_PRECISE_WIDTH)
    start = frame["time_start_unix"].to_numpy(dtype=float)[precise]
    end = frame["time_end_unix"].to_numpy(dtype=float)[precise]
    years = _year_id(start, cuts)
    chosen = np.flatnonzero(years == 2019)
    rng = np.random.default_rng(0)
    take = rng.choice(chosen, size=min(AUDIT_PLANS, len(chosen)), replace=False)
    lam = profiles[2018]
    level_gap = 0.0
    day_gap = 0.0
    hour_gap = 0.0
    for index in take:
        mid = (start[index] + end[index]) / 2.0
        obs_start, obs_end = _windows(np.array([mid]), "day")
        plan = fimut_time_tilted(_from_unix(float(obs_start[0])), _from_unix(float(obs_end[0])), lam)
        for level in ("小时", "日", "周", "旬", "月"):
            level_gap = max(level_gap, abs(plan.level_total(level) - plan.retained))
        for entity_id, mass in plan.masses("日").items():
            prefix = entity_id[2:]
            hour_sum = sum(value for key, value in plan.masses("小时").items() if key[1:9] == prefix)
            day_gap = max(day_gap, abs(hour_sum - mass))
        reconstructed = np.zeros(24)
        for atom in plan.atoms:
            if atom.level == "小时" and plan.retained > 0:
                reconstructed[int(atom.entity_id[-2:])] += atom.mass / plan.retained
        batch = hour_occupancy_batch(obs_start, obs_end, lam)[0]
        hour_gap = max(hour_gap, float(np.max(np.abs(reconstructed - batch))))
    return {
        "year": 2019,
        "profile_year": 2018,
        "plans": int(len(take)),
        "window": "day",
        "max_level_residual": level_gap,
        "max_day_pushforward_gap": day_gap,
        "max_hour_probability_gap": hour_gap,
    }


def _pair_batches(count: int, block: int = 700):
    if count <= 6000:
        yield np.triu_indices(count, 1)
        return
    for origin in range(0, count, block):
        stop = min(count, origin + block)
        left = np.arange(origin, stop)
        right = np.arange(origin, count)
        grid_i, grid_j = np.meshgrid(left, right, indexing="ij")
        mask = grid_j > grid_i
        yield grid_i[mask], grid_j[mask]


def _same_grid_stats(start: np.ndarray, end: np.ndarray, grid: np.ndarray) -> dict:
    order = np.argsort(grid, kind="mergesort")
    grid = grid[order]
    start = start[order]
    end = end[order]
    if len(grid) == 0:
        empty = np.zeros(len(TAUS))
        return {"pairs": 0, "point": empty, "soft": empty.copy(), "possible": empty.copy(), "short_pairs": 0, "long_pairs": 0, "short_point": empty.copy(), "short_soft": empty.copy(), "long_point": empty.copy(), "long_soft": empty.copy()}
    change = np.flatnonzero(grid[1:] != grid[:-1]) + 1
    bounds = np.r_[0, change, len(grid)]
    point = np.zeros(len(TAUS))
    soft = np.zeros(len(TAUS))
    possible = np.zeros(len(TAUS))
    short_point = np.zeros(len(TAUS))
    short_soft = np.zeros(len(TAUS))
    long_point = np.zeros(len(TAUS))
    long_soft = np.zeros(len(TAUS))
    pairs = 0
    short_pairs = 0
    long_pairs = 0
    for left, right in zip(bounds[:-1], bounds[1:]):
        size = right - left
        if size < 2:
            continue
        left_i, right_i = np.triu_indices(size, 1)
        a = start[left:right][left_i]
        b = end[left:right][left_i]
        c = start[left:right][right_i]
        d = end[left:right][right_i]
        pairs += len(a)
        gap = np.maximum(np.maximum(a, c) - np.minimum(b, d), 0.0)
        short = ((b - a) <= SHORT_WIDTH) & ((d - c) <= SHORT_WIDTH)
        long = ((b - a) >= LONG_WIDTH) | ((d - c) >= LONG_WIDTH)
        short_pairs += int(short.sum())
        long_pairs += int(long.sum())
        for index, tau in enumerate(TAUS):
            close = np.abs(a - c) <= tau
            maybe = gap <= tau
            weight = uniform_close_probability_many(a, b, c, d, tau)
            point[index] += close.sum()
            possible[index] += maybe.sum()
            soft[index] += weight.sum()
            short_point[index] += close[short].sum()
            short_soft[index] += weight[short].sum()
            long_point[index] += close[long].sum()
            long_soft[index] += weight[long].sum()
    return {
        "pairs": pairs,
        "point": point,
        "soft": soft,
        "possible": possible,
        "short_pairs": short_pairs,
        "long_pairs": long_pairs,
        "short_point": short_point,
        "short_soft": short_soft,
        "long_point": long_point,
        "long_soft": long_soft,
    }


def _all_pair_stats(start: np.ndarray, end: np.ndarray) -> tuple[list[int], np.ndarray, np.ndarray]:
    point = start_close_counts(start, TAUS)
    soft = np.zeros(len(TAUS))
    possible = np.zeros(len(TAUS))
    scanned = np.zeros(len(TAUS), dtype=np.int64)
    for left_i, right_i in _pair_batches(len(start)):
        a = start[left_i]
        b = end[left_i]
        c = start[right_i]
        d = end[right_i]
        gap = np.maximum(np.maximum(a, c) - np.minimum(b, d), 0.0)
        separation = np.abs(a - c)
        for index, tau in enumerate(TAUS):
            scanned[index] += int(np.count_nonzero(separation <= tau))
            possible[index] += float(np.count_nonzero(gap <= tau))
            soft[index] += float(uniform_close_probability_many(a, b, c, d, tau).sum())
    for index, tau in enumerate(TAUS):
        if int(scanned[index]) != int(point[index]):
            raise RuntimeError(f"起点接近对的两种数法不一致：τ={tau} 扫描 {scanned[index]} 排序 {point[index]}")
    return point, soft, possible


def _ratio_summary(obs: np.ndarray, exp: np.ndarray, rng: np.random.Generator) -> dict:
    point = float(obs.sum() / exp.sum()) if exp.sum() > 0 else float("nan")
    index = np.arange(len(obs))
    draws = []
    for _ in range(BOOTSTRAP):
        take = rng.choice(index, size=len(index), replace=True)
        denom = exp[take].sum()
        draws.append(obs[take].sum() / denom if denom > 0 else np.nan)
    draws = np.asarray(draws, dtype=float)
    finite = draws[np.isfinite(draws)]
    macro_keep = exp >= MIN_MACRO_EXPECTED
    macro = None
    macro_ci = None
    if int(macro_keep.sum()) >= 2:
        city_ratio = obs[macro_keep] / exp[macro_keep]
        macro_index = np.arange(len(city_ratio))
        macro_draws = []
        for _ in range(BOOTSTRAP):
            take = rng.choice(macro_index, size=len(macro_index), replace=True)
            macro_draws.append(city_ratio[take].mean())
        macro = float(city_ratio.mean())
        macro_ci = _ci(macro_draws)
    return {
        "micro": point,
        "micro_ci": _ci(finite) if len(finite) else None,
        "macro_expected_at_least_5": macro,
        "macro_ci": macro_ci,
        "cities_in_macro": int(macro_keep.sum()),
    }


def run_knox(frame: pd.DataFrame, cuts: dict[int, tuple[float, float]]) -> tuple[dict, list[dict]]:
    width = frame["time_end_unix"].to_numpy(dtype=float) - frame["time_start_unix"].to_numpy(dtype=float)
    start_all = frame["time_start_unix"].to_numpy(dtype=float)
    end_all = frame["time_end_unix"].to_numpy(dtype=float)
    city_all = frame["city_code"].to_numpy()
    grid_all = frame["grid_code"].astype(str).to_numpy()
    years = _year_id(start_all, cuts)
    usable = frame["grid_code"].notna().to_numpy() & ~np.isin(grid_all, ("", "nan", "None", "NaN"))
    rows: list[dict] = []
    summaries = {}
    for year in YEARS:
        chosen = np.flatnonzero(usable & (years == year) & (width >= 0))
        city = city_all[chosen]
        order = np.argsort(city, kind="mergesort")
        city = city[order]
        start = start_all[chosen][order]
        end = end_all[chosen][order]
        grid = grid_all[chosen][order]
        change = np.flatnonzero(city[1:] != city[:-1]) + 1 if len(city) else np.array([], dtype=int)
        bounds = np.r_[0, change, len(city)]
        kept = 0
        _log(f"Knox {year} 有网格的事件 {len(city)}")
        for left, right in zip(bounds[:-1], bounds[1:]):
            size = right - left
            if size < MIN_KNOX_EVENTS:
                continue
            grid_city = grid[left:right]
            _unique, counts = np.unique(grid_city, return_counts=True)
            same_pairs = int(np.sum(counts * (counts - 1) // 2))
            if same_pairs < MIN_KNOX_PAIRS:
                continue
            stats = _same_grid_stats(start[left:right], end[left:right], grid_city)
            if int(stats["pairs"]) != same_pairs:
                raise RuntimeError(f"同网格对数不一致：{stats['pairs']} 与 {same_pairs}")
            point_all, soft_all, possible_all = _all_pair_stats(start[left:right], end[left:right])
            choose = size * (size - 1) / 2.0
            record = {
                "year": year,
                "city": city[left],
                "events": size,
                "pairs": int(stats["pairs"]),
                "short_pairs": int(stats["short_pairs"]),
                "long_pairs": int(stats["long_pairs"]),
            }
            for index, tau in enumerate(TAUS):
                name = TAU_NAME[tau]
                record[f"{name}_point_obs"] = float(stats["point"][index])
                record[f"{name}_point_exp"] = float(same_pairs * point_all[index] / choose)
                record[f"{name}_soft_obs"] = float(stats["soft"][index])
                record[f"{name}_soft_exp"] = float(same_pairs * soft_all[index] / choose)
                record[f"{name}_possible_obs"] = float(stats["possible"][index])
                record[f"{name}_possible_exp"] = float(same_pairs * possible_all[index] / choose)
                record[f"{name}_short_point"] = float(stats["short_point"][index])
                record[f"{name}_short_soft"] = float(stats["short_soft"][index])
                record[f"{name}_long_point"] = float(stats["long_point"][index])
                record[f"{name}_long_soft"] = float(stats["long_soft"][index])
            rows.append(record)
            kept += 1
            if kept % 40 == 0:
                _log(f"  已完成 {kept} 个城市")
        _log(f"  {year} 纳入城市 {kept}")
        year_rows = [row for row in rows if row["year"] == year]
        summaries[str(year)] = _knox_year(year_rows)
    return summaries, rows


def _knox_year(rows: list[dict]) -> dict:
    if not rows:
        return {"cities": 0}
    rng = np.random.default_rng(1)
    out = {"cities": len(rows), "events": int(sum(row["events"] for row in rows)), "pairs": int(sum(row["pairs"] for row in rows))}
    for tau in TAUS:
        name = TAU_NAME[tau]
        obs_point = np.array([row[f"{name}_point_obs"] for row in rows])
        exp_point = np.array([row[f"{name}_point_exp"] for row in rows])
        obs_soft = np.array([row[f"{name}_soft_obs"] for row in rows])
        exp_soft = np.array([row[f"{name}_soft_exp"] for row in rows])
        obs_possible = np.array([row[f"{name}_possible_obs"] for row in rows])
        exp_possible = np.array([row[f"{name}_possible_exp"] for row in rows])
        point_ratio = _ratio_summary(obs_point, exp_point, rng)
        soft_ratio = _ratio_summary(obs_soft, exp_soft, rng)
        possible_ratio = _ratio_summary(obs_possible, exp_possible, rng)
        gap_draws = []
        index = np.arange(len(rows))
        for _ in range(BOOTSTRAP):
            take = rng.choice(index, size=len(index), replace=True)
            soft_den = exp_soft[take].sum()
            point_den = exp_point[take].sum()
            if soft_den <= 0 or point_den <= 0:
                continue
            gap_draws.append(obs_soft[take].sum() / soft_den - obs_point[take].sum() / point_den)
        out[name] = {
            "mean_point": float(obs_point.sum() / sum(row["pairs"] for row in rows)),
            "mean_soft": float(obs_soft.sum() / sum(row["pairs"] for row in rows)),
            "mean_possible": float(obs_possible.sum() / sum(row["pairs"] for row in rows)),
            "point_ratio": point_ratio,
            "soft_ratio": soft_ratio,
            "possible_ratio": possible_ratio,
            "soft_minus_point_ratio": {
                "micro": soft_ratio["micro"] - point_ratio["micro"],
                "micro_ci": _ci(gap_draws) if gap_draws else None,
            },
            "subgroups": {
                "both_width_at_most_2h": _subgroup(rows, name, "short", rng),
                "either_width_at_least_10d": _subgroup(rows, name, "long", rng),
            },
        }
    return out


def _subgroup(rows: list[dict], tau_name: str, kind: str, rng: np.random.Generator) -> dict:
    eligible = [row for row in rows if row[f"{kind}_pairs"] >= MIN_SUB_PAIRS]
    pairs = np.array([row[f"{kind}_pairs"] for row in eligible], dtype=float)
    point = np.array([row[f"{tau_name}_{kind}_point"] for row in eligible], dtype=float)
    soft = np.array([row[f"{tau_name}_{kind}_soft"] for row in eligible], dtype=float)
    all_pairs = float(sum(row[f"{kind}_pairs"] for row in rows))
    all_point = float(sum(row[f"{tau_name}_{kind}_point"] for row in rows))
    all_soft = float(sum(row[f"{tau_name}_{kind}_soft"] for row in rows))
    block = {
        "cities": len(eligible),
        "pairs_in_eligible_cities": int(pairs.sum()) if len(eligible) else 0,
        "pairs_all_cities": int(all_pairs),
        "pooled_mean_point": all_point / all_pairs if all_pairs else None,
        "pooled_mean_soft": all_soft / all_pairs if all_pairs else None,
    }
    if len(eligible) < 2 or pairs.sum() <= 0:
        block["soft_minus_point"] = None
        return block
    diff = _bootstrap_diff(soft, point, pairs, rng)
    block["mean_point"] = float(point.sum() / pairs.sum())
    block["mean_soft"] = float(soft.sum() / pairs.sum())
    block["soft_minus_point"] = diff
    return block


def _protocol() -> dict:
    return {
        "latent_time": "精确粒度且区间宽度不超过 2 小时的记录。标签是区间中点所在的本地小时，不是叙述里的代表钟点。",
        "profile": "上一日历年的全部精确记录，小时计数加 0.5，相当于 24 维 Dirichlet(1/2) 先验下的后验均值。不在测试年上估计，也不按测试子集重估。",
        "tasks": {
            "day": "中点所在公历日。整日对齐时，倾斜与不分区间的气候态相同。",
            "stated_period": "项目书时段：凌晨、早晨、上午、中午、下午、傍晚、夜间。观测是中点所在的那一段。",
            "asym_3_9": "固定为中点前 3 小时到后 9 小时。窗口中点比潜在小时晚 3 小时，中点狄拉克和起点狄拉克的命中率应为 0，只用于核对实现。",
        },
        "scores": "Brier 是主指标，越低越好。对数分数只用于在潜在小时上有正概率的模型。随机化 PIT 使用每个任务独立的 default_rng(0)，城市自助法使用 default_rng(1)。",
        "bootstrap": f"有至少 {MIN_CITY} 条测试记录的城市，放回抽样 {BOOTSTRAP} 次。micro 按事件加权，macro 按城市等权。",
        "negative_control": "把小时强度在钟面上旋转 12 小时。",
        "knox": "同一 500 米网格、同一日历年内的无序事件对。零假设是城市内随机重排区间：期望 = 同网格对数 × 全市该时间权重之和 / 城市内全部对数。空间带不含四邻。",
        "knox_subgroups": "两边宽度都不超过 2 小时；或者至少一边不少于 10 天。阈值不在结果上看完后再改。",
        "filtration_audit": f"2019 年精确记录中抽 {AUDIT_PLANS} 条，用 2018 年强度做整日计划，检查各层质量和、日等于其小时之和、以及与批量占据概率一致。种子 0。",
        "not_changed": "近重复网格排序仍以区间是否与查询窗口相交计分。不重建 sqlite 或 Neo4j。",
        "citations": [
            "Knox, E. G. (1964). The detection of space-time interactions. Applied Statistics, 13(1), 25–29. DOI 10.2307/2985220.",
            "小时倾斜后的粗尺度是推前，不是分层各自再加权。均匀占据是强度恒为 1 的特例。",
        ],
    }


def main() -> None:
    started = time.perf_counter()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = _load()
    cuts = _cuts()
    _log(f"读入 {len(frame)} 行")
    latent, profiles = run_latent(frame, cuts)
    audit = run_audit(frame, cuts, profiles)
    _log(
        "滤子核对 "
        f"层 {audit['max_level_residual']:.3e} "
        f"日 {audit['max_day_pushforward_gap']:.3e} "
        f"小时 {audit['max_hour_probability_gap']:.3e}"
    )
    report = {"protocol": _protocol(), "latent": latent, "filtration_audit": audit}
    path = REPORT_DIR / "measure_report.json"
    path.write_text(json.dumps(_clean(report), ensure_ascii=False, indent=2), encoding="utf-8")
    knox, rows = run_knox(frame, cuts)
    report["knox"] = knox
    report["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    path.write_text(json.dumps(_clean(report), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(REPORT_DIR / "measure_knox_cities.csv", index=False, encoding="utf-8-sig")
    _log(f"写出 {path}")
    _log(f"用时 {report['elapsed_seconds']} 秒")


if __name__ == "__main__":
    main()
