"""三条对齐模块的八组消融。

用法：py -3 run_tkg_ablation.py

读者、宽度带和开关在运行前写定。结果出来之后不改这些常数，
也不按 Hits@10 去重配三个模块的权重。
近重复和隐藏钟点的数字从已经跑完的报告里原样抄入，不再重估。
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from tkg_pipeline.align import finest_time_entity
from tkg_pipeline.config import OUTPUT_DIR
from tkg_pipeline.imut import _from_unix, selected_bins
from tkg_pipeline.triad import (
    CONFIGS,
    DAY_L,
    HOUR_L,
    LEVEL_NAME,
    MAX_EXPAND,
    MONTH_L,
    NONE,
    TEN_L,
    WEEK_L,
    absolute_errors,
    build_geometry,
)

CLEANED = OUTPUT_DIR / "cleaned_crime_records.csv"
REPORT_DIR = OUTPUT_DIR / "tkg" / "fimut"
EVAL_PATH = REPORT_DIR / "eval_report.json"
MEASURE_PATH = REPORT_DIR / "measure_report.json"
BOOTSTRAP = 2000
MIN_CITY = 30
AUDIT_ROWS = 400
READERS = ("hour", "day", "month", "window6", "joint")
DROPS = (
    ("drop_adapted", "011"),
    ("drop_filtration", "101"),
    ("drop_circadian", "110"),
)


def _log(message: str) -> None:
    print(message, flush=True)


def _ci(samples: np.ndarray) -> list[float]:
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


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


def _load() -> pd.DataFrame:
    frame = pd.read_csv(
        CLEANED,
        usecols=["city_code", "time_grain", "time_start_unix", "time_end_unix"],
    )
    frame = frame.dropna(subset=["time_start_unix", "time_end_unix"]).copy()
    frame = frame[frame["time_end_unix"] >= frame["time_start_unix"]]
    frame["city_code"] = frame["city_code"].astype("string").fillna("missing")
    return frame


def _audit(start: np.ndarray, end: np.ndarray, geo) -> int:
    rng = np.random.default_rng(0)
    size = min(AUDIT_ROWS, len(start))
    pick = rng.choice(len(start), size=size, replace=False)
    for index in pick:
        left = _from_unix(int(start[index]))
        right = _from_unix(int(end[index]))
        _entity, level = finest_time_entity(left, right)
        ladder = selected_bins(left, right)[0][1]
        got_hard = LEVEL_NAME[int(geo.hard[index])]
        got_ladder = LEVEL_NAME[int(geo.ladder[index])]
        if got_hard != level or got_ladder != ladder:
            raise AssertionError(
                f"格子不一致：起点 {int(start[index])} 终点 {int(end[index])}，"
                f"硬格子 {got_hard}/{level}，阶梯 {got_ladder}/{ladder}"
            )
    return int(size)


def _metric_matrix(errors: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    packed: dict[str, dict[str, np.ndarray]] = {}
    for name, _adapted, _filtration, _circadian in CONFIGS:
        joint = (errors[name]["hour"] + errors[name]["day"] + errors[name]["month"]) / 3.0
        packed[name] = {
            "hour": errors[name]["hour"],
            "day": errors[name]["day"],
            "month": errors[name]["month"],
            "window6": errors[name]["window6"],
            "joint": joint,
        }
    return packed


def _city_sums(values: np.ndarray, codes: np.ndarray, n_city: int) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(n_city, dtype=np.float64)
    np.add.at(sums, codes, values)
    counts = np.bincount(codes, minlength=n_city).astype(np.float64)
    return sums, counts


def _bootstrap(sums: np.ndarray, counts: np.ndarray, draw: np.ndarray) -> dict[str, float | list[float] | int]:
    active = counts > 0
    if not np.any(active) or draw.size == 0:
        return {
            "mae": None,
            "mae_ci": None,
            "macro_mae": None,
            "macro_mae_ci": None,
            "events": int(counts[active].sum()) if np.any(active) else 0,
            "cities": int(active.sum()),
            "macro_cities": 0,
        }
    point = float(sums[active].sum() / counts[active].sum())
    eligible = counts >= MIN_CITY
    macro = float(np.mean(sums[eligible] / counts[eligible])) if np.any(eligible) else None
    total = sums[draw].sum(axis=1)
    weight = counts[draw].sum(axis=1)
    micro = total / np.maximum(weight, 1.0)
    if np.any(eligible):
        city_mae = np.full(len(sums), np.nan, dtype=np.float64)
        city_mae[eligible] = sums[eligible] / counts[eligible]
        macro_ci = _ci(np.nanmean(city_mae[draw], axis=1))
    else:
        macro_ci = None
    return {
        "mae": point,
        "mae_ci": _ci(micro),
        "macro_mae": macro,
        "macro_mae_ci": macro_ci,
        "events": int(counts[active].sum()),
        "cities": int(active.sum()),
        "macro_cities": int(eligible.sum()),
    }


def _draws(counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    active = np.flatnonzero(counts > 0)
    if len(active) == 0:
        return np.zeros((BOOTSTRAP, 0), dtype=np.int64)
    picked = rng.integers(0, len(active), size=(BOOTSTRAP, len(active)))
    return active[picked]


def _pack_metrics(packed, codes, n_city: int, mask: np.ndarray | None, rng: np.random.Generator) -> dict:
    if mask is None:
        use_codes = codes
    else:
        use_codes = codes[mask]
    counts = np.bincount(use_codes, minlength=n_city).astype(np.float64) if len(use_codes) else np.zeros(n_city)
    draw = _draws(counts, rng)
    out = {"events": int(len(use_codes)), "cities": int(np.count_nonzero(counts))}
    configs = {}
    for name, _a, _f, _c in CONFIGS:
        configs[name] = {}
        for reader in READERS:
            values = packed[name][reader] if mask is None else packed[name][reader][mask]
            sums, city_counts = _city_sums(values, use_codes, n_city)
            configs[name][reader] = _bootstrap(sums, city_counts, draw)
    out["configurations"] = configs
    deltas = {}
    for label, other in DROPS:
        deltas[label] = {}
        for reader in READERS:
            full = packed["111"][reader] if mask is None else packed["111"][reader][mask]
            dropped = packed[other][reader] if mask is None else packed[other][reader][mask]
            sums, city_counts = _city_sums(full - dropped, use_codes, n_city)
            stats = _bootstrap(sums, city_counts, draw)
            deltas[label][reader] = stats
    out["full_minus_ablation"] = deltas
    return out


def _best(block: dict, reader: str) -> list[str]:
    scores = {name: block["configurations"][name][reader]["mae"] for name, _a, _f, _c in CONFIGS}
    finite = {name: value for name, value in scores.items() if value is not None}
    lowest = min(finite.values())
    return [name for name, value in finite.items() if abs(value - lowest) <= 1e-8]


def _external() -> dict:
    evaluation = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    measure = json.loads(MEASURE_PATH.read_text(encoding="utf-8"))
    ranking = {}
    for year in ("2017", "2018", "2019"):
        ranking[year] = {}
        for window in ("7d", "30d", "90d"):
            cell = evaluation["near_repeat"][year][window]
            ranking[year][window] = {
                "events": cell["hard_overlap"]["events"],
                "cities": cell["hard_overlap"]["cities"],
                "start_hits10": cell["hard_start"]["hits_at_10"],
                "start_hits10_ci": cell["hard_start"]["hits_at_10_ci"],
                "incidence_hits10": cell["hard_overlap"]["hits_at_10"],
                "incidence_hits10_ci": cell["hard_overlap"]["hits_at_10_ci"],
                "occupancy_hits10": cell["fimut"]["hits_at_10"],
                "occupancy_hits10_ci": cell["fimut"]["hits_at_10_ci"],
            }
    hours = {}
    for year in ("2017", "2018", "2019"):
        day = measure["latent"]["scores"]["all_precise"][year]["day"]
        paired = day["paired_brier_and_nll"]
        hours[year] = {
            "events": day["models"]["tilt"]["events"],
            "cities": day["models"]["tilt"]["cities"],
            "start_brier": day["models"]["start"]["brier"],
            "start_brier_ci": day["models"]["start"]["brier_ci"],
            "uniform_brier": day["models"]["uniform"]["brier"],
            "tilt_brier": day["models"]["tilt"]["brier"],
            "tilt_brier_ci": day["models"]["tilt"]["brier_ci"],
            "tilt_minus_uniform": paired["tilt_minus_uniform"]["micro"],
            "tilt_minus_uniform_ci": paired["tilt_minus_uniform"]["micro_ci"],
            "tilt_minus_shift12": paired["tilt_minus_shift12"]["micro"],
            "tilt_minus_shift12_ci": paired["tilt_minus_shift12"]["micro_ci"],
            "uniform_pit_tv": day["models"]["uniform"]["pit_tv"],
            "tilt_pit_tv": day["models"]["tilt"]["pit_tv"],
        }
    return {
        "source": "eval_report.json 与 measure_report.json，本脚本不重算这两项",
        "ranking_note": "相交就计 1 只用模块一的支撑。模块二和模块三不改变这个示性函数，因此 Hits@10 与 111 相同，与 100 相同。关掉模块一后退回起点。占有率不是推荐的排序分数。",
        "hour_note": "隐藏整天后的钟点：起点是关掉模块一，均匀区间是打开模块一并关掉模块三，倾斜是两块都打开。模块二不改变小时条件分布。负的 Brier 差表示倾斜更好。",
        "near_repeat_hits10": ranking,
        "latent_hour_day_window": hours,
    }


def main() -> None:
    started = time.perf_counter()
    frame = _load()
    start = np.rint(frame["time_start_unix"].to_numpy(dtype=np.float64)).astype(np.int64)
    end = np.rint(frame["time_end_unix"].to_numpy(dtype=np.float64)).astype(np.int64)
    _log(f"记录 {len(frame)}")
    geo = build_geometry(start, end)
    checked = _audit(start, end, geo)
    _log(f"与硬对齐、阶梯核对 {checked} 条")
    _truth, errors = absolute_errors(geo)
    month_gap = float(errors["111"]["month"].max())
    hour_mask = (geo.ladder == HOUR_L) | (geo.width <= MAX_EXPAND)
    day_mask = np.isin(geo.ladder, (HOUR_L, DAY_L))
    hour_gap = float(errors["111"]["hour"][hour_mask].max()) if np.any(hour_mask) else 0.0
    day_gap = float(errors["111"]["day"][day_mask].max()) if np.any(day_mask) else 0.0
    if max(month_gap, hour_gap, day_gap) > 1e-8:
        raise AssertionError(f"全开配置没有回到长度比例：月 {month_gap} 小时 {hour_gap} 日 {day_gap}")
    packed = _metric_matrix(errors)
    codes, _uniques = pd.factorize(frame["city_code"].to_numpy(), sort=False)
    codes = np.asarray(codes, dtype=np.int64)
    n_city = int(codes.max()) + 1 if len(codes) else 0
    rng = np.random.default_rng(0)
    overall = _pack_metrics(packed, codes, n_city, None, rng)
    hard_names = {
        NONE: "none",
        HOUR_L: "hour",
        DAY_L: "day",
        WEEK_L: "week",
        TEN_L: "ten",
        MONTH_L: "month",
    }
    by_hard = {}
    for level, name in hard_names.items():
        mask = geo.hard == level
        by_hard[name] = _pack_metrics(packed, codes, n_city, mask, rng)
        _log(f"硬格子 {name} {int(mask.sum())}")
    bands = (
        ("le_2h", None, 2 * 3600),
        ("h2_to_36h", 2 * 3600, 36 * 3600),
        ("h36_to_45d", 36 * 3600, MAX_EXPAND),
        ("gt_45d", MAX_EXPAND, None),
    )
    by_width = {}
    for name, left, right in bands:
        mask = np.ones(len(geo.width), dtype=bool)
        if left is not None:
            mask &= geo.width > left
        if right is not None:
            mask &= geo.width <= right
        by_width[name] = _pack_metrics(packed, codes, n_city, mask, rng)
        _log(f"宽度 {name} {int(mask.sum())}")
    report = {
        "protocol": {
            "modules": {
                "adapted": "模块一，裁剪支撑。关闭时用包住全区间的最细硬格子，没有硬格子则用起点",
                "filtration": "模块二，只把质量向上推到更粗的格子。关闭时读者只能看见最细一层",
                "circadian": "模块三，宽度不超过 45 天时补出小时原子。本表的误差按长度计算，不乘小时强度",
            },
            "configurations": "三位二进制按支撑、推前、昼夜排列，111 为三块全开",
            "readers": {
                "hour": "只能读小时实体，查询是中点所在的时钟小时",
                "day": "只能读日实体，查询是中点所在的公历日",
                "month": "只能读月实体，查询是中点所在的公历月",
                "window6": "用已经保存的最细划分，回答中点前后各三小时",
                "joint": "小时、日、月三个读者的绝对误差的平均，不含六小时窗口",
            },
            "truth": "区间与查询格的重叠长度除以区间长度",
            "bootstrap_cities": BOOTSTRAP,
            "bootstrap_seed": 0,
            "macro_minimum_events": MIN_CITY,
            "audit_rows": checked,
            "identity_max_abs_error": {"month": month_gap, "hour_when_atoms_exist": hour_gap, "day_when_atoms_exist": day_gap},
        },
        "n": int(len(frame)),
        "overall": overall,
        "best_configuration": {reader: _best(overall, reader) for reader in READERS},
        "by_hard_level": by_hard,
        "by_width": by_width,
        "external_tasks": _external(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "ablation_report.json"
    path.write_text(json.dumps(_clean(report), ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"写入 {path}")
    joint = {name: overall["configurations"][name]["joint"]["mae"] for name, _a, _f, _c in CONFIGS}
    window = {name: overall["configurations"][name]["window6"]["mae"] for name, _a, _f, _c in CONFIGS}
    _log("联合绝对误差 " + " ".join(f"{name}={value:.4f}" for name, value in joint.items()))
    _log("六小时窗口 " + " ".join(f"{name}={value:.4f}" for name, value in window.items()))
    for label, _other in DROPS:
        delta = overall["full_minus_ablation"][label]["joint"]
        _log(f"{label} 联合差 {delta['mae']:.4f} 区间 {delta['mae_ci']}")


if __name__ == "__main__":
    main()
