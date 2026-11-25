"""三条可关闭的对齐模块，以及只读某一层时间实体时的质量估计。

模块一，裁剪支撑：原子是区间与格子的交集，质量正比于长度。
关闭时，若存在能包住整段区间的最细标准格，就把质量均匀铺在这个格子上；
没有这种格子时，退化为区间起点上的狄拉克测度。

模块二，滤子推前：更粗的格子只接收更细原子的质量，不另做一次分配。
推前只向上，不把月拆回日。关闭时，读者只能看见当前最细的那一层。

模块三，昼夜展开：宽度不超过 45 天时补出小时原子。
勒贝格查询仍按长度读这些小时，不用小时强度；强度只进入隐藏钟点的 Brier 分数。
关闭时，超出阶梯的区间没有小时原子。

八种开关互相独立。近重复排序用的是「区间是否与窗口相交」，
这个示性函数不读推前，也不读小时强度。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OFFSET = 8 * 3600
HOUR = 3600
DAY = 86400
WINDOW6 = 6 * HOUR
LADDER_HOUR = 36 * HOUR
LADDER_DAY = 10 * DAY
LADDER_TEN = 32 * DAY
MAX_EXPAND = 45 * DAY

NONE = 0
HOUR_L = 1
DAY_L = 2
WEEK_L = 3
TEN_L = 4
MONTH_L = 5
LEVEL_NAME = {
    NONE: "",
    HOUR_L: "小时",
    DAY_L: "日",
    WEEK_L: "周",
    TEN_L: "旬",
    MONTH_L: "月",
}

# 名称按「支撑、推前、昼夜」三位二进制书写，1 表示打开。
CONFIGS = (
    ("000", False, False, False),
    ("100", True, False, False),
    ("010", False, True, False),
    ("001", False, False, True),
    ("110", True, True, False),
    ("101", True, False, True),
    ("011", False, True, True),
    ("111", True, True, True),
)
READERS = ("hour", "day", "month", "window6")


@dataclass(frozen=True)
class IntervalGeometry:
    start: np.ndarray
    end: np.ndarray
    width: np.ndarray
    ladder: np.ndarray
    hard: np.ndarray
    hard_left: np.ndarray
    hard_right: np.ndarray
    hour_left: np.ndarray
    hour_right: np.ndarray
    day_left: np.ndarray
    day_right: np.ndarray
    month_left: np.ndarray
    month_right: np.ndarray
    window_left: np.ndarray
    window_right: np.ndarray
    day_b0: np.ndarray
    day_b1: np.ndarray
    day_c0: np.ndarray
    day_c1: np.ndarray
    ten_b0: np.ndarray
    ten_b1: np.ndarray
    ten_c0: np.ndarray
    ten_c1: np.ndarray
    week_b0: np.ndarray
    week_b1: np.ndarray
    week_c0: np.ndarray
    week_c1: np.ndarray
    month_b0: np.ndarray
    month_b1: np.ndarray
    month_c0: np.ndarray
    month_c1: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.start))


def _as_unix(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == object:
        raise TypeError("时间必须是 Unix 秒")
    if np.issubdtype(arr.dtype, np.datetime64):
        raise TypeError("时间必须是 Unix 秒")
    out = np.rint(np.asarray(values, dtype=np.float64)).astype(np.int64)
    return np.atleast_1d(out)


def _unix_from_shanghai(series: pd.Series) -> np.ndarray:
    """时区时间转 Unix 秒。pandas 的内部单位可能是微秒，不能写死纳秒。"""
    utc = series.dt.tz_convert("UTC")
    raw = utc.astype("int64").to_numpy()
    scale = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}[utc.dtype.unit]
    return raw // scale


def _calendar(unix: np.ndarray) -> dict[str, np.ndarray]:
    """上海时区下，每个时刻所在周、旬、月的左右端点，Unix 秒。"""
    if len(unix) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return {key: empty for key in ("week_left", "week_right", "ten_left", "ten_right", "month_left", "month_right")}
    moment = pd.Series(pd.to_datetime(np.asarray(unix, dtype=np.int64), unit="s", utc=True))
    moment = moment.dt.tz_convert("Asia/Shanghai")
    month_left_ts = moment.dt.floor("D") - pd.to_timedelta(moment.dt.day.to_numpy() - 1, unit="D")
    month_left = _unix_from_shanghai(month_left_ts)
    dim = moment.dt.days_in_month.to_numpy(dtype=np.int64)
    month_right = month_left + dim * DAY
    day = moment.dt.day.to_numpy(dtype=np.int64)
    ten_start = np.where(day <= 10, 1, np.where(day <= 20, 11, 21)).astype(np.int64)
    ten_left = month_left + (ten_start - 1) * DAY
    ten_right = np.where(day <= 10, month_left + 10 * DAY, np.where(day <= 20, month_left + 20 * DAY, month_right))
    monday = moment.dt.floor("D") - pd.to_timedelta(moment.dt.day_of_week.to_numpy(), unit="D")
    week_left = _unix_from_shanghai(monday)
    return {
        "week_left": week_left,
        "week_right": week_left + 7 * DAY,
        "ten_left": ten_left.astype(np.int64),
        "ten_right": ten_right.astype(np.int64),
        "month_left": month_left.astype(np.int64),
        "month_right": month_right.astype(np.int64),
    }


def _hour_bounds(unix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local = unix + OFFSET
    left = (local // HOUR) * HOUR - OFFSET
    return left.astype(np.int64), (left + HOUR).astype(np.int64)


def _day_bounds(unix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local = unix + OFFSET
    left = (local // DAY) * DAY - OFFSET
    return left.astype(np.int64), (left + DAY).astype(np.int64)


def _slice_cal(cal: dict[str, np.ndarray], n: int, index: int) -> dict[str, np.ndarray]:
    start = index * n
    stop = (index + 1) * n
    return {key: value[start:stop] for key, value in cal.items()}


def build_geometry(start_unix, end_unix) -> IntervalGeometry:
    """一次算出硬格子、阶梯粒度和各读者要用的查询格。"""
    start = _as_unix(start_unix)
    end = _as_unix(end_unix)
    if start.shape != end.shape:
        raise ValueError("起止时间的长度必须相同")
    end = np.maximum(end, start)
    width = end - start
    mid = start + width // 2
    window_left = mid - WINDOW6 // 2
    window_right = mid + WINDOW6 // 2
    # 六小时窗口的右端点是开区间，最后一个落入的格子用右端点前一秒。
    window_probe = np.maximum(window_left, window_right - 1)
    n = len(start)
    stacked = np.concatenate([start, mid, window_left, window_probe])
    parts = [_slice_cal(_calendar(stacked), n, index) for index in range(4)]
    start_cal, mid_cal, left_cal, right_cal = parts
    hour_left, hour_right = _hour_bounds(start)
    day_left, day_right = _day_bounds(start)
    in_hour = end <= hour_right
    in_day = end <= day_right
    in_week = end <= start_cal["week_right"]
    in_ten = end <= start_cal["ten_right"]
    in_month = end <= start_cal["month_right"]
    hard = np.full(n, NONE, dtype=np.int8)
    hard = np.where(in_month, MONTH_L, hard)
    hard = np.where(in_ten, TEN_L, hard)
    hard = np.where(in_week, WEEK_L, hard)
    hard = np.where(in_day, DAY_L, hard)
    hard = np.where(in_hour, HOUR_L, hard)
    hard_left = np.zeros(n, dtype=np.int64)
    hard_right = np.zeros(n, dtype=np.int64)
    for level, left, right in (
        (HOUR_L, hour_left, hour_right),
        (DAY_L, day_left, day_right),
        (WEEK_L, start_cal["week_left"], start_cal["week_right"]),
        (TEN_L, start_cal["ten_left"], start_cal["ten_right"]),
        (MONTH_L, start_cal["month_left"], start_cal["month_right"]),
    ):
        pick = hard == level
        hard_left = np.where(pick, left, hard_left)
        hard_right = np.where(pick, right, hard_right)
    exact_hour = (start == hour_left) & (end == hour_right)
    exact_day = (start == day_left) & (end == day_right)
    exact_week = (start == start_cal["week_left"]) & (end == start_cal["week_right"])
    exact_ten = (start == start_cal["ten_left"]) & (end == start_cal["ten_right"])
    exact_month = (start == start_cal["month_left"]) & (end == start_cal["month_right"])
    ladder = np.full(n, MONTH_L, dtype=np.int8)
    ladder = np.where(width > LADDER_TEN, MONTH_L, TEN_L)
    ladder = np.where(width <= LADDER_DAY, DAY_L, ladder)
    ladder = np.where(width <= LADDER_HOUR, HOUR_L, ladder)
    ladder = np.where(exact_month, MONTH_L, ladder)
    ladder = np.where(exact_ten, TEN_L, ladder)
    ladder = np.where(exact_week, WEEK_L, ladder)
    ladder = np.where(exact_day, DAY_L, ladder)
    ladder = np.where(exact_hour, HOUR_L, ladder)
    probe_hour_l, probe_hour_r = _hour_bounds(mid)
    probe_day_l, probe_day_r = _day_bounds(mid)
    day_b0, day_b1 = _day_bounds(window_left)
    day_c0, day_c1 = _day_bounds(window_probe)
    return IntervalGeometry(
        start=start,
        end=end,
        width=width,
        ladder=ladder,
        hard=hard,
        hard_left=hard_left,
        hard_right=hard_right,
        hour_left=probe_hour_l,
        hour_right=probe_hour_r,
        day_left=probe_day_l,
        day_right=probe_day_r,
        month_left=mid_cal["month_left"],
        month_right=mid_cal["month_right"],
        window_left=window_left,
        window_right=window_right,
        day_b0=day_b0,
        day_b1=day_b1,
        day_c0=day_c0,
        day_c1=day_c1,
        ten_b0=left_cal["ten_left"],
        ten_b1=left_cal["ten_right"],
        ten_c0=right_cal["ten_left"],
        ten_c1=right_cal["ten_right"],
        week_b0=left_cal["week_left"],
        week_b1=left_cal["week_right"],
        week_c0=right_cal["week_left"],
        week_c1=right_cal["week_right"],
        month_b0=left_cal["month_left"],
        month_b1=left_cal["month_right"],
        month_c0=right_cal["month_left"],
        month_c1=right_cal["month_right"],
    )


def fraction(left, right, query_left, query_right) -> np.ndarray:
    """均匀测度落在查询区间里的比例。点质量用示性函数。"""
    left = _as_unix(left)
    right = _as_unix(right)
    query_left = _as_unix(query_left)
    query_right = _as_unix(query_right)
    width = right - left
    point = width <= 0
    overlap = np.minimum(right, query_right) - np.maximum(left, query_left)
    overlap = np.maximum(overlap, 0)
    out = np.zeros(len(left), dtype=np.float64)
    np.divide(overlap, np.maximum(width, 1), out=out, where=~point)
    inside = (left >= query_left) & (left < query_right)
    return np.where(point, inside.astype(np.float64), out)


def interval_incidence(start, end, query_left, query_right) -> np.ndarray:
    """区间与查询窗口是否相交。不依赖三个模块的开关。"""
    start = _as_unix(start)
    end = _as_unix(end)
    query_left = _as_unix(query_left)
    query_right = _as_unix(query_right)
    end = np.maximum(end, start)
    return (np.minimum(end, query_right) > np.maximum(start, query_left)).astype(np.float64)


def _dirac(point: np.ndarray, query_left: np.ndarray, query_right: np.ndarray) -> np.ndarray:
    return ((point >= query_left) & (point < query_right)).astype(np.float64)


def _two_cell_spread(start, end, left_a, right_a, left_b, right_b, query_left, query_right) -> np.ndarray:
    first = fraction(start, end, left_a, right_a) * fraction(left_a, right_a, query_left, query_right)
    same = (left_a == left_b) & (right_a == right_b)
    second = fraction(start, end, left_b, right_b) * fraction(left_b, right_b, query_left, query_right)
    return np.where(same, first, first + second)


def _hour_estimate(geo: IntervalGeometry, adapted: bool, circadian: bool) -> np.ndarray:
    query_left, query_right = geo.hour_left, geo.hour_right
    truth = fraction(geo.start, geo.end, query_left, query_right)
    if adapted:
        has_hours = (geo.ladder == HOUR_L) | (circadian & (geo.width <= MAX_EXPAND))
        return np.where(has_hours, truth, 0.0)
    hard_width = geo.hard_right - geo.hard_left
    expand = circadian & (geo.hard != NONE) & (hard_width <= MAX_EXPAND)
    visible = (geo.hard == HOUR_L) | expand
    spread = fraction(geo.hard_left, geo.hard_right, query_left, query_right)
    missing = np.where(geo.hard == NONE, _dirac(geo.start, query_left, query_right), 0.0)
    return np.where(visible, spread, missing)


def _day_estimate(geo: IntervalGeometry, adapted: bool, filtration: bool) -> np.ndarray:
    query_left, query_right = geo.day_left, geo.day_right
    truth = fraction(geo.start, geo.end, query_left, query_right)
    if adapted:
        has_days = (geo.ladder == DAY_L) | (filtration & (geo.ladder == HOUR_L))
        return np.where(has_days, truth, 0.0)
    visible = (geo.hard == DAY_L) | (filtration & (geo.hard == HOUR_L))
    spread = fraction(geo.hard_left, geo.hard_right, query_left, query_right)
    missing = np.where(geo.hard == NONE, _dirac(geo.start, query_left, query_right), 0.0)
    return np.where(visible, spread, missing)


def _month_estimate(geo: IntervalGeometry, adapted: bool, filtration: bool) -> np.ndarray:
    query_left, query_right = geo.month_left, geo.month_right
    truth = fraction(geo.start, geo.end, query_left, query_right)
    if adapted:
        has_months = (geo.ladder == MONTH_L) | filtration
        return np.where(has_months, truth, 0.0)
    visible = (geo.hard == MONTH_L) | (filtration & (geo.hard != NONE))
    spread = fraction(geo.hard_left, geo.hard_right, query_left, query_right)
    missing = np.where(geo.hard == NONE, _dirac(geo.start, query_left, query_right), 0.0)
    return np.where(visible, spread, missing)


def _window_estimate(geo: IntervalGeometry, adapted: bool, filtration: bool, circadian: bool) -> np.ndarray:
    """用已保存的最细划分回答中点前后各三小时的查询。"""
    query_left, query_right = geo.window_left, geo.window_right
    truth = fraction(geo.start, geo.end, query_left, query_right)
    if adapted:
        hour_atoms = (geo.ladder == HOUR_L) | (circadian & (geo.width <= MAX_EXPAND))
        day_atoms = (geo.ladder == DAY_L) | (filtration & (geo.ladder == HOUR_L))
        ten_atoms = (geo.ladder == TEN_L) | (filtration & np.isin(geo.ladder, (HOUR_L, DAY_L)))
        week_atoms = (geo.ladder == WEEK_L) | (filtration & np.isin(geo.ladder, (HOUR_L, DAY_L)))
        month_atoms = (geo.ladder == MONTH_L) | filtration
        day_spread = _two_cell_spread(geo.start, geo.end, geo.day_b0, geo.day_b1, geo.day_c0, geo.day_c1, query_left, query_right)
        ten_spread = _two_cell_spread(geo.start, geo.end, geo.ten_b0, geo.ten_b1, geo.ten_c0, geo.ten_c1, query_left, query_right)
        week_spread = _two_cell_spread(geo.start, geo.end, geo.week_b0, geo.week_b1, geo.week_c0, geo.week_c1, query_left, query_right)
        month_spread = _two_cell_spread(geo.start, geo.end, geo.month_b0, geo.month_b1, geo.month_c0, geo.month_c1, query_left, query_right)
        estimate = np.zeros(geo.n, dtype=np.float64)
        estimate = np.where(month_atoms, month_spread, estimate)
        estimate = np.where(week_atoms, week_spread, estimate)
        estimate = np.where(ten_atoms, ten_spread, estimate)
        estimate = np.where(day_atoms, day_spread, estimate)
        estimate = np.where(hour_atoms, truth, estimate)
        return estimate
    hard_width = geo.hard_right - geo.hard_left
    hour_atoms = (geo.hard == HOUR_L) | (circadian & (geo.hard != NONE) & (hard_width <= MAX_EXPAND))
    day_atoms = (geo.hard == DAY_L) | (filtration & (geo.hard == HOUR_L))
    ten_atoms = (geo.hard == TEN_L) | (filtration & np.isin(geo.hard, (HOUR_L, DAY_L)))
    week_atoms = (geo.hard == WEEK_L) | (filtration & np.isin(geo.hard, (HOUR_L, DAY_L)))
    month_atoms = (geo.hard == MONTH_L) | (filtration & (geo.hard != NONE))
    cell = fraction(geo.hard_left, geo.hard_right, query_left, query_right)
    estimate = np.where(geo.hard == NONE, _dirac(geo.start, query_left, query_right), 0.0)
    estimate = np.where(month_atoms, cell, estimate)
    estimate = np.where(week_atoms, cell, estimate)
    estimate = np.where(ten_atoms, cell, estimate)
    estimate = np.where(day_atoms, cell, estimate)
    estimate = np.where(hour_atoms, cell, estimate)
    return estimate


def truth_fractions(geo: IntervalGeometry) -> dict[str, np.ndarray]:
    return {
        "hour": fraction(geo.start, geo.end, geo.hour_left, geo.hour_right),
        "day": fraction(geo.start, geo.end, geo.day_left, geo.day_right),
        "month": fraction(geo.start, geo.end, geo.month_left, geo.month_right),
        "window6": fraction(geo.start, geo.end, geo.window_left, geo.window_right),
    }


def configuration_estimates(geo: IntervalGeometry) -> dict[str, dict[str, np.ndarray]]:
    """八种开关下，四类读者给出的质量比例。"""
    estimates: dict[str, dict[str, np.ndarray]] = {}
    for name, adapted, filtration, circadian in CONFIGS:
        estimates[name] = {
            "hour": _hour_estimate(geo, adapted, circadian),
            "day": _day_estimate(geo, adapted, filtration),
            "month": _month_estimate(geo, adapted, filtration),
            "window6": _window_estimate(geo, adapted, filtration, circadian),
        }
    return estimates


def absolute_errors(geo: IntervalGeometry) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    truth = truth_fractions(geo)
    estimates = configuration_estimates(geo)
    errors = {
        name: {reader: np.abs(estimates[name][reader] - truth[reader]) for reader in READERS}
        for name, _, _, _ in CONFIGS
    }
    return truth, errors
