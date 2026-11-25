"""滤子一致的区间鞅非平衡传输（FIMUT）。

IMUT 把一条犯罪记录的质量放在最细的一层时间格上。项目书要求时间实体同时
具备小时、日、周、月，空间实体同时具备城市网格和省、地、县。FIMUT 只在
IMUT 选定的最细划分上做零位移重叠，再把质量推到所有更粗、且能盖住这些
原子的格子上。因此每一层都是上一层的条件期望：同一层的质量之和等于保留
质量，一个月的质量等于落在这个月里的那些日质量之和。

空间坐标是一个点。它落进哪个 500 米网格由包含关系决定，网格编码仍是
「6 位地级码 + 3 位 X + 3 位 Y」。点到该网格均匀分布的 2-Wasserstein
距离只作为网格表示误差，不把质量搬到相邻网格。省、地、县是这个点所在的
更粗空间 σ-代数，置信度与网格相同。

有效四元组沿用研究方案的五个字段：实体1、关系、实体2、ISO 8601 时间、
置信度。整条时间计划的保留质量低于 LOW_CONFIDENCE 时，不输出时间四元组；
保留质量达标时，各层切片都保留，因为切片是同一条计划的分量。

潜在时刻的参照测度是区间上的勒贝格测度，也可以再乘一个外部的 24 小时强度
后重新归一化。强度来自测试年之前的精确时刻，不在当前区间内部估计。
日、周、旬、月必须是这些小时原子的推前：把每一层分开倾斜会破坏
「各层质量之和相等」。强度恒为 1 时，小时计划与均匀重叠一致。

两条区间上的潜在时刻若相互独立，时间差落在阈值内的概率由分段线性积分
给出。它是 Knox 时空相互作用统计量在区间观测下的成对权重。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from tkg_model.holidays import holiday_name, period_name
from tkg_pipeline.align import TZ, finest_time_entity, to_iso, to_unix
from tkg_pipeline.config import GRID_SIZE_M, LOW_CONFIDENCE
from tkg_pipeline.imut import (
    ETA_HOURS,
    _day_bins,
    _from_unix,
    _month_bins,
    _ten_bins,
    imut_align,
    rectangle_w2_squared,
    retention_mass,
)

COARSER = {
    "小时": ("日", "周", "旬", "月"),
    "日": ("周", "旬", "月"),
    "周": ("月",),
    "旬": ("月",),
    "月": (),
}


@dataclass(frozen=True)
class TimeAtom:
    entity_id: str
    level: str
    mass: float
    iso8601: str
    holiday: str = ""
    period: str = ""


@dataclass(frozen=True)
class FiltrationPlan:
    atoms: tuple[TimeAtom, ...]
    finest_level: str
    retained: float
    hard_entity_id: str
    anchor_iso: str
    spatial_links: tuple[tuple[str, str, float], ...] = ()
    grid_w2_m: float | None = None

    def masses(self, level: str) -> dict[str, float]:
        return {atom.entity_id: atom.mass for atom in self.atoms if atom.level == level}

    def level_total(self, level: str) -> float:
        return float(sum(self.masses(level).values()))

    def quadruples(self, crime_id: str) -> list[dict]:
        """研究方案中的五字段四元组。时间计划整体过低时不拆出时间边。"""
        rows: list[dict] = []
        if self.retained >= LOW_CONFIDENCE:
            for atom in self.atoms:
                rows.append(
                    {
                        "entity1_id": crime_id,
                        "relation_type": "时空关联",
                        "entity2_id": atom.entity_id,
                        "iso8601_timestamp": atom.iso8601,
                        "confidence": atom.mass,
                        "level": atom.level,
                        "holiday": atom.holiday,
                    }
                )
        for entity_id, level, confidence in self.spatial_links:
            if confidence < LOW_CONFIDENCE or not entity_id:
                continue
            rows.append(
                {
                    "entity1_id": crime_id,
                    "relation_type": "时空关联",
                    "entity2_id": entity_id,
                    "iso8601_timestamp": self.anchor_iso,
                    "confidence": confidence,
                    "level": level,
                    "holiday": "",
                }
            )
        return rows


def _week_bins(start: datetime, end: datetime) -> list[tuple[str, str, datetime, datetime]]:
    cursor = (start - timedelta(days=start.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    bins = []
    while cursor < end and len(bins) < 12:
        nxt = cursor + timedelta(days=7)
        iso = cursor.isocalendar()
        bins.append((f"TW{iso[0]}{iso[1]:02d}", "周", cursor, nxt))
        cursor = nxt
    return bins


def _generator(level: str):
    return {
        "小时": None,
        "日": _day_bins,
        "周": _week_bins,
        "旬": _ten_bins,
        "月": _month_bins,
    }[level]


def _overlap_masses(level: str, start: datetime, end: datetime, retained: float) -> list[TimeAtom]:
    width = (end - start).total_seconds()
    if width <= 0 or retained <= 0:
        return []
    atoms = []
    for entity_id, _, left, right in _generator(level)(start, end):
        lo = max(start, left)
        hi = min(end, right)
        seconds = (hi - lo).total_seconds()
        if seconds <= 0:
            continue
        moment = lo
        atoms.append(
            TimeAtom(
                entity_id,
                level,
                float(retained * seconds / width),
                to_iso(moment),
                holiday_name(moment.date()) if level == "日" else "",
                "",
            )
        )
    return atoms


def _finest_atoms(plan_start: datetime, plan_end: datetime) -> tuple[tuple[TimeAtom, ...], str, float, str]:
    finest = imut_align(plan_start, plan_end)
    atoms = []
    for entity_id, mass, (lo, _hi) in zip(finest.entity_ids, finest.mass, finest.bounds):
        moment = _from_unix(lo)
        atoms.append(
            TimeAtom(
                entity_id,
                finest.level,
                float(mass),
                to_iso(moment),
                holiday_name(moment.date()) if finest.level in {"小时", "日"} else "",
                period_name(moment.hour) if finest.level == "小时" else "",
            )
        )
    return tuple(atoms), finest.level, float(finest.retained), finest.hard_entity_id


def fimut_time(start: datetime, end: datetime) -> FiltrationPlan:
    """一条时间区间在小时、日、周、旬、月上的滤子计划。"""
    if start.tzinfo is None:
        start = start.replace(tzinfo=TZ)
    if end.tzinfo is None:
        end = end.replace(tzinfo=TZ)
    if end <= start:
        end = start + timedelta(seconds=1)
    finest_atoms, level, retained, hard_id = _finest_atoms(start, end)
    atoms = list(finest_atoms)
    for coarser in COARSER.get(level, ()):
        atoms.extend(_overlap_masses(coarser, start, end, retained))
    return FiltrationPlan(tuple(atoms), level, retained, hard_id, to_iso(start))


def fimut_time_unix(start_ts: float, end_ts: float) -> FiltrationPlan:
    return fimut_time(_from_unix(start_ts), _from_unix(end_ts))


def grid_w2_meters(proj_x: float, proj_y: float, origin_x: float, origin_y: float, grid_x: int, grid_y: int, size: float = GRID_SIZE_M) -> float:
    """点到它所落入的 500 米网格上均匀分布的 2-Wasserstein 距离，单位米。"""
    xmin = origin_x + grid_x * size
    ymin = origin_y + grid_y * size
    squared = rectangle_w2_squared(proj_x, proj_y, xmin, xmin + size, ymin, ymin + size)
    return float(np.sqrt(max(squared, 0.0)))


def spatial_hierarchy(grid_id: str, province_code: str, city_code: str, admin_code: str, confidence: float) -> tuple[tuple[str, str, float], ...]:
    """点被哪一级空间 σ-代数观测到。每一级都是同一个点，置信度相同。"""
    links = []
    if grid_id:
        links.append((grid_id, "城市网格", float(confidence)))
    if admin_code:
        links.append((f"GD{admin_code}", "县级", float(confidence)))
    if city_code:
        links.append((f"GC{city_code}", "地级", float(confidence)))
    if province_code:
        links.append((f"GP{province_code}", "省级", float(confidence)))
    return tuple(links)


def restricted_mass(start: float, end: float, query_start: float, query_end: float, eta_hours: float = ETA_HOURS) -> float:
    """区间测度落在查询窗口里的质量。

    保留质量仍由整段区间的宽度决定。窗口只取与区间相交的那一段，
    所以跨在窗口两侧的长区间不会被算成一整次案发。
    """
    if query_end <= query_start or end < start:
        return 0.0
    width = max(float(end) - float(start), 1.0)
    overlap = min(float(end), float(query_end)) - max(float(start), float(query_start))
    if overlap <= 0:
        return 0.0
    return float(retention_mass(float(start), float(start) + width, eta_hours) * overlap / width)


def contact_score(start: float, end: float, query: float, window_seconds: float) -> float:
    """区间与查询窗口相交时，按接触段中点到查询时刻的距离衰减。

    衰减尺度固定为窗口长度，不在测试集上挑选。窗口外的记录得 0。
    整段盖住窗口时，中点在窗口中央，得 exp(-1/2)。
    """
    if window_seconds <= 0 or end < start:
        return 0.0
    opening = float(query) - float(window_seconds)
    if float(end) <= opening or float(start) >= float(query):
        return 0.0
    left = max(float(start), opening)
    right = min(float(end), float(query))
    if right <= left:
        return 0.0
    age = float(query) - 0.5 * (left + right)
    return float(np.exp(-age / float(window_seconds)))


def tie_hit_probability(better: int, tied_with_self: int, k: int) -> float:
    """并列名次随机拆开时，真实网格进入前 k 名的概率。"""
    if tied_with_self <= 0 or k <= 0:
        return 0.0
    start = better + 1
    end = better + tied_with_self
    if end <= k:
        return 1.0
    if start > k:
        return 0.0
    return (k - start + 1) / (end - start + 1)


_LOCAL_OFFSET = 8 * 3600.0
_MAX_TILT_HOURS = 24 * 45


def _hour_weight(hour_weight) -> np.ndarray:
    if hour_weight is None:
        return np.ones(24, dtype=float)
    weight = np.asarray(hour_weight, dtype=float)
    if weight.shape != (24,) or not np.isfinite(weight).all() or np.any(weight < 0) or float(weight.sum()) <= 0:
        raise ValueError("小时强度必须是 24 个非负有限数，且总和为正")
    return weight


def hour_occupancy_batch(start_unix, end_unix, hour_weight=None) -> np.ndarray:
    """区间上潜在时刻落在 0–23 时的条件概率，形状 (n, 24)。

    hour_weight 为 None 时，密度正比于长度。给定强度时，密度正比于
    强度乘长度，支撑仍限制在区间内部。
    """
    start = np.atleast_1d(np.asarray(start_unix, dtype=float))
    end = np.atleast_1d(np.asarray(end_unix, dtype=float))
    if start.shape != end.shape:
        raise ValueError("起止时间的形状必须相同")
    weight = _hour_weight(hour_weight)
    loc_s = start + _LOCAL_OFFSET
    loc_e = np.where(end > start, end + _LOCAL_OFFSET, loc_s + 1.0)
    h0 = np.floor(loc_s / 3600.0).astype(np.int64)
    h1 = np.ceil(loc_e / 3600.0).astype(np.int64) - 1
    h1 = np.maximum(h1, h0)
    span = int((h1 - h0).max()) + 1 if len(h0) else 1
    if span > _MAX_TILT_HOURS:
        raise ValueError("区间超过 45 天，不再逐小时展开")
    mass = np.zeros((len(start), 24), dtype=float)
    row = np.arange(len(start))
    for step in range(span):
        hour_index = h0 + step
        active = hour_index <= h1
        if not np.any(active):
            continue
        left = np.maximum(loc_s, hour_index * 3600.0)
        right = np.minimum(loc_e, (hour_index + 1) * 3600.0)
        seconds = np.maximum(0.0, right - left)
        hod = np.mod(hour_index, 24)
        np.add.at(mass, (row[active], hod[active]), seconds[active] * weight[hod[active]])
    total = mass.sum(axis=1, keepdims=True)
    flat = np.full_like(mass, 1.0 / 24.0)
    return np.divide(mass, total, out=flat, where=total > 0)


def hour_occupancy(start: datetime, end: datetime, hour_weight=None) -> np.ndarray:
    """单条区间的 24 小时条件概率。"""
    if start.tzinfo is None:
        start = start.replace(tzinfo=TZ)
    if end.tzinfo is None:
        end = end.replace(tzinfo=TZ)
    return hour_occupancy_batch(float(to_unix(start)), float(to_unix(end)), hour_weight)[0]


def _calendar_id(level: str, moment: datetime) -> str:
    if level == "日":
        return f"TD{moment.strftime('%Y%m%d')}"
    if level == "周":
        iso = moment.isocalendar()
        return f"TW{iso[0]}{iso[1]:02d}"
    if level == "旬":
        index = "1" if moment.day <= 10 else "2" if moment.day <= 20 else "3"
        return f"TN{moment.strftime('%Y%m')}{index}"
    if level == "月":
        return f"TM{moment.strftime('%Y%m')}"
    raise KeyError(level)


def fimut_time_tilted(start: datetime, end: datetime, hour_weight, eta_hours: float = ETA_HOURS) -> FiltrationPlan:
    """先在小时原子上倾斜，再把质量推前到日、周、旬、月。

    每一层的质量和都等于保留质量。粗层是细层的和，不是另一次倾斜。
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=TZ)
    if end.tzinfo is None:
        end = end.replace(tzinfo=TZ)
    if end <= start:
        end = start + timedelta(seconds=1)
    weight = _hour_weight(hour_weight)
    retained = retention_mass(float(to_unix(start)), float(to_unix(end)), eta_hours)
    hard_id, _hard_level = finest_time_entity(start, end)
    cursor = start.replace(minute=0, second=0, microsecond=0)
    pieces: list[tuple[datetime, datetime, float]] = []
    while cursor < end:
        if len(pieces) >= _MAX_TILT_HOURS:
            raise ValueError("区间超过 45 天，不再逐小时展开")
        nxt = cursor + timedelta(hours=1)
        lo = max(start, cursor)
        hi = min(end, nxt)
        seconds = (hi - lo).total_seconds()
        if seconds > 0:
            pieces.append((cursor, lo, seconds * float(weight[cursor.hour])))
        cursor = nxt
    raw = np.array([item[2] for item in pieces], dtype=float)
    total = float(raw.sum())
    scale = retained / total if total > 0 else 0.0
    buckets = {level: {} for level in ("小时", "日", "周", "旬", "月")}
    atoms: list[TimeAtom] = []
    for (hour_start, lo, _), value in zip(pieces, raw):
        mass = float(value * scale)
        atoms.append(
            TimeAtom(
                f"T{hour_start.strftime('%Y%m%d%H')}",
                "小时",
                mass,
                to_iso(lo),
                holiday_name(hour_start.date()),
                period_name(hour_start.hour),
            )
        )
        for level in ("日", "周", "旬", "月"):
            key = _calendar_id(level, hour_start)
            buckets[level][key] = buckets[level].get(key, 0.0) + mass
    for level in ("日", "周", "旬", "月"):
        for entity_id, mass in buckets[level].items():
            moment = next(hour_start for hour_start, _, _ in pieces if _calendar_id(level, hour_start) == entity_id)
            atoms.append(
                TimeAtom(
                    entity_id,
                    level,
                    float(mass),
                    to_iso(moment),
                    holiday_name(moment.date()) if level == "日" else "",
                    "",
                )
            )
    return FiltrationPlan(tuple(atoms), "小时", float(retained), hard_id, to_iso(start))


def _piece_area(left: np.ndarray, right: np.ndarray, c: np.ndarray, d: np.ndarray, tau: float) -> np.ndarray:
    """在不含断点的一段上，积分 max(0, min(d, u+τ) - max(c, u-τ))。"""
    width = right - left
    raw_left = np.minimum(d, left + tau) - np.maximum(c, left - tau)
    raw_right = np.minimum(d, right + tau) - np.maximum(c, right - tau)
    area = np.zeros(left.shape, dtype=float)
    active = width > 1e-12
    both = active & (raw_left >= 0) & (raw_right >= 0)
    area = np.where(both, 0.5 * (raw_left + raw_right) * width, area)
    denom = raw_left - raw_right
    frac = np.divide(raw_left, denom, out=np.zeros_like(raw_left), where=np.abs(denom) > 1e-15)
    down = active & (raw_left > 0) & (raw_right < 0)
    area = np.where(down, 0.5 * raw_left * frac * width, area)
    up = active & (raw_left < 0) & (raw_right > 0)
    area = np.where(up, 0.5 * raw_right * (1.0 - frac) * width, area)
    return area


def uniform_close_probability_many(left_start, left_end, right_start, right_end, tau: float) -> np.ndarray:
    """独立均匀潜在时刻满足 |U-V|≤τ 的概率。

    区间长度为正时，积分在 u = 右端点 ± τ 处分成至多三段，每段被积函数线性，
    梯形或三角公式是精确的。长度为零时退化为狄拉克点。
    """
    a = np.asarray(left_start, dtype=float).reshape(-1)
    b = np.asarray(left_end, dtype=float).reshape(-1)
    c = np.asarray(right_start, dtype=float).reshape(-1)
    d = np.asarray(right_end, dtype=float).reshape(-1)
    if not (len(a) == len(b) == len(c) == len(d)):
        raise ValueError("四条边界的长度必须相同")
    threshold = float(tau)
    out = np.zeros(len(a), dtype=float)
    if threshold < 0 or len(a) == 0:
        return out
    length_u = b - a
    length_v = d - c
    both_points = (np.abs(length_u) <= 1e-9) & (np.abs(length_v) <= 1e-9) & (length_u >= 0) & (length_v >= 0)
    out[both_points] = (np.abs(a[both_points] - c[both_points]) <= threshold).astype(float)
    only_u = (np.abs(length_u) <= 1e-9) & (length_v > 0) & (length_u >= 0)
    if np.any(only_u):
        inside = np.minimum(d, a + threshold) - np.maximum(c, a - threshold)
        out[only_u] = np.maximum(0.0, inside[only_u]) / length_v[only_u]
    only_v = (np.abs(length_v) <= 1e-9) & (length_u > 0) & (length_v >= 0)
    if np.any(only_v):
        inside = np.minimum(b, c + threshold) - np.maximum(a, c - threshold)
        out[only_v] = np.maximum(0.0, inside[only_v]) / length_u[only_v]
    live = (length_u > 1e-9) & (length_v > 1e-9)
    if not np.any(live):
        return np.clip(out, 0.0, 1.0)
    gap = np.maximum(a, c) - np.minimum(b, d)
    span = np.maximum(b, d) - np.minimum(a, c)
    out[live & (gap > threshold)] = 0.0
    out[live & (span <= threshold)] = 1.0
    middle = live & (gap <= threshold) & (span > threshold)
    if np.any(middle):
        aa, bb, cc, dd = a[middle], b[middle], c[middle], d[middle]
        first = np.clip(cc + threshold, aa, bb)
        second = np.clip(dd - threshold, aa, bb)
        low = np.minimum(first, second)
        high = np.maximum(first, second)
        area = _piece_area(aa, low, cc, dd, threshold)
        area += _piece_area(low, high, cc, dd, threshold)
        area += _piece_area(high, bb, cc, dd, threshold)
        out[middle] = area / (length_u[middle] * length_v[middle])
    return np.clip(out, 0.0, 1.0)


def uniform_close_probability(left_start: float, left_end: float, right_start: float, right_end: float, tau: float) -> float:
    """两条区间的成对接近概率。"""
    return float(uniform_close_probability_many(left_start, left_end, right_start, right_end, tau)[0])


def start_close_counts(starts, taus: tuple[float, ...]) -> list[int]:
    """按起点排序后，统计 |起点差|≤τ 的无序点对。与成对扫描使用同一侧闭规则。"""
    ordered = np.sort(np.asarray(starts, dtype=float))
    index = np.arange(len(ordered))
    counts = []
    for tau in taus:
        nxt = np.searchsorted(ordered, ordered + float(tau), side="right")
        counts.append(int((nxt - index - 1).clip(min=0).sum()))
    return counts


def with_space(plan: FiltrationPlan, links: tuple[tuple[str, str, float], ...], grid_w2_m: float | None) -> FiltrationPlan:
    return FiltrationPlan(
        plan.atoms,
        plan.finest_level,
        plan.retained,
        plan.hard_entity_id,
        plan.anchor_iso,
        links,
        grid_w2_m,
    )
