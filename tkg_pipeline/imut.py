"""区间鞅非平衡传输对齐（IMUT）。

现有硬对齐把整段区间塞进一个能包住它的时间格。IMUT 把每条犯罪记录看成
区间上的均匀测度，只允许质量流向与该区间相交的时间格，因此耦合是适配的
（不能把未来或过去的质量搬进一个不相交的格子）。格子之间的分配等于区间
与格子的重叠比例，这是线上二次费用的零位移鞅耦合。保留下来的总质量由
区间到自身质心的 1-Wasserstein 距离经熵正则非平衡传输给出：区间越宽，
保留质量越低。

公式与出处：
- 一维均匀分布的分位函数给出 W2。Villani, Optimal Transport, 2009, Ch. 2。
  Unif[a,b] 与 Unif[c,d] 的 W2^2 = (a-c)^2 + (a-c)((b-a)-(d-c)) + ((b-a)-(d-c))^2 / 3。
- Unif[s,e] 到质心狄拉克测度的 W1 等于区间长度的四分之一。
- 非平衡熵正则传输的质量销毁：Liero, Mielke, Savaré, Invent. Math. 2018；
  Chizat, Peyré, Schmitzer, Vialard, Math. Comp. 2018。
- 鞅约束与适配（因果）耦合：Beiglböck, Henry-Labordère, Penkner, Finance Stoch. 2013；
  Acciaio, Backhoff-Veraguas, Zalashko, Stochastic Process. Appl. 2020；
  Eckstein, Pammer, Ann. Appl. Probab. 34:675–713, 2024。线上分区上的适配投影有显式解，
  不需要迭代 Sinkhorn。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from tkg_pipeline.align import TZ, candidate_bins, finest_time_entity, to_unix

ETA_HOURS = 720.0
HOUR = 3600.0
DAY = 86400.0


@dataclass(frozen=True)
class TransportPlan:
    entity_ids: tuple[str, ...]
    level: str
    mass: tuple[float, ...]
    bounds: tuple[tuple[float, float], ...]
    retained: float
    hard_entity_id: str
    hard_level: str

    @property
    def support(self) -> int:
        return int(sum(value > 0.0 for value in self.mass))

    @property
    def top_entity_id(self) -> str:
        if not self.mass:
            return ""
        return self.entity_ids[int(np.argmax(self.mass))]

    @property
    def top_mass(self) -> float:
        if not self.mass:
            return 0.0
        return float(max(self.mass))


def w2_squared_uniforms(left0: float, left1: float, right0: float, right1: float) -> float:
    """两个一维均匀分布的平方 2-Wasserstein 距离。端点相等时退化为质点。"""
    if left1 < left0 or right1 < right0:
        return float("inf")
    if left1 == left0:
        left1 = left0
    if right1 == right0:
        right1 = right0
    shift = left0 - right0
    scale = (left1 - left0) - (right1 - right0)
    return float(shift * shift + shift * scale + scale * scale / 3.0)


def w1_to_barycenter(start: float, end: float) -> float:
    """均匀分布到自身质心的 1-Wasserstein 距离，等于区间长度的四分之一。"""
    width = max(end - start, 0.0)
    return width / 4.0


def rectangle_w2_squared(x: float, y: float, xmin: float, xmax: float, ymin: float, ymax: float) -> float:
    """平面质点到矩形上均匀分布的平方 2-Wasserstein 距离。两个坐标分离。"""

    def axis(point: float, low: float, high: float) -> float:
        length = high - low
        if length <= 0:
            return (point - low) ** 2
        return (point - low) ** 2 - (point - low) * length + length * length / 3.0

    return axis(x, xmin, xmax) + axis(y, ymin, ymax)


def retention_mass(start: float, end: float, eta_hours: float = ETA_HOURS) -> float:
    """非平衡传输保留的总质量。宽度以小时计，尺度是 eta_hours。"""
    width_hours = max(end - start, 0.0) / HOUR
    if eta_hours <= 0:
        return 0.0
    return float(np.exp(-width_hours / eta_hours))


def _from_unix(value: float) -> datetime:
    return (datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=float(value))).astimezone(TZ)


def _hour_bins(start: datetime, end: datetime) -> list[tuple[str, str, datetime, datetime]]:
    cursor = start.replace(minute=0, second=0, microsecond=0)
    bins = []
    while cursor < end and len(bins) < 48:
        nxt = cursor + timedelta(hours=1)
        bins.append((f"T{cursor.strftime('%Y%m%d%H')}", "小时", cursor, nxt))
        cursor = nxt
    return bins


def _day_bins(start: datetime, end: datetime) -> list[tuple[str, str, datetime, datetime]]:
    cursor = start.replace(hour=0, minute=0, second=0, microsecond=0)
    bins = []
    while cursor < end and len(bins) < 16:
        nxt = cursor + timedelta(days=1)
        bins.append((f"TD{cursor.strftime('%Y%m%d')}", "日", cursor, nxt))
        cursor = nxt
    return bins


def _ten_bins(start: datetime, end: datetime) -> list[tuple[str, str, datetime, datetime]]:
    cursor = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    bins = []
    seen: set[str] = set()
    while cursor < end and len(bins) < 8:
        if cursor.day <= 10:
            left, right, index = cursor.replace(day=1), cursor.replace(day=11), "1"
        elif cursor.day <= 20:
            left, right, index = cursor.replace(day=11), cursor.replace(day=21), "2"
        else:
            nxt_month = cursor.replace(day=28) + timedelta(days=4)
            nxt_month = nxt_month.replace(day=1)
            left, right, index = cursor.replace(day=21), nxt_month, "3"
        entity_id = f"TN{left.strftime('%Y%m')}{index}"
        if entity_id not in seen and right > start and left < end:
            seen.add(entity_id)
            bins.append((entity_id, "旬", left, right))
        cursor = right
    return bins


def _month_bins(start: datetime, end: datetime) -> list[tuple[str, str, datetime, datetime]]:
    cursor = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    bins = []
    while cursor < end and len(bins) < 18:
        nxt = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        bins.append((f"TM{cursor.strftime('%Y%m')}", "月", cursor, nxt))
        cursor = nxt
    return bins


def overlapping_bins(start: datetime, end: datetime) -> list[tuple[str, str, datetime, datetime]]:
    """按区间宽度选择最细且仍然适配的格子层，只保留与区间相交的格子。"""
    if end <= start:
        end = start + timedelta(seconds=1)
    width = (end - start).total_seconds()
    if width <= 36 * HOUR:
        bins = _hour_bins(start, end)
    elif width <= 10 * DAY:
        bins = _day_bins(start, end)
    elif width <= 32 * DAY:
        bins = _ten_bins(start, end)
    else:
        bins = _month_bins(start, end)
    return [item for item in bins if item[3] > start and item[2] < end]


def _exact_cell(start: datetime, end: datetime) -> tuple[str, str, datetime, datetime] | None:
    """区间本身就是某个标准时间格时，观测对这个划分可测，计划退化为单点。"""
    for entity_id, level, left, right in candidate_bins(start):
        if left == start and right == end:
            return entity_id, level, left, right
    return None


def selected_bins(start: datetime, end: datetime) -> list[tuple[str, str, datetime, datetime]]:
    exact = _exact_cell(start, end)
    if exact is not None:
        return [exact]
    return overlapping_bins(start, end)


def imut_align(start: datetime, end: datetime, eta_hours: float = ETA_HOURS) -> TransportPlan:
    """一条区间记录的适配非平衡传输计划。"""
    if start.tzinfo is None:
        start = start.replace(tzinfo=TZ)
    if end.tzinfo is None:
        end = end.replace(tzinfo=TZ)
    if end <= start:
        end = start + timedelta(seconds=1)
    bins = selected_bins(start, end)
    hard_id, hard_level = finest_time_entity(start, end)
    if not bins:
        return TransportPlan((), "", (), (), 0.0, hard_id, hard_level)
    overlaps = []
    weights = []
    for _, _, left, right in bins:
        lo = max(start, left)
        hi = min(end, right)
        seconds = max((hi - lo).total_seconds(), 0.0)
        overlaps.append((lo, hi))
        weights.append(seconds)
    weight = np.array(weights, dtype=float)
    total = float(weight.sum())
    if total <= 0:
        return TransportPlan((), bins[0][1], (), (), 0.0, hard_id, hard_level)
    retained = retention_mass(to_unix(start), to_unix(end), eta_hours)
    mass = weight / total * retained
    return TransportPlan(
        tuple(item[0] for item in bins),
        bins[0][1],
        tuple(float(value) for value in mass),
        tuple((float(to_unix(lo)), float(to_unix(hi))) for lo, hi in overlaps),
        retained,
        hard_id,
        hard_level,
    )


def imut_align_unix(start_ts: float, end_ts: float, eta_hours: float = ETA_HOURS) -> TransportPlan:
    return imut_align(_from_unix(start_ts), _from_unix(end_ts), eta_hours)


def adapted_succession_mass(earlier: TransportPlan, later: TransportPlan, window: float = DAY) -> float:
    """因果切片上的联合质量。

    只累加「前一段整体早于后一段，且两段并起来仍落在 window 秒内」的质量积。
    支撑用的是区间与时间格的交集，也就是零位移耦合实际占用的那一段。
    """
    score = 0.0
    for mass_a, (start_a, end_a) in zip(earlier.mass, earlier.bounds):
        for mass_b, (start_b, end_b) in zip(later.mass, later.bounds):
            if end_a <= start_b and start_b > start_a and (end_b - start_a) <= window:
                score += mass_a * mass_b
    return float(score)
