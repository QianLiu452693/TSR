"""犯罪记录到时序知识图谱的时空对齐。

时间按“解析—补全—归一”处理，但补全出来的代表时刻只用于 ISO 8601 锚点。
图谱里的时间实体是能完整包住不确定区间的最细粒度，短时序边只在两个区间
必定有先后、且任意一对时刻都不超过 24 小时时才建立。

formatted_datetime 在 Figshare 28106939 中分和秒恒为 00，按小时区间解释，
不能把它当成秒级观测。地名与坐标的一致性单独记入空间对齐置信度。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

TZ = timezone(timedelta(hours=8))
SUCCESSION_WINDOW = 24 * 3600

_PERIODS = (
    ("凌晨", 0, 5, 2),
    ("清晨", 6, 8, 6),
    ("早晨", 6, 9, 7),
    ("早上", 7, 10, 8),
    ("上午", 8, 12, 9),
    ("中午", 11, 14, 12),
    ("午后", 14, 16, 14),
    ("下午", 14, 18, 15),
    ("黄昏", 17, 20, 18),
    ("傍晚", 18, 20, 18),
    ("晚上", 18, 23, 20),
    ("晚间", 18, 23, 20),
    ("夜里", 20, 24, 21),
    ("夜间", 20, 24, 21),
    ("深夜", 22, 24, 23),
    ("晚", 18, 23, 20),
)
_PM = ("下午", "傍晚", "黄昏", "晚上", "晚间", "夜间", "夜里", "深夜", "午后")
_YMD = re.compile(r"(?<!\d)(\d{4})(?!\d)\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_YM = re.compile(r"(?<!\d)(\d{4})(?!\d)\s*年\s*(\d{1,2})\s*月")
STUDY_START = datetime(2000, 1, 1, tzinfo=TZ)
STUDY_END = datetime(2020, 1, 1, tzinfo=TZ)
_MD = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_DAY = re.compile(r"(\d{1,2})\s*日")
_MONTH = re.compile(r"(\d{1,2})\s*月")
_ISO = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?")
_SLASH = re.compile(r"(\d{4})[./](\d{1,2})[./](\d{1,2})(?:\s+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?")
_HMS = re.compile(r"(\d{1,2})\s*(?:时|点)\s*(?:(\d{1,2})\s*分)?\s*(?:(\d{1,2})\s*秒)?")
_EMPTY = {"", "nan", "nat", "none", "<na>"}


@dataclass(frozen=True)
class TimeAlignment:
    start: datetime
    end: datetime
    anchor: datetime
    grain: str
    imputed: bool
    confidence: float
    source: str = "incident"
    entity_id: str = ""
    entity_level: str = ""

    @property
    def width_seconds(self) -> float:
        return (self.end - self.start).total_seconds()


def _within_study(item: TimeAlignment) -> bool:
    """丢弃研究窗口以外的表述，避免 20018年、1998年 把整条记录拉出 2000–2019。"""
    return STUDY_START <= item.start < STUDY_END and item.end <= STUDY_END


def to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt.astimezone(TZ).strftime("%Y-%m-%dT%H:%M:%S+08:00")


def to_unix(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return int(dt.timestamp())


def hour_token(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%Y%m%d%H")


def confidence_for_width(seconds: float) -> float:
    if seconds <= 1.5:
        return 0.99
    if seconds <= 90:
        return 0.97
    if seconds <= 3601:
        return 0.95
    if seconds <= 6 * 3600:
        return 0.82
    if seconds <= 86401:
        return 0.70
    if seconds <= 10 * 86400:
        return 0.55
    if seconds <= 32 * 86400:
        return 0.42
    if seconds <= 93 * 86400:
        return 0.35
    return 0.28


def certainly_successive(start_a: int, end_a: int, start_b: int, end_b: int, window: int = SUCCESSION_WINDOW) -> bool:
    """后一区间是否必定发生在前一区间之后，且任意一对时刻都不超过 window 秒。"""
    if end_a < start_a or end_b < start_b:
        return False
    if start_a == end_a and start_b == end_b:
        delta = start_b - start_a
        return 0 < delta <= window
    if start_b < end_a or start_b <= start_a:
        return False
    return (end_b - start_a) <= window


def certain_successor_indices(start: np.ndarray, end: np.ndarray, window: int = SUCCESSION_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """start 已按升序排列。每个事件最多连向其后第一个必定相继的事件。"""
    start = np.asarray(start, dtype=np.int64)
    end = np.asarray(end, dtype=np.int64)
    left: list[int] = []
    right: list[int] = []
    n = int(start.shape[0])
    for i in range(n):
        limit = int(start[i]) + window
        j = i + 1
        while j < n and int(start[j]) <= limit:
            if certainly_successive(int(start[i]), int(end[i]), int(start[j]), int(end[j]), window):
                left.append(i)
                right.append(j)
                break
            j += 1
    if not left:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def _preclean(text: str) -> str:
    text = str(text).strip().replace("〇", "0").replace("○", "0")
    text = re.sub(r"(?<=\d)[Oo](?=\d)", "0", text)
    text = text.replace("月份", "月")
    return text


def _at(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime | None:
    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=TZ)
    except ValueError:
        return None


def _month_last(year: int, month: int) -> int:
    nxt = _add_months(year, month, 1)
    if nxt is None:
        return 28
    return (nxt - timedelta(days=1)).day


def _add_months(year: int, month: int, count: int) -> datetime | None:
    month_index = year * 12 + (month - 1) + count
    return _at(month_index // 12, month_index % 12 + 1, 1)


def _period(text: str) -> tuple[int, int, int] | None:
    for word, start, end, rep in _PERIODS:
        if word in text:
            return start, end, rep
    return None


def _apply_clock(text: str, hour: int, minute: int) -> tuple[int, int]:
    if any(word in text for word in _PM) and hour < 12:
        hour += 12
    if "中午" in text and hour < 11:
        hour = 12
    if hour == 24:
        hour = 23
    return hour, minute


@dataclass
class _Mention:
    start: datetime | None
    end: datetime | None
    anchor: datetime | None
    grain: str
    imputed: bool
    has_date: bool
    year: int | None = None
    month: int | None = None
    clock: tuple[int, int, int, str] | None = None
    period: tuple[int, int, int] | None = None

    def materialize(self) -> TimeAlignment | None:
        if not self.has_date or self.start is None or self.end is None or self.anchor is None:
            return None
        if self.end <= self.start:
            return None
        anchor = self.anchor if self.start <= self.anchor < self.end else self.start
        width = (self.end - self.start).total_seconds()
        return TimeAlignment(
            self.start,
            self.end,
            anchor,
            self.grain,
            self.imputed,
            confidence_for_width(width),
        )


def _clock_bin(day: datetime, hour: int, minute: int, second: int, stated_minute: bool, stated_second: bool) -> tuple[datetime, datetime, datetime, str, bool] | None:
    hour, minute = max(0, min(23, hour)), max(0, min(59, minute))
    second = max(0, min(59, second))
    anchor = day.replace(hour=hour, minute=minute if stated_minute else 0, second=second if stated_second else 0)
    if stated_second:
        return anchor, anchor + timedelta(seconds=1), anchor, "second", False
    if stated_minute:
        anchor = anchor.replace(second=0)
        return anchor, anchor + timedelta(minutes=1), anchor, "minute", False
    anchor = anchor.replace(minute=0, second=0)
    return anchor, anchor + timedelta(hours=1), anchor, "hour", False


def _interpret(text: str, year: int | None, month: int | None) -> _Mention | None:
    text = text.strip()
    if not text or re.search(r"[xX×]", text):
        return None
    around = "左右" in text or "前后" in text
    period = _period(text)
    clock = _HMS.search(text)
    stated_minute = bool(clock and clock.group(2) is not None)
    stated_second = bool(clock and clock.group(3) is not None)
    hour = minute = second = None
    if clock:
        hour, minute = _apply_clock(text, int(clock.group(1)), int(clock.group(2) or 0))
        second = int(clock.group(3) or 0)

    found_year = found_month = found_day = None
    day_lo = day_hi = None
    matched = _YMD.search(text)
    if matched:
        found_year, found_month, found_day = (int(matched.group(i)) for i in range(1, 4))
    else:
        matched = _ISO.search(text) or _SLASH.search(text)
        if matched:
            found_year, found_month, found_day = (int(matched.group(i)) for i in range(1, 4))
            if matched.group(4) is not None and hour is None:
                hour, minute = _apply_clock(text, int(matched.group(4)), int(matched.group(5) or 0))
                second = int(matched.group(6) or 0)
                stated_minute = matched.group(5) is not None
                stated_second = matched.group(6) is not None
        else:
            matched = _YM.search(text)
            if matched:
                found_year, found_month = int(matched.group(1)), int(matched.group(2))
            else:
                matched = _MD.search(text)
                if matched and year is not None:
                    found_year = year
                    found_month, found_day = int(matched.group(1)), int(matched.group(2))
                else:
                    matched = _DAY.search(text)
                    if matched and year is not None and month is not None:
                        found_year, found_month, found_day = year, month, int(matched.group(1))
                    else:
                        matched = _MONTH.search(text)
                        if matched and year is not None:
                            found_year, found_month = year, int(matched.group(1))

    known_month = found_month if found_month is not None else month
    known_year = found_year if found_year is not None else year
    if "上旬" in text and known_month is not None:
        day_lo, day_hi = 1, 10
    elif "中旬" in text and known_month is not None:
        day_lo, day_hi = 11, 20
    elif "下旬" in text and known_month is not None:
        day_lo, day_hi = 21, _month_last(known_year or 2000, known_month)
    elif ("月初" in text) and known_month is not None:
        day_lo, day_hi = 1, 5
    elif ("月底" in text or "月末" in text) and known_month is not None:
        last = _month_last(known_year or 2000, known_month)
        day_lo, day_hi = max(1, last - 4), last
    elif found_day is not None:
        day_lo = day_hi = found_day
    elif known_month is not None:
        day_lo, day_hi = 1, _month_last(known_year or 2000, known_month)

    use_year = found_year if found_year is not None else year
    use_month = found_month if found_month is not None else month
    if use_year is None or use_month is None or day_lo is None or day_hi is None:
        if hour is None and period is None:
            return None
        return _Mention(None, None, None, "hour", True, False, clock=(hour or 0, minute or 0, second or 0, "clock"), period=period)

    if around and found_day is not None and hour is None:
        day_lo = max(1, found_day - 1)
        day_hi = min(_month_last(use_year, use_month), found_day + 1)

    start = _at(use_year, use_month, day_lo)
    end_day = _at(use_year, use_month, day_hi)
    if start is None or end_day is None:
        return None
    end = end_day + timedelta(days=1)
    unique_day = day_lo == day_hi
    grain = "day"
    imputed = True
    anchor = start
    if found_month is not None and found_day is None and "旬" not in text and "月初" not in text and "月底" not in text and "月末" not in text:
        grain = "month"
    if "上旬" in text or "中旬" in text or "下旬" in text:
        grain = "ten_day"
        rep = 5 if day_lo == 1 else 15 if day_lo == 11 else 25
        rep = min(rep, day_hi)
        anchor = _at(use_year, use_month, rep) or start

    if unique_day and hour is not None:
        built = _clock_bin(start, hour, minute or 0, second or 0, stated_minute, stated_second)
        if built is None:
            return None
        anchor, end, _, grain, imputed = built
        start = anchor
    elif unique_day and period is not None and hour is None:
        p0, p1, prep = period
        anchor = start.replace(hour=prep)
        start = start.replace(hour=p0)
        end = end_day.replace(hour=0) + timedelta(hours=p1)
        grain = "period"
        imputed = True
    elif (not unique_day) and hour is not None and grain == "ten_day":
        anchor = (anchor or start).replace(hour=hour, minute=0, second=0)
    elif (not unique_day) and period is not None and grain == "ten_day":
        anchor = (anchor or start).replace(hour=period[2])

    if not unique_day and grain == "day":
        grain = "span"
    if anchor < start or anchor >= end:
        anchor = start
    return _Mention(start, end, anchor, grain, imputed, True, use_year, use_month, period=period)


def _materialize_dateless(text: str, hint: TimeAlignment) -> TimeAlignment | None:
    """只有钟点或时段、没有日期时，日期取 formatted_datetime 的公历日。"""
    period = _period(text)
    clock = _HMS.search(text)
    day = (hint.start if hint.grain == "hour" else hint.anchor).replace(hour=0, minute=0, second=0, microsecond=0)
    if clock:
        hour, minute = _apply_clock(text, int(clock.group(1)), int(clock.group(2) or 0))
        second = int(clock.group(3) or 0)
        built = _clock_bin(day, hour, minute, second, clock.group(2) is not None, clock.group(3) is not None)
        if built is None:
            return None
        anchor, end, _, grain, imputed = built
        return TimeAlignment(anchor, end, anchor, grain, imputed, confidence_for_width((end - anchor).total_seconds()))
    if period is None:
        return None
    p0, p1, prep = period
    start = day.replace(hour=p0)
    end = day + timedelta(hours=p1)
    anchor = day.replace(hour=prep)
    if anchor < start or anchor >= end:
        anchor = start
    return TimeAlignment(start, end, anchor, "period", True, confidence_for_width((end - start).total_seconds()))


def _parse_formatted(text: str | None) -> TimeAlignment | None:
    if text is None:
        return None
    raw = _preclean(text)
    if raw.lower() in _EMPTY:
        return None
    matched = _ISO.search(raw)
    if not matched:
        return None
    year, month, day = int(matched.group(1)), int(matched.group(2)), int(matched.group(3))
    if matched.group(4) is None:
        start = _at(year, month, day)
        if start is None:
            return None
        return TimeAlignment(start, start + timedelta(days=1), start, "day", True, 0.70, "formatted")
    hour = int(matched.group(4))
    minute = int(matched.group(5) or 0)
    second = int(matched.group(6) or 0)
    start = _at(year, month, day, 0 if hour == 24 else hour, 0, 0)
    if start is None:
        return None
    # 该字段在源数据集中被四舍五入到小时。分和秒都为 0 时按小时区间，不当作秒级事实。
    if minute == 0 and second == 0:
        aligned = TimeAlignment(start, start + timedelta(hours=1), start, "hour", False, 0.95, "formatted")
        return aligned if _within_study(aligned) else None
    if second == 0:
        anchor = start.replace(minute=minute)
        return TimeAlignment(anchor, anchor + timedelta(minutes=1), anchor, "minute", False, 0.97, "formatted")
    anchor = start.replace(minute=minute, second=second)
    return TimeAlignment(anchor, anchor + timedelta(seconds=1), anchor, "second", False, 0.99, "formatted")


def _mentions_from_incident(text: str | None) -> list[TimeAlignment | str]:
    if text is None:
        return []
    raw = _preclean(text)
    if raw.lower() in _EMPTY:
        return []
    year: int | None = None
    month: int | None = None
    found: list[TimeAlignment | str] = []
    for part in re.split(r"[,，;；]", raw):
        part = part.strip()
        if not part:
            continue
        if "同年" in part and year is not None:
            part = part.replace("同年", f"{year}年")
        if re.search(r"[xX×]", part):
            continue
        pieces = re.split(r"至|到", part, maxsplit=1)
        ranged = len(pieces) == 2 and re.search(r"\d", pieces[0]) and re.search(r"\d", pieces[1])
        if ranged:
            left = _interpret(pieces[0], year, month)
            if left and left.has_date:
                year, month = left.year, left.month
            right = _interpret(pieces[1], year, month)
            if left and right and left.has_date and right.has_date and left.start and right.end and right.end > left.start:
                anchor = left.anchor if left.anchor and left.start <= left.anchor < right.end else left.start
                width = (right.end - left.start).total_seconds()
                grain = "span" if width > 36 * 3600 else left.grain
                spanned = TimeAlignment(left.start, right.end, anchor, grain, True, confidence_for_width(width))
                if _within_study(spanned):
                    found.append(spanned)
                year, month = right.year, right.month
                continue
        one = _interpret(part, year, month)
        if one is None:
            continue
        if one.has_date and one.start is not None:
            year, month = one.year, one.month
            made = one.materialize()
            if made is not None and _within_study(made):
                found.append(made)
        else:
            found.append(part)
    return found


def _contains_instant(interval: TimeAlignment, moment: datetime) -> bool:
    return interval.start <= moment < interval.end


def _inside(inner: TimeAlignment, outer: TimeAlignment) -> bool:
    return outer.start <= inner.start and inner.end <= outer.end


def _with_source(alignment: TimeAlignment, source: str, bonus: float = 0.0, scale: float = 1.0) -> TimeAlignment:
    confidence = alignment.confidence * scale + bonus
    confidence = min(0.99, max(0.2, confidence))
    return replace(alignment, source=source, confidence=round(confidence, 4))


def _envelope(items: list[TimeAlignment], source: str) -> TimeAlignment:
    start = min(item.start for item in items)
    end = max(item.end for item in items)
    anchor = items[0].anchor if start <= items[0].anchor < end else start
    width = (end - start).total_seconds()
    return TimeAlignment(start, end, anchor, "span", True, confidence_for_width(width), source)


def _identify(alignment: TimeAlignment) -> TimeAlignment:
    entity_id, level = finest_time_entity(alignment.start, alignment.end)
    return replace(alignment, entity_id=entity_id, entity_level=level, confidence=round(alignment.confidence, 4))


def align_time(incident_time: str | None, formatted_datetime: str | None) -> TimeAlignment | None:
    """把原始时间表述对齐成一个带区间、粒度和置信度的时间对象。"""
    formatted = _parse_formatted(formatted_datetime)
    raw_mentions = _mentions_from_incident(incident_time)
    mentions: list[TimeAlignment] = []
    for item in raw_mentions:
        if isinstance(item, TimeAlignment):
            mentions.append(item)
            continue
        if formatted is not None:
            bound = _materialize_dateless(item, formatted)
            if bound is not None:
                mentions.append(bound)
    if not mentions and formatted is None:
        return None
    if not mentions and formatted is not None:
        return _identify(formatted)
    if formatted is None:
        chosen = mentions[0] if len(mentions) == 1 else _envelope(mentions, "narrative_span")
        source = "incident" if len(mentions) == 1 else "narrative_span"
        return _identify(_with_source(chosen, source))

    def agrees(mention: TimeAlignment) -> bool:
        return _contains_instant(mention, formatted.anchor) or _inside(mention, formatted) or _inside(formatted, mention)

    if len(mentions) == 1:
        mention = mentions[0]
        if agrees(mention):
            return _identify(_with_source(mention, "agree", bonus=0.02))
        return _identify(_with_source(mention, "conflict", scale=0.5))

    matched = [mention for mention in mentions if agrees(mention)]
    if matched:
        chosen = min(matched, key=lambda item: item.width_seconds)
        return _identify(_with_source(chosen, "mention_selected", bonus=0.02, scale=0.97))
    spanned = _envelope(mentions, "narrative_span")
    return _identify(_with_source(spanned, "narrative_span", scale=0.6))


def parse_raw_time(incident_time: str | None, formatted_datetime: str | None) -> tuple[datetime, str] | None:
    aligned = align_time(incident_time, formatted_datetime)
    if aligned is None:
        return None
    return aligned.anchor, aligned.grain


def _week_bounds(moment: datetime) -> tuple[datetime, datetime, str]:
    monday = (moment - timedelta(days=moment.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    iso = monday.isocalendar()
    return monday, monday + timedelta(days=7), f"TW{iso[0]}{iso[1]:02d}"


def candidate_bins(moment: datetime) -> list[tuple[str, str, datetime, datetime]]:
    """返回该时刻所在的小时、日、周、旬、月区间。"""
    hour0 = moment.replace(minute=0, second=0, microsecond=0)
    day0 = hour0.replace(hour=0)
    if moment.day <= 10:
        ten0, ten1 = day0.replace(day=1), day0.replace(day=11)
        ten_id = f"TN{moment.strftime('%Y%m')}1"
    elif moment.day <= 20:
        ten0, ten1 = day0.replace(day=11), day0.replace(day=21)
        ten_id = f"TN{moment.strftime('%Y%m')}2"
    else:
        ten0 = day0.replace(day=21)
        nxt = _add_months(moment.year, moment.month, 1)
        ten1 = nxt if nxt is not None else ten0 + timedelta(days=10)
        ten_id = f"TN{moment.strftime('%Y%m')}3"
    week0, week1, week_id = _week_bounds(moment)
    month0 = day0.replace(day=1)
    month1 = _add_months(moment.year, moment.month, 1) or (month0 + timedelta(days=31))
    return [
        (f"T{hour0.strftime('%Y%m%d%H')}", "小时", hour0, hour0 + timedelta(hours=1)),
        (f"TD{day0.strftime('%Y%m%d')}", "日", day0, day0 + timedelta(days=1)),
        (week_id, "周", week0, week1),
        (ten_id, "旬", ten0, ten1),
        (f"TM{month0.strftime('%Y%m')}", "月", month0, month1),
    ]


def contained_bins(start: datetime, end: datetime) -> list[tuple[str, str, datetime, datetime]]:
    if end < start:
        return []
    probe = start if end > start else start
    bins = []
    for entity_id, level, left, right in candidate_bins(probe):
        if left <= start and end <= right:
            bins.append((entity_id, level, left, right))
    bins.sort(key=lambda item: item[3] - item[2])
    return bins


def finest_time_entity(start: datetime, end: datetime) -> tuple[str, str]:
    bins = contained_bins(start, end)
    if not bins:
        return "", ""
    return bins[0][0], bins[0][1]


def time_parent_links(start: datetime, end: datetime) -> list[tuple[str, str]]:
    """只连接确实包含该区间的粒度，避免把补全钟点写成小时实体。"""
    levels = {level: entity_id for entity_id, level, _, _ in contained_bins(start, end)}
    links: list[tuple[str, str]] = []
    if "小时" in levels and "日" in levels:
        links.append((levels["小时"], levels["日"]))
    if "日" in levels and "月" in levels:
        links.append((levels["日"], levels["月"]))
    if "日" in levels and "周" in levels:
        links.append((levels["日"], levels["周"]))
    if "日" not in levels and "旬" in levels and "月" in levels:
        links.append((levels["旬"], levels["月"]))
    return links


def _norm_place(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        return ""
    text = str(value).strip()
    if text.lower() in _EMPTY:
        return ""
    for suffix in ("特别行政区", "维吾尔自治区", "回族自治区", "壮族自治区", "自治区", "自治州", "自治县", "地区", "林区", "盟", "省", "市", "区", "县", "旗"):
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)]
            break
    return text


def _place_status(county: object, city: object, province: object, admin: object, city_name: object, province_name: object) -> str:
    county_n, admin_n = _norm_place(county), _norm_place(admin)
    if county_n and admin_n:
        if county_n == admin_n or county_n in admin_n or admin_n in county_n:
            return "agree"
        return "conflict"
    city_n, city_ref = _norm_place(city), _norm_place(city_name)
    if city_n and city_ref:
        if city_n == city_ref or city_n in city_ref or city_ref in city_n:
            return "city_agree"
        return "conflict"
    prov_n, prov_ref = _norm_place(province), _norm_place(province_name)
    if prov_n and prov_ref:
        if prov_n == prov_ref or prov_n in prov_ref or prov_ref in prov_n:
            return "province_agree"
        return "conflict"
    return "missing_text"


def score_spatial(frame: pd.DataFrame) -> pd.DataFrame:
    """点面命中、近邻距离和案发地名称是否指向同一行政区，合成空间对齐置信度。"""
    out = frame.copy()
    base = out.get("admin_match", pd.Series(index=out.index, dtype="string")).map({"pip": 0.98, "nearest": 0.70, "court": 0.55}).astype(float)
    base = base.fillna(0.50)
    if "admin_distance_m" in out.columns:
        distance = pd.to_numeric(out["admin_distance_m"], errors="coerce")
        nearest = out["admin_match"].eq("nearest") if "admin_match" in out.columns else pd.Series(False, index=out.index)
        base.loc[nearest & distance.le(500)] = 0.86
        base.loc[nearest & distance.gt(500) & distance.le(2000)] = 0.74
        base.loc[nearest & distance.gt(2000)] = 0.62
    county = out["incident_county"] if "incident_county" in out.columns else pd.Series("", index=out.index)
    city = out["incident_city"] if "incident_city" in out.columns else pd.Series("", index=out.index)
    province = out["incident_province"] if "incident_province" in out.columns else pd.Series("", index=out.index)
    admin = out["admin_name"] if "admin_name" in out.columns else pd.Series("", index=out.index)
    city_name = out["city_name"] if "city_name" in out.columns else pd.Series("", index=out.index)
    province_name = out["province_name"] if "province_name" in out.columns else pd.Series("", index=out.index)
    status = [
        _place_status(a, b, c, d, e, f)
        for a, b, c, d, e, f in zip(county.tolist(), city.tolist(), province.tolist(), admin.tolist(), city_name.tolist(), province_name.tolist())
    ]
    agreement = pd.Series(status, index=out.index, dtype="string")
    confidence = base.to_numpy(dtype=float)
    labels = agreement.to_numpy()
    confidence = np.where(labels == "agree", np.minimum(0.99, confidence + 0.01), confidence)
    confidence = np.where(labels == "conflict", confidence * 0.75, confidence)
    out["place_agreement"] = agreement
    out["spatial_align_confidence"] = np.round(confidence, 4)
    return out
