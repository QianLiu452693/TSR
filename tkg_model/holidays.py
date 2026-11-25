"""公历固定节日、周末与春节日期。只标注节日当天。"""

from __future__ import annotations

from datetime import date, datetime

# 农历正月初一对应的公历日期。
SPRING_FESTIVAL = {
    2000: date(2000, 2, 5),
    2001: date(2001, 1, 24),
    2002: date(2002, 2, 12),
    2003: date(2003, 2, 1),
    2004: date(2004, 1, 22),
    2005: date(2005, 2, 9),
    2006: date(2006, 1, 29),
    2007: date(2007, 2, 18),
    2008: date(2008, 2, 7),
    2009: date(2009, 1, 26),
    2010: date(2010, 2, 14),
    2011: date(2011, 2, 3),
    2012: date(2012, 1, 23),
    2013: date(2013, 2, 10),
    2014: date(2014, 1, 31),
    2015: date(2015, 2, 19),
    2016: date(2016, 2, 8),
    2017: date(2017, 1, 28),
    2018: date(2018, 2, 16),
    2019: date(2019, 2, 5),
}

PERIODS = (
    (0, 5, "凌晨"),
    (6, 8, "早晨"),
    (9, 11, "上午"),
    (12, 13, "中午"),
    (14, 17, "下午"),
    (18, 19, "傍晚"),
    (20, 23, "夜间"),
)


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(str(value))


def period_name(hour: int) -> str:
    for start, end, name in PERIODS:
        if start <= hour <= end:
            return name
    return "未知"


def holiday_name(day: date) -> str:
    if (day.month, day.day) == (1, 1):
        return "元旦"
    if (day.month, day.day) == (5, 1):
        return "劳动节"
    if (day.month, day.day) == (10, 1):
        return "国庆节"
    if SPRING_FESTIVAL.get(day.year) == day:
        return "春节"
    return ""


def time_features(moment: datetime) -> dict:
    day = moment.date()
    name = holiday_name(day)
    iso_year, iso_week, _ = day.isocalendar()
    return {
        "hour_id": "T" + moment.strftime("%Y%m%d%H"),
        "day_id": "TD" + moment.strftime("%Y%m%d"),
        "week_id": f"TW{iso_year}{iso_week:02d}",
        "month_id": "TM" + moment.strftime("%Y%m"),
        "period": period_name(moment.hour),
        "is_weekend": day.weekday() >= 5,
        "is_holiday": bool(name),
        "holiday_name": name,
        "iso_year": iso_year,
        "iso_week": iso_week,
    }
