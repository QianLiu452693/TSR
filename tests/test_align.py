"""时空对齐的区间、实体粒度和相继判定。"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from tkg_pipeline.align import (
    TZ,
    align_time,
    certainly_successive,
    finest_time_entity,
    score_spatial,
    time_parent_links,
    to_iso,
    to_unix,
)

HOUR = 3600


def test_clock_and_period_keep_the_stated_precision():
    hour = align_time("2017年8月20日17时许", None)
    assert hour is not None
    assert to_iso(hour.anchor) == "2017-08-20T17:00:00+08:00"
    assert hour.grain == "hour"
    assert hour.entity_level == "小时"
    assert hour.entity_id == "T2017082017"
    assert hour.width_seconds == HOUR

    period = align_time("2018年3月28日下午", None)
    assert period is not None
    assert to_iso(period.anchor) == "2018-03-28T15:00:00+08:00"
    assert period.grain == "period"
    assert period.entity_level == "日"
    assert period.entity_id == "TD20180328"
    assert all(not (child.startswith("T") and child[1].isdigit()) for child, _ in time_parent_links(period.start, period.end))

    minute = align_time("2018年7月10日下午1时30分许", None)
    assert minute is not None
    assert to_iso(minute.anchor) == "2018-07-10T13:30:00+08:00"
    assert minute.grain == "minute"
    assert minute.entity_id == "T2018071013"

    exact = align_time("2018年2月16日16时40分", None)
    assert exact is not None and to_iso(exact.anchor) == "2018-02-16T16:40:00+08:00"

    numeric = align_time("2017-08-20 17:00:00", None)
    assert numeric is not None and to_iso(numeric.anchor) == "2017-08-20T17:00:00+08:00"


def test_coarse_text_is_not_collapsed_to_an_hour():
    ten = align_time("2017年12月上旬某日下午", None)
    assert ten is not None
    assert to_iso(ten.anchor) == "2017-12-05T15:00:00+08:00"
    assert ten.grain == "ten_day"
    assert ten.entity_id == "TN2017121"
    assert ten.entity_level == "旬"
    assert ten.end - ten.start == timedelta(days=10)

    month = align_time("2016年4月份某日", None)
    assert month is not None
    assert month.grain == "month"
    assert month.entity_level == "月"
    assert month.entity_id == "TM201604"

    span = align_time("2013年3月至5月间", None)
    assert span is not None
    assert span.grain == "span"
    assert span.entity_id == ""
    assert span.start == datetime(2013, 3, 1, tzinfo=TZ)
    assert span.end == datetime(2013, 6, 1, tzinfo=TZ)

    days = align_time("2017年1月10日至12日", None)
    assert days is not None
    assert days.end - days.start == timedelta(days=3)
    assert days.entity_level == "周"
    assert days.entity_id.startswith("TW")


def test_formatted_hour_confirms_but_does_not_shrink_coarser_text():
    agreed = align_time("2018年3月28日下午", "2018-03-28 15:00:00")
    assert agreed is not None
    assert agreed.source == "agree"
    assert agreed.entity_level == "日"
    assert agreed.confidence > 0.82

    finer = align_time("2018年2月16日16时40分", "2018-02-16 16:00:00")
    assert finer is not None
    assert finer.grain == "minute"
    assert finer.source == "agree"

    conflict = align_time("2017年8月20日17时许", "2017-08-20 04:00:00")
    assert conflict is not None
    assert conflict.source == "conflict"
    assert conflict.anchor.hour == 17
    assert conflict.confidence < 0.95

    selected = align_time(
        "2018年3月28日下午,2018年4月1日下午2时许",
        "2018-04-01 14:00:00",
    )
    assert selected is not None
    assert selected.source == "mention_selected"
    assert selected.anchor.day == 1
    assert selected.grain == "hour"


def test_extra_digits_and_out_of_window_years_do_not_stretch_the_interval():
    garbled = align_time("2018年4月7日凌晨4时左右;20018年4月7日凌晨4时左右", None)
    assert garbled is not None
    assert garbled.start.year == 2018
    assert garbled.end.year == 2018
    typo = align_time("2014年9月24日凌晨4时许,20014年9月28日下午", None)
    assert typo is not None
    assert typo.start.year == 2014 and typo.end.year == 2014
    assert align_time("1998年12月", None) is None


def test_clock_without_a_date_uses_the_formatted_day():
    bound = align_time("17时许", "2017-08-20 17:00:00")
    assert bound is not None
    assert to_iso(bound.anchor) == "2017-08-20T17:00:00+08:00"
    assert bound.source == "agree"
    repaired = align_time("2O12年8月28日1时许", None)
    assert repaired is not None and repaired.anchor.year == 2012
    masked = align_time("20XX年X月XX日22时许", "2018-01-07 21:00:00")
    assert masked is not None and masked.source == "formatted"


def test_succession_requires_the_whole_interval():
    hour = align_time("2018年1月2日10时许", None)
    later = align_time("2018年1月2日15时许", None)
    day = align_time("2018年1月2日", None)
    assert hour and later and day
    assert certainly_successive(to_unix(hour.start), to_unix(hour.end), to_unix(later.start), to_unix(later.end))
    assert not certainly_successive(to_unix(day.start), to_unix(day.end), to_unix(later.start), to_unix(later.end))
    nxt = align_time("2018年1月3日", None)
    assert nxt is not None
    assert not certainly_successive(to_unix(day.start), to_unix(day.end), to_unix(nxt.start), to_unix(nxt.end))


def test_place_name_changes_spatial_confidence():
    frame = pd.DataFrame(
        [
            {"admin_match": "pip", "admin_distance_m": 0, "incident_county": "西湖区", "incident_city": "杭州市", "incident_province": "浙江省", "admin_name": "西湖区", "city_name": "杭州市", "province_name": "浙江省"},
            {"admin_match": "pip", "admin_distance_m": 0, "incident_county": "拱墅区", "incident_city": "杭州市", "incident_province": "浙江省", "admin_name": "西湖区", "city_name": "杭州市", "province_name": "浙江省"},
            {"admin_match": "nearest", "admin_distance_m": 3200, "incident_county": "", "incident_city": "", "incident_province": "", "admin_name": "西湖区", "city_name": "杭州市", "province_name": "浙江省"},
        ]
    )
    scored = score_spatial(frame)
    assert scored.loc[0, "place_agreement"] == "agree"
    assert scored.loc[0, "spatial_align_confidence"] == 0.99
    assert scored.loc[1, "place_agreement"] == "conflict"
    assert scored.loc[1, "spatial_align_confidence"] < scored.loc[0, "spatial_align_confidence"]
    assert scored.loc[2, "place_agreement"] == "missing_text"
    assert scored.loc[2, "spatial_align_confidence"] == 0.62
    assert finest_time_entity(datetime(2018, 1, 1, tzinfo=TZ), datetime(2018, 1, 1, 1, tzinfo=TZ))[1] == "小时"
