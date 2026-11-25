"""IMUT 的闭式距离、保留质量和适配相继。"""

from datetime import datetime, timedelta

from tkg_pipeline.align import TZ, certainly_successive, to_unix
from tkg_pipeline.imut import (
    adapted_succession_mass,
    imut_align,
    rectangle_w2_squared,
    w1_to_barycenter,
    w2_squared_uniforms,
)


def _at(year, month, day, hour=0, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=TZ)


def test_w2_matches_known_cases():
    assert w2_squared_uniforms(0, 2, 0, 2) == 0.0
    # 质点在均匀分布中心时，距离等于该均匀分布的方差
    assert abs(w2_squared_uniforms(1, 1, 0, 2) - (4 / 12)) < 1e-9
    assert abs(w1_to_barycenter(0, 8) - 2) < 1e-12
    # 点在正方形中心：每个坐标的方差是边长平方的 1/12
    assert abs(rectangle_w2_squared(250, 250, 0, 500, 0, 500) - 2 * (500**2) / 12) < 1e-6


def test_hour_interval_peaks_on_that_hour():
    start = _at(2017, 8, 20, 17)
    plan = imut_align(start, start + timedelta(hours=1))
    assert plan.top_entity_id == "T2017082017"
    assert plan.hard_entity_id == "T2017082017"
    assert plan.retained > 0.95
    assert plan.support == 1


def test_period_splits_across_hours_instead_of_collapsing_to_a_day():
    start = _at(2018, 3, 28, 14)
    end = start + timedelta(hours=4)
    plan = imut_align(start, end)
    assert plan.level == "小时"
    assert plan.hard_level == "日"
    assert plan.support == 4
    assert abs(sum(plan.mass) - plan.retained) < 1e-9
    assert plan.retained > 0.9
    assert abs(max(plan.mass) - min(plan.mass)) < 1e-9


def test_long_span_discards_mass():
    start = _at(2013, 3, 1)
    end = _at(2013, 6, 1)
    plan = imut_align(start, end)
    assert plan.level == "月"
    assert plan.hard_entity_id == ""
    assert plan.retained < 0.08
    assert plan.support >= 2


def test_succession_mass_follows_the_causal_constraint():
    first = _at(2018, 1, 2, 10)
    second = _at(2018, 1, 2, 15)
    early = imut_align(first, first + timedelta(hours=1))
    late = imut_align(second, second + timedelta(hours=1))
    mass = adapted_succession_mass(early, late)
    assert mass > 0.9
    assert certainly_successive(to_unix(first), to_unix(first + timedelta(hours=1)), to_unix(second), to_unix(second + timedelta(hours=1)))

    day = _at(2018, 1, 2)
    day_plan = imut_align(day, day + timedelta(days=1))
    next_day = imut_align(day + timedelta(days=1), day + timedelta(days=2))
    soft = adapted_succession_mass(day_plan, next_day)
    assert day_plan.top_entity_id == "TD20180102"
    assert soft < 1e-6
    assert not certainly_successive(
        to_unix(day),
        to_unix(day + timedelta(days=1)),
        to_unix(day + timedelta(days=1)),
        to_unix(day + timedelta(days=2)),
    )

    afternoon = imut_align(_at(2018, 1, 2, 14), _at(2018, 1, 2, 18))
    next_afternoon = imut_align(_at(2018, 1, 3, 14), _at(2018, 1, 3, 18))
    partial = adapted_succession_mass(afternoon, next_afternoon)
    assert 0.05 < partial < 0.5
    assert not certainly_successive(
        to_unix(_at(2018, 1, 2, 14)),
        to_unix(_at(2018, 1, 2, 18)),
        to_unix(_at(2018, 1, 3, 14)),
        to_unix(_at(2018, 1, 3, 18)),
    )
