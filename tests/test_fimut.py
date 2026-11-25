"""FIMUT 的滤子一致性、节假日属性和网格表示误差。"""

from datetime import datetime, timedelta

from tkg_pipeline.align import TZ, to_unix
import numpy as np

from tkg_pipeline.fimut import (
    contact_score,
    fimut_time,
    fimut_time_tilted,
    grid_w2_meters,
    hour_occupancy,
    restricted_mass,
    spatial_hierarchy,
    start_close_counts,
    tie_hit_probability,
    uniform_close_probability,
    uniform_close_probability_many,
    with_space,
)


def _at(year, month, day, hour=0):
    return datetime(year, month, day, hour, tzinfo=TZ)


def test_afternoon_rolls_hours_up_to_one_day():
    plan = fimut_time(_at(2018, 3, 28, 14), _at(2018, 3, 28, 18))
    hours = plan.masses("小时")
    days = plan.masses("日")
    assert plan.finest_level == "小时"
    assert len(hours) == 4
    assert set(days) == {"TD20180328"}
    assert abs(sum(hours.values()) - plan.retained) < 1e-9
    assert abs(sum(days.values()) - plan.retained) < 1e-9
    assert abs(plan.level_total("周") - plan.retained) < 1e-9
    assert abs(plan.level_total("月") - plan.retained) < 1e-9
    assert abs(plan.level_total("旬") - plan.retained) < 1e-9
    day = next(atom for atom in plan.atoms if atom.level == "日")
    rows = plan.quadruples("C0101")
    day_row = next(row for row in rows if row["entity2_id"] == "TD20180328")
    hour_rows = [row for row in rows if row["level"] == "小时"]
    assert abs(day_row["confidence"] - sum(row["confidence"] for row in hour_rows)) < 1e-9
    assert set(day_row) >= {"entity1_id", "relation_type", "entity2_id", "iso8601_timestamp", "confidence"}


def test_exact_day_is_not_split_into_hours_and_keeps_the_holiday():
    plan = fimut_time(_at(2019, 2, 5), _at(2019, 2, 6))
    assert plan.finest_level == "日"
    assert plan.masses("小时") == {}
    assert plan.masses("日")["TD20190205"] > 0.9
    day = next(atom for atom in plan.atoms if atom.entity_id == "TD20190205")
    assert day.holiday == "春节"
    assert abs(plan.level_total("月") - plan.level_total("日")) < 1e-9


def test_cross_month_days_add_up_to_each_month():
    plan = fimut_time(_at(2018, 1, 30), _at(2018, 2, 4))
    january_days = sum(mass for entity_id, mass in plan.masses("日").items() if entity_id.startswith("TD201801"))
    february_days = sum(mass for entity_id, mass in plan.masses("日").items() if entity_id.startswith("TD201802"))
    assert abs(january_days - plan.masses("月")["TM201801"]) < 1e-8
    assert abs(february_days - plan.masses("月")["TM201802"]) < 1e-8
    assert abs(january_days + february_days - plan.retained) < 1e-8


def test_long_span_stays_on_months_and_drops_the_quadruple():
    plan = fimut_time(_at(2013, 3, 1), _at(2013, 6, 1))
    assert plan.finest_level == "月"
    assert plan.masses("日") == {}
    assert plan.retained < 0.08
    assert plan.quadruples("C0101") == []


def test_contact_score_decays_inside_the_window_and_stops_outside():
    query = to_unix(_at(2018, 4, 8, 12))
    window = 7 * 24 * 3600
    recent = contact_score(query - 3600, query, query, window)
    older = contact_score(query - 6 * 24 * 3600, query - 6 * 24 * 3600 + 3600, query, window)
    covered = contact_score(query - 30 * 24 * 3600, query, query, window)
    assert recent > 0.99
    assert older < recent
    assert abs(covered - 2.718281828 ** -0.5) < 1e-6
    assert contact_score(query - 10 * 24 * 3600, query - 8 * 24 * 3600, query, window) == 0.0


def test_window_mass_uses_only_the_overlapping_piece():
    start = to_unix(_at(2018, 3, 28, 14))
    end = to_unix(_at(2018, 3, 28, 18))
    full = restricted_mass(start, end, start, end)
    half = restricted_mass(start, end, start, to_unix(_at(2018, 3, 28, 16)))
    assert abs(full - 2 * half) < 1e-9
    assert full > 0.99
    assert restricted_mass(start, end, end, end + 3600) == 0.0
    assert tie_hit_probability(0, 1, 10) == 1.0
    assert abs(tie_hit_probability(0, 30, 10) - 10 / 30) < 1e-12
    assert tie_hit_probability(10, 1, 10) == 0.0


def test_grid_containment_reports_w2_without_moving_the_point():
    center = grid_w2_meters(250, 250, 0, 0, 0, 0)
    corner = grid_w2_meters(0, 0, 0, 0, 0, 0)
    assert center < corner
    assert abs(center - (500 / (6 ** 0.5))) < 1e-6
    plan = fimut_time(_at(2018, 3, 28, 17), _at(2018, 3, 28, 18))
    linked = with_space(plan, spatial_hierarchy("GG330100019022", "330000", "330100", "330102", 0.98), center)
    rows = linked.quadruples("C0101")
    spatial = [row for row in rows if row["level"] == "城市网格"]
    assert spatial[0]["entity2_id"] == "GG330100019022"
    assert spatial[0]["confidence"] == 0.98
    assert {row["entity2_id"] for row in rows if row["level"] in {"省级", "地级", "县级"}} == {"GP330000", "GC330100", "GD330102"}
    empty = with_space(plan, spatial_hierarchy("", "330000", "330100", "330102", 0.2), None)
    assert all(row["level"] != "城市网格" for row in empty.quadruples("C0101"))
    assert all(row["level"] not in {"省级", "地级", "县级"} for row in empty.quadruples("C0101"))


def _residual(plan) -> float:
    totals = [abs(plan.level_total(level) - plan.retained) for level in ("小时", "日", "周", "旬", "月")]
    return max(totals)


def test_flat_tilt_matches_the_uniform_hour_plan():
    start, end = _at(2018, 3, 28, 14), _at(2018, 3, 28, 18)
    plain = fimut_time(start, end)
    tilted = fimut_time_tilted(start, end, np.ones(24))
    assert tilted.finest_level == "小时"
    assert abs(tilted.retained - plain.retained) < 1e-12
    assert _residual(tilted) < 1e-9
    for entity_id, mass in plain.masses("小时").items():
        assert abs(tilted.masses("小时")[entity_id] - mass) < 1e-9
    assert abs(tilted.level_total("日") - sum(tilted.masses("小时").values())) < 1e-9
    scaled = fimut_time_tilted(start, end, np.full(24, 5.0))
    for entity_id, mass in tilted.masses("小时").items():
        assert abs(scaled.masses("小时")[entity_id] - mass) < 1e-9


def test_tilt_pushes_forward_and_moves_mass_across_midnight():
    start, end = _at(2018, 1, 1, 22), _at(2018, 1, 2, 8)
    weight = np.ones(24)
    weight[2] = 10.0
    plan = fimut_time_tilted(start, end, weight)
    assert _residual(plan) < 1e-9
    hours = plan.masses("小时")
    first = sum(mass for entity_id, mass in hours.items() if entity_id.startswith("T20180101"))
    second = sum(mass for entity_id, mass in hours.items() if entity_id.startswith("T20180102"))
    assert abs(first - plan.masses("日")["TD20180101"]) < 1e-9
    assert abs(second - plan.masses("日")["TD20180102"]) < 1e-9
    uniform = fimut_time_tilted(start, end, np.ones(24))
    assert abs(uniform.masses("日")["TD20180101"] / uniform.retained - 2 / 10) < 1e-9
    assert plan.masses("日")["TD20180102"] / plan.retained > uniform.masses("日")["TD20180102"] / uniform.retained
    occupancy = hour_occupancy(start, end, weight)
    assert abs(occupancy.sum() - 1) < 1e-12
    assert abs(occupancy[2] - 10 / 19) < 1e-9
    partial = hour_occupancy(_at(2018, 3, 28, 14).replace(minute=30), _at(2018, 3, 28, 16))
    assert abs(partial[14] - 1 / 3) < 1e-9
    assert abs(partial[15] - 2 / 3) < 1e-9
    assert partial[:14].sum() < 1e-12
    assert partial[16:].sum() < 1e-12


def test_uniform_close_probability_matches_known_areas_and_simulation():
    assert abs(uniform_close_probability(0, 1, 0, 1, 0.5) - 0.75) < 1e-12
    base = 1_500_000_000.0
    assert abs(uniform_close_probability(base, base + 1, base, base + 1, 0.5) - 0.75) < 1e-9
    assert uniform_close_probability(0, 1, 10, 11, 2) == 0.0
    assert uniform_close_probability(0, 1, 1.5, 2.5, 0.5) == 0.0
    assert uniform_close_probability(0, 1, 2, 3, 5) == 1.0
    assert uniform_close_probability(5, 5, 5, 5, 0) == 1.0
    assert uniform_close_probability(5, 5, 8, 8, 2) == 0.0
    assert uniform_close_probability(5, 5, 8, 8, 3) == 1.0
    assert abs(uniform_close_probability(0, 0, 0, 10, 2) - 0.2) < 1e-12
    assert start_close_counts([0, 10, 30], (15, 25, 30)) == [1, 2, 3]
    rng = np.random.default_rng(1)
    left = rng.uniform(0, 10, 24)
    width = rng.uniform(0.2, 4, 24)
    right = rng.uniform(0, 10, 24)
    other = rng.uniform(0.2, 4, 24)
    predicted = uniform_close_probability_many(left, left + width, right, right + other, 1.5)
    for index in range(len(left)):
        draw_u = rng.uniform(left[index], left[index] + width[index], 12000)
        draw_v = rng.uniform(right[index], right[index] + other[index], 12000)
        observed = float(np.mean(np.abs(draw_u - draw_v) <= 1.5))
        assert abs(observed - predicted[index]) < 0.02
