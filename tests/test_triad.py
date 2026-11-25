"""三条对齐模块的格子、开关和读者。"""

from datetime import datetime, timedelta

import numpy as np

from tkg_pipeline.align import TZ, candidate_bins, finest_time_entity, to_unix
from tkg_pipeline.imut import selected_bins
from tkg_pipeline.triad import (
    DAY_L,
    HOUR_L,
    LEVEL_NAME,
    MONTH_L,
    WEEK_L,
    absolute_errors,
    build_geometry,
    fraction,
    interval_incidence,
    truth_fractions,
)


def _at(year, month, day, hour=0):
    return datetime(year, month, day, hour, tzinfo=TZ)


def _pair(start: datetime, end: datetime):
    return build_geometry([to_unix(start)], [to_unix(end)])


def test_exact_cells_match_the_hard_aligner():
    anchors = [
        _at(2016, 2, 21, 3),
        _at(2016, 2, 29, 0),
        _at(2018, 2, 28, 23),
        _at(2018, 3, 10, 0),
        _at(2018, 3, 11, 0),
        _at(2018, 3, 20, 12),
        _at(2018, 3, 21, 0),
        _at(2018, 12, 31, 23),
        _at(2019, 1, 1, 0),
        _at(2017, 5, 17, 14),
    ]
    for moment in anchors:
        for _entity, level, left, right in candidate_bins(moment):
            geo = _pair(left, right)
            assert LEVEL_NAME[int(geo.ladder[0])] == level
            assert LEVEL_NAME[int(geo.hard[0])] == level
            assert int(geo.hard_left[0]) == to_unix(left)
            assert int(geo.hard_right[0]) == to_unix(right)


def test_hard_level_and_ladder_match_existing_aligners():
    rng = np.random.default_rng(0)
    base = _at(2016, 1, 1)
    widths = [0, 1, 4, 24, 36, 48, 24 * 5, 24 * 8, 24 * 10, 24 * 12, 24 * 20, 24 * 31, 24 * 32, 24 * 40, 24 * 100]
    for _ in range(240):
        start = base + timedelta(days=int(rng.integers(0, 1400)), hours=int(rng.integers(0, 24)))
        width = int(rng.choice(widths))
        end = start if width == 0 else start + timedelta(hours=width)
        geo = _pair(start, end)
        _entity, level = finest_time_entity(start, end)
        bins = selected_bins(start, end)
        assert LEVEL_NAME[int(geo.hard[0])] == level
        assert LEVEL_NAME[int(geo.ladder[0])] == bins[0][1]
        if level:
            assert int(geo.hard_left[0]) == to_unix(contained_left(start, end))
            assert int(geo.hard_right[0]) == to_unix(contained_right(start, end))


def contained_left(start, end):
    from tkg_pipeline.align import contained_bins

    return contained_bins(start, end)[0][2]


def contained_right(start, end):
    from tkg_pipeline.align import contained_bins

    return contained_bins(start, end)[0][3]


def test_afternoon_readers_follow_the_three_switches():
    geo = _pair(_at(2018, 3, 28, 14), _at(2018, 3, 28, 18))
    assert int(geo.hard[0]) == DAY_L
    assert int(geo.ladder[0]) == HOUR_L
    truth = truth_fractions(geo)
    assert abs(float(truth["hour"][0]) - 0.25) < 1e-12
    assert abs(float(truth["day"][0]) - 1.0) < 1e-12
    assert abs(float(truth["month"][0]) - 1.0) < 1e-12
    assert abs(float(truth["window6"][0]) - 1.0) < 1e-12
    _truth, errors = absolute_errors(geo)
    # 只有支撑、没有推前：小时查询是对的，日和月实体没有被写出来。
    assert float(errors["100"]["hour"][0]) < 1e-12
    assert abs(float(errors["100"]["day"][0]) - 1.0) < 1e-12
    assert abs(float(errors["100"]["month"][0]) - 1.0) < 1e-12
    assert float(errors["100"]["window6"][0]) < 1e-12
    # 只有推前：日和月被硬格子抬上去，小时仍按整天摊开。
    assert abs(float(errors["010"]["hour"][0]) - 0.25) < 1e-12
    assert float(errors["010"]["day"][0]) < 1e-12
    assert float(errors["010"]["month"][0]) < 1e-12
    assert abs(float(errors["010"]["window6"][0]) - 0.75) < 1e-12
    # 三块一起：四类读者都对准长度比例。
    for reader in ("hour", "day", "month", "window6"):
        assert float(errors["111"]["hour" if reader == "hour" else reader][0]) < 1e-12
    # 关掉推前后，日、月误差从 0 回到 1。
    assert abs(float(errors["101"]["day"][0]) - 1.0) < 1e-12
    assert abs(float(errors["101"]["month"][0]) - 1.0) < 1e-12
    # 关掉支撑后，六小时窗口退回整天的 6/24。
    assert abs(float(errors["011"]["window6"][0]) - 0.75) < 1e-12


def test_exact_day_needs_circadian_for_the_hour_reader():
    geo = _pair(_at(2018, 3, 28, 0), _at(2018, 3, 29, 0))
    assert int(geo.ladder[0]) == DAY_L
    assert int(geo.hard[0]) == DAY_L
    _truth, errors = absolute_errors(geo)
    assert abs(float(errors["110"]["hour"][0]) - 1.0 / 24.0) < 1e-12
    assert float(errors["111"]["hour"][0]) < 1e-12
    assert float(errors["111"]["day"][0]) < 1e-12
    assert float(errors["111"]["month"][0]) < 1e-12
    assert float(errors["100"]["month"][0]) == 1.0


def test_cross_month_span_is_exact_only_with_support_and_filtration():
    start, end = _at(2018, 1, 30), _at(2018, 2, 4)
    geo = _pair(start, end)
    assert int(geo.hard[0]) == WEEK_L
    assert int(geo.ladder[0]) == DAY_L
    truth = truth_fractions(geo)
    assert abs(float(truth["month"][0]) - 0.6) < 1e-12
    _truth, errors = absolute_errors(geo)
    assert float(errors["111"]["month"][0]) < 1e-12
    assert float(errors["111"]["day"][0]) < 1e-12
    assert float(errors["110"]["month"][0]) < 1e-12
    assert float(errors["100"]["month"][0]) > 0.5
    # 硬周跨月，只开推前时月读者摊的是整周，不是原区间。
    assert float(errors["010"]["month"][0]) > 1e-6


def test_empty_hard_cell_uses_a_start_dirac_until_support_is_on():
    geo = _pair(_at(2018, 1, 1), _at(2018, 4, 1))
    assert int(geo.hard[0]) == 0
    assert int(geo.ladder[0]) == MONTH_L
    _truth, errors = absolute_errors(geo)
    assert float(errors["000"]["month"][0]) > 0.2
    assert float(errors["010"]["month"][0]) > 0.2
    assert float(errors["111"]["month"][0]) < 1e-12
    assert float(errors["100"]["month"][0]) < 1e-12


def test_filtration_and_circadian_do_not_change_incidence():
    start = np.array([to_unix(_at(2018, 1, 1)), to_unix(_at(2018, 1, 30))], dtype=np.int64)
    end = np.array([to_unix(_at(2018, 1, 2)), to_unix(_at(2018, 2, 4))], dtype=np.int64)
    query_left = np.array([to_unix(_at(2018, 1, 1, 12)), to_unix(_at(2018, 1, 1, 12))])
    query_right = np.array([to_unix(_at(2018, 1, 1, 18)), to_unix(_at(2018, 1, 1, 18))])
    incidence = interval_incidence(start, end, query_left, query_right)
    assert np.array_equal(incidence, np.array([1.0, 0.0]))
    again = interval_incidence(start, end, query_left, query_right)
    assert np.array_equal(incidence, again)


def test_pushforward_month_error_is_zero_on_a_batch():
    rng = np.random.default_rng(1)
    base = _at(2017, 6, 1)
    starts = []
    ends = []
    for _ in range(80):
        start = base + timedelta(days=int(rng.integers(0, 400)), hours=int(rng.integers(0, 24)))
        end = start + timedelta(hours=int(rng.choice([2, 20, 24 * 3, 24 * 15, 24 * 50, 24 * 80])))
        starts.append(to_unix(start))
        ends.append(to_unix(end))
    geo = build_geometry(starts, ends)
    truth, errors = absolute_errors(geo)
    assert float(errors["111"]["month"].max()) < 1e-9
    hour_exact = (geo.ladder == HOUR_L) | (geo.width <= 45 * 86400)
    assert float(errors["111"]["hour"][hour_exact].max()) < 1e-9
    day_exact = np.isin(geo.ladder, (HOUR_L, DAY_L))
    assert float(errors["111"]["day"][day_exact].max()) < 1e-9
    # 支撑打开但推前关闭时，小时阶梯不再写出月实体，月读者得到 0。
    lost = geo.ladder == HOUR_L
    assert lost.any()
    assert np.allclose(errors["100"]["month"][lost], truth["month"][lost])
    inside = lost & (truth["month"] > 0.99)
    assert inside.any()
    assert float(errors["100"]["month"][inside].min()) > 0.99


def test_fraction_of_a_point_and_a_partial_overlap():
    assert abs(float(fraction([10], [10], [10], [11])[0]) - 1.0) < 1e-12
    assert float(fraction([10], [10], [11], [12])[0]) == 0.0
    assert abs(float(fraction([0], [10], [2], [6])[0]) - 0.4) < 1e-12


def run_all():
    test_exact_cells_match_the_hard_aligner()
    test_hard_level_and_ladder_match_existing_aligners()
    test_afternoon_readers_follow_the_three_switches()
    test_exact_day_needs_circadian_for_the_hour_reader()
    test_cross_month_span_is_exact_only_with_support_and_filtration()
    test_empty_hard_cell_uses_a_start_dirac_until_support_is_on()
    test_filtration_and_circadian_do_not_change_incidence()
    test_pushforward_month_error_is_zero_on_a_batch()
    test_fraction_of_a_point_and_a_partial_overlap()
    print("triad tests passed")


if __name__ == "__main__":
    run_all()
