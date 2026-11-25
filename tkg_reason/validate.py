"""推理结果的时间切分检验、五折评估和城市示范。

Hits@K 的候选集是同一城市里的网格。K=10 在成百上千个网格里很严，
结果按实测填写，不把方案里的 0.85 写成已经达到。
人口、路网和重点场所不在数据里，示范建议只依据案件历史。
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from tkg_reason.infer import (
    ANY_FEATURES,
    DAY,
    FOCUS,
    REASON_DIR,
    TZ8,
    GraphData,
    _log,
    _plain,
    assemble,
    build_catalog,
    cross_region,
    fit_head,
    latest_dense_end,
    load_frame,
    neighbor_sum,
    ranking_metrics,
    score_head,
    sequence_forecast,
)
from tkg_pipeline.config import OUTPUT_DIR

EAST = {"11", "12", "13", "21", "31", "32", "33", "35", "37", "44", "46"}
CENTRAL = {"14", "34", "36", "41", "42", "43"}
REGION_NAME = {"east": "东部", "central": "中部", "west": "西部", "county": "县域"}


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    size = len(scores)
    while start < size:
        stop = start + 1
        while stop < size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    sum_pos = float(ranks[labels].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def city_ranks(scores: np.ndarray, city_index: np.ndarray) -> np.ndarray:
    ranks = np.full(len(scores), np.inf)
    for city in np.unique(city_index):
        members = np.flatnonzero(city_index == city)
        values = scores[members]
        order = np.argsort(-values, kind="mergesort")
        ordered = values[order]
        group_start = np.arange(len(ordered))
        same = np.ones(len(ordered), dtype=bool)
        same[1:] = ordered[1:] == ordered[:-1]
        change = np.where(same, 0, group_start)
        start = np.maximum.accumulate(change)
        ranks[members[order]] = start + 1
    return ranks


def retrieval_metrics(ranks: np.ndarray, event_grids: np.ndarray, grid_city_size: np.ndarray) -> dict:
    size = grid_city_size[event_grids]
    keep = np.isfinite(ranks[event_grids]) & (size >= 30)
    used = ranks[event_grids][keep]
    if len(used) == 0:
        return {"events": 0}
    return {
        "events": int(len(used)),
        "median_candidates": float(np.median(size[keep])),
        "mrr": float(np.mean(1.0 / used)),
        "hits_at_1": float(np.mean(used <= 1)),
        "hits_at_3": float(np.mean(used <= 3)),
        "hits_at_10": float(np.mean(used <= 10)),
    }


def classification_at_fraction(scores: np.ndarray, labels: np.ndarray, event_grids: np.ndarray, fraction: float) -> dict:
    labels = labels.astype(bool)
    metric = ranking_metrics(scores, event_grids, fraction)
    k = metric["flagged_grids"]
    top = np.argpartition(scores, -k)[-k:]
    predicted = np.zeros(len(scores), dtype=bool)
    predicted[top] = True
    true_positive = int(np.sum(predicted & labels))
    false_positive = int(np.sum(predicted & ~labels))
    false_negative = int(np.sum(~predicted & labels))
    true_negative = int(np.sum(~predicted & ~labels))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "accuracy": (true_positive + true_negative) / len(scores),
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "auc": roc_auc(scores, labels),
        "event_recall": metric["capture"],
        "flagged_grids": k,
        "positive_grids": int(labels.sum()),
        "events": metric["events"],
    }


def fit_snapshot_model(data: GraphData, decision: int, rng: np.random.Generator) -> dict:
    cutoffs = [decision - DAY * 30 * (index + 1) for index in range(8)]
    features, labels = assemble(data, cutoffs, 30 * DAY, data.any_features, None, rng)
    nonnegative = tuple(index for index, name in enumerate(ANY_FEATURES) if name.startswith("log_"))
    return fit_head(features, labels, nonnegative=nonnegative)


def evaluate_decision(data: GraphData, head: dict, decision: int, city_index: np.ndarray, city_sizes: np.ndarray) -> dict:
    end = min(decision + 30 * DAY - 1, int(data.unix.max()))
    features = data.any_features(decision)
    scores = score_head(features, head)
    labels, event_grids = data.labels(decision, end, None)
    baseline = np.log1p(data.counts(decision - 90 * DAY, decision - 1))
    grid_city_size = city_sizes[city_index]
    model_ranks = city_ranks(scores, city_index)
    base_ranks = city_ranks(baseline, city_index)
    return {
        "decision": datetime.fromtimestamp(decision, TZ8).date().isoformat(),
        "model_retrieval": retrieval_metrics(model_ranks, event_grids, grid_city_size),
        "baseline_retrieval": retrieval_metrics(base_ranks, event_grids, grid_city_size),
        "classification": classification_at_fraction(scores, labels, event_grids, 0.05),
        "baseline_classification": classification_at_fraction(baseline, labels, event_grids, 0.05),
        "scores": scores,
        "end": end,
    }


def qualitative(data: GraphData, decision: int, end: int) -> dict:
    history = data.counts(decision - 90 * DAY, decision - 1)
    recent = data.counts(decision - 7 * DAY, decision - 1)
    neighbor, _ = neighbor_sum(recent, data.neighbors)
    labels, event_grids = data.labels(decision, end, None)
    k = max(1, int(round(len(history) * 0.05)))
    hot = np.zeros(len(history), dtype=bool)
    hot[np.argpartition(history, -k)[-k:]] = True
    repeat_grid = recent[event_grids] > 0
    repeat_near = (recent[event_grids] + neighbor[event_grids]) > 0
    night = data.night & (data.unix >= decision) & (data.unix <= end)
    test = (data.unix >= decision) & (data.unix <= end)
    return {
        "test_events": int(test.sum()),
        "share_in_top5pct_historical_grids": float(hot[event_grids].mean()) if len(event_grids) else None,
        "share_same_grid_within_7d": float(repeat_grid.mean()) if len(event_grids) else None,
        "share_grid_or_neighbor_within_7d": float(repeat_near.mean()) if len(event_grids) else None,
        "night_share": float(night.sum() / test.sum()) if test.sum() else None,
        "positive_grid_share": float(labels.mean()),
    }


def choose_cities(data: GraphData, train_end: int) -> list[dict]:
    train = data.unix < train_end
    counts: dict[str, int] = {}
    names: dict[str, str] = {}
    provinces: dict[str, str] = {}
    for index in np.flatnonzero(train):
        code = str(data.city_code[data.grid[index]])
        counts[code] = counts.get(code, 0) + 1
        names.setdefault(code, str(data.city_name[data.grid[index]]))
        provinces.setdefault(code, code[:2])
    picked = []
    used = set()

    def best(region: str, allowed) -> None:
        candidates = [code for code, province in provinces.items() if province in allowed and code not in used]
        if not candidates:
            return
        code = max(candidates, key=lambda item: counts[item])
        used.add(code)
        picked.append({"role": region, "city_code": code, "city_name": names[code], "train_events": int(counts[code])})

    best("east", EAST)
    best("central", CENTRAL)
    best("west", set(provinces.values()) - EAST - CENTRAL)
    county_counts: dict[str, int] = {}
    county_name: dict[str, str] = {}
    county_city: dict[str, str] = {}
    for index in np.flatnonzero(train):
        name = str(data.admin_name[data.grid[index]])
        if not (name.endswith("县") or name.endswith("旗")):
            continue
        code = str(data.admin_code[data.grid[index]])
        county_counts[code] = county_counts.get(code, 0) + 1
        county_name[code] = name
        county_city[code] = str(data.city_name[data.grid[index]])
    if county_counts:
        code = max(county_counts, key=county_counts.get)
        picked.append(
            {
                "role": "county",
                "city_code": code,
                "city_name": county_name[code],
                "parent_city": county_city[code],
                "train_events": int(county_counts[code]),
            }
        )
    return picked


def city_tables(data: GraphData, scores: np.ndarray, decision: int, end: int, cities: list[dict]) -> list[dict]:
    selected = data.grid[(data.unix >= decision) & (data.unix <= end)]
    holdout = np.bincount(selected, minlength=data.n_grids) if len(selected) else np.zeros(data.n_grids, dtype=np.int64)
    history = data.counts(decision - 90 * DAY, decision - 1)
    night_count = np.bincount(
        data.grid[(data.unix >= decision - 90 * DAY) & (data.unix < decision) & data.night],
        minlength=data.n_grids,
    ).astype(np.float64)
    tables = []
    for city in cities:
        if city["role"] == "county":
            mask = data.admin_code == city["city_code"]
        else:
            mask = data.city_code == city["city_code"]
        indexes = np.flatnonzero(mask)
        if len(indexes) == 0:
            continue
        order = indexes[np.argsort(scores[indexes])[::-1]]
        rows = []
        for rank, index in enumerate(order[:12], start=1):
            total = history[index]
            rows.append(
                {
                    "rank": rank,
                    "grid_id": data.grid_ids[index],
                    "admin_name": data.admin_name[index],
                    "score": round(float(scores[index]), 4),
                    "hist_90d": int(total),
                    "night_share": round(float(night_count[index] / total), 4) if total else None,
                    "holdout_30d": int(holdout[index]),
                }
            )
        top = order[: max(1, int(round(len(indexes) * 0.05)))]
        tables.append(
            {
                **city,
                "region": REGION_NAME[city["role"]],
                "grids": int(len(indexes)),
                "holdout_events": int(holdout[indexes].sum()),
                "top5pct_event_recall": float(holdout[top].sum() / holdout[indexes].sum()) if holdout[indexes].sum() else None,
                "rows": rows,
            }
        )
    return tables


def series_errors(forecast: dict) -> dict:
    output = {}
    for code, item in forecast.items():
        actual = np.array(item["test_weeks"], dtype=np.float64)
        predicted = np.array(item["teacher_forecast"], dtype=np.float64)
        size = min(len(actual), len(predicted))
        if size == 0:
            continue
        gap = predicted[:size] - actual[:size]
        naive_gap = actual[: size - 1] - actual[1:size] if size > 1 else np.zeros(0)
        # 上一周预测：用前一周实际值预测当周，和序列错一位
        output[code] = {
            "gru_mae": item["teacher_mae"],
            "gru_rmse": float(np.sqrt(np.mean(gap**2))),
            "naive_mae": item["naive_last_week_mae"],
            "self_gru_mae": item["self_gru_mae"],
            "weeks": int(size),
            "naive_rmse": float(np.sqrt(np.mean((actual[1:size] - actual[: size - 1]) ** 2))) if size > 1 else None,
        }
        del naive_gap
    return output


def mean_metric(rows: list[dict], path: tuple[str, ...]) -> float | None:
    values = []
    for row in rows:
        item = row
        for key in path:
            item = item.get(key) if isinstance(item, dict) else None
        if isinstance(item, (int, float)):
            values.append(float(item))
    return float(np.mean(values)) if values else None


def write_report(result: dict) -> None:
    split = result["split_80_20"]
    folds = result["folds"]
    retrieval = split["model_retrieval"]
    baseline = split["baseline_retrieval"]
    classify = split["classification"]
    lines = [
        "# 犯罪热点预测与防控决策参考报告",
        "",
        f"评估截止 {result['horizon_end']}。8:2 时间切分的测试起点是 {split['decision']}，测试长度 30 天。五折采用向前扩展：每一折只用该折开始之前的案件重训。",
        "",
        "## 定量评估",
        "",
        "排序指标在同一城市的网格中计算。只统计候选网格不少于 30 个的城市，避免个位数网格把 Hits@10 抬满。",
        "",
        f"- 8:2 测试案件 {retrieval.get('events', 0)} 起，城市内候选网格中位数 {retrieval.get('median_candidates', 0) or 0:.0f}。",
        f"- 模型 MRR {_fmt(retrieval.get('mrr'))}，Hits@1 {_fmt(retrieval.get('hits_at_1'))}，Hits@3 {_fmt(retrieval.get('hits_at_3'))}，Hits@10 {_fmt(retrieval.get('hits_at_10'))}。",
        f"- 前 90 天案件数基线 MRR {_fmt(baseline.get('mrr'))}，Hits@10 {_fmt(baseline.get('hits_at_10'))}。",
        f"- 五折平均 Hits@10 {_fmt(result['cv_hits_at_10'])}，MRR {_fmt(result['cv_mrr'])}。",
        "",
        "Hits@10 的方案目标是 0.85。在数百个同城网格里把真实发案网格排进前 10，本次结果没有达到这个数。网格级高风险识别用的是前 5% 而不是前 10 名。",
        "",
        f"30 天高风险网格（前 5%）的网格级准确率 {classify['accuracy']:.3f}，精确率 {classify['precision']:.3f}，召回率 {classify['recall']:.3f}，F1 {classify['f1']:.3f}，AUC {classify['auc']:.3f}。同一阈值下，前 90 天案件数的事件级召回率 {split['baseline_classification']['event_recall']:.3f}，AUC {split['baseline_classification']['auc']:.3f}。模型在前 5% 这一档覆盖更多案件，AUC 也更高；把真实网格排进同城前 10 名时，历史案件数更靠前。准确率高，主要因为大多数网格在 30 天内没有案件。",
        "",
        "周尺度误差：",
        "",
    ]
    names = {"C0101": "盗窃", "C0104": "诈骗", "C0201": "故意伤害"}
    for code, item in result["series"].items():
        lines.append(
            f"- {names.get(code, code)}：门控循环 MAE {item['gru_mae']:.2f}、RMSE {item['gru_rmse']:.2f}；沿用上一周 MAE {item['naive_mae']:.2f}、RMSE {item['naive_rmse']:.2f}。"
        )
    link = result["cross_region"]
    lines.extend(
        [
            "",
            f"跨区县边界关联：训练期保留 {link['edges_kept_min5']} 条（次数不少于 5）。其中测试期再次出现的比例（准确率）为 {_fmt(link.get('link_precision'))}，测试期达到 5 次但训练期未保留的比例（漏检率）为 {_fmt(link.get('link_miss_rate'))}。",
            "",
            "## 和犯罪学判断的对照",
            "",
            f"测试月 {result['qualitative']['test_events']} 起案件里，{result['qualitative']['share_in_top5pct_historical_grids']:.1%} 落在历史案件最多的 5% 网格，{result['qualitative']['share_same_grid_within_7d']:.1%} 发生在此前 7 天已有案件的同一网格，{result['qualitative']['share_grid_or_neighbor_within_7d']:.1%} 发生在该网格或四邻域此前 7 天已有案件的位置。这和犯罪热点、近重复的方向一致。",
            "",
            f"测试月夜间（20 时至次日 6 时）案件占 {result['qualitative']['night_share']:.1%}。日常活动理论需要潜在犯罪人、合适目标和防控条件三项，这三项都没有观测。现有结果只能说明案件在近期已发案的网格及其邻格重复出现。",
            "",
            "没有接入人口、商业、道路和警务年鉴，不能把月份起伏解释成人口密度或政策调整。",
            "",
            "## 四个示范地区",
            "",
            "地区按训练期案件量选取：东部、中部、西部各一个地级市，另加案件最多的县或旗。建议只针对下列网格和时段，不规定警力人数，也不规划具体路口。",
            "",
        ]
    )
    for city in result["cities"]:
        title = city["city_name"]
        if city["role"] == "county":
            title = f"{city.get('parent_city', '')}{city['city_name']}"
        recall = city["top5pct_event_recall"]
        lines.append(f"### {city['region']}：{title}")
        lines.append("")
        lines.append(
            f"训练期 {city['train_events']} 起，网格 {city['grids']} 个。按全市网格前 5% 的模型分数，覆盖该市留出 30 天案件的 {recall:.1%}。"
            if recall is not None
            else f"训练期 {city['train_events']} 起。"
        )
        lines.append("")
        lines.append("| 排名 | 网格 | 区县 | 分数 | 前90天 | 夜间占比 | 留出30天 |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
        for row in city["rows"][:8]:
            night = "" if row["night_share"] is None else f"{row['night_share']:.0%}"
            lines.append(
                f"| {row['rank']} | {row['grid_id']} | {row['admin_name']} | {row['score']:.3f} | {row['hist_90d']} | {night} | {row['holdout_30d']} |"
            )
        night_rows = [row for row in city["rows"] if row["night_share"] is not None and row["night_share"] >= 0.5 and row["hist_90d"] >= 3]
        lines.append("")
        if night_rows:
            example = night_rows[0]
            lines.append(
                f"夜间占比不低于一半且前 90 天至少 3 起的网格，巡查时段放在 20:00 至次日 06:00。例如 {example['admin_name']} {example['grid_id']}，夜间占比 {example['night_share']:.0%}。"
            )
        else:
            lines.append("这份名单里没有同时满足“前 90 天至少 3 起、夜间占比不低于一半”的网格，不单列夜间巡查网格。")
        lines.append("留出 30 天的案件数写在最后一列，用来核对，不是 2019 年 8 月以后的预测。")
        lines.append("")
    lines.extend(
        [
            "## 使用边界",
            "",
            "前 5% 网格覆盖的案件比例高于只按历史件数取前 5%，AUC 也更高。若只看同城前 10 名，历史案件数的 Hits@10 更高。大多数被标出的网格在接下来 30 天没有案件。跨区边在这 30 天里没有重复出现。盗窃的周误差上，沿用上一周的 MAE 小于门控循环。示范不能代替当地警情平台的实时数据。",
            "",
        ]
    )
    path = REASON_DIR / "防控决策参考报告.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value) -> str:
    if value is None:
        return "无法计算"
    return f"{value:.3f}"


def main() -> None:
    frame = load_frame()
    catalog = build_catalog(frame)
    data = GraphData(frame, catalog)
    horizon_end = latest_dense_end(data.unix)
    usable = np.sort(data.unix[data.unix <= horizon_end])
    split = int(usable[int(len(usable) * 0.8)])
    if split + 30 * DAY > horizon_end:
        split = horizon_end - 30 * DAY
    _log(f"8:2 测试起点 {datetime.fromtimestamp(split, TZ8).date()}")
    rng = np.random.default_rng(7)
    head = fit_snapshot_model(data, split, rng)
    city_codes = data.city_code
    city_labels, city_index = np.unique(city_codes, return_inverse=True)
    city_sizes = np.bincount(city_index)
    evaluated = evaluate_decision(data, head, split, city_index, city_sizes)
    scores = evaluated.pop("scores")
    end = evaluated.pop("end")
    quantiles = [int(usable[int(len(usable) * i / 6)]) for i in range(1, 6)]
    folds = []
    for decision in quantiles:
        if decision < int(data.unix.min()) + 180 * DAY or decision + 14 * DAY > horizon_end:
            continue
        _log(f"五折起点 {datetime.fromtimestamp(decision, TZ8).date()}")
        fold_head = fit_snapshot_model(data, decision, rng)
        fold_result = evaluate_decision(data, fold_head, decision, city_index, city_sizes)
        fold_result.pop("scores")
        fold_result.pop("end")
        folds.append(fold_result)
    _log("周尺度误差与跨区关联")
    forecast = sequence_forecast(data, split)
    region = cross_region(data, split)
    cities = city_tables(data, scores, split, end, choose_cities(data, split))
    result = {
        "horizon_end": datetime.fromtimestamp(horizon_end, TZ8).date().isoformat(),
        "split_80_20": evaluated,
        "folds": folds,
        "cv_mrr": mean_metric(folds, ("model_retrieval", "mrr")),
        "cv_hits_at_10": mean_metric(folds, ("model_retrieval", "hits_at_10")),
        "cv_auc": mean_metric(folds, ("classification", "auc")),
        "qualitative": qualitative(data, split, end),
        "series": series_errors(forecast),
        "cross_region": {
            "edges_kept_min5": region["edges_kept_min5"],
            "link_precision": region["link_precision"],
            "link_miss_rate": region["link_miss_rate"],
            "link_true_positive": region["link_true_positive"],
            "link_false_positive": region["link_false_positive"],
            "link_false_negative": region["link_false_negative"],
            "top20_any_recurring_in_test": region["top20_any_recurring_in_test"],
        },
        "cities": cities,
        "hits_at_10_target": 0.85,
        "target_met": bool(evaluated["model_retrieval"].get("hits_at_10", 0) >= 0.85),
    }
    REASON_DIR.mkdir(parents=True, exist_ok=True)
    (REASON_DIR / "validation_report.json").write_text(json.dumps(_plain(result), ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(result)
    for city in cities:
        pd.DataFrame(city["rows"]).to_csv(
            REASON_DIR / f"demo_{city['role']}_{city['city_code']}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    _log(
        json.dumps(
            {
                "mrr": result["split_80_20"]["model_retrieval"].get("mrr"),
                "hits_at_10": result["split_80_20"]["model_retrieval"].get("hits_at_10"),
                "auc": result["split_80_20"]["classification"].get("auc"),
                "f1": result["split_80_20"]["classification"].get("f1"),
            },
            ensure_ascii=False,
        )
    )
    _log(f"报告已写入 {REASON_DIR / '防控决策参考报告.md'}")


if __name__ == "__main__":
    main()
