"""Primary-cut tie rules, city-clustered bootstrap, and theft split.

Runs on the cleaned table now on disk. That table is not the 724,399-row
audit, so the JSON is a separate measurement and must not be pasted over
the archived headline unless the Hits@10 pair matches.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from tkg_reason.infer import DAY, TZ8, GraphData, build_catalog, load_frame, score_head
from tkg_reason.validate import fit_snapshot_model, latest_dense_end

OUT = Path(r"E:\Cursor_code\phdstart\output\tkg\reason\primary_robust.json")


def tie_ranks(scores: np.ndarray, city_index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best = np.full(len(scores), np.inf)
    worst = np.full(len(scores), np.inf)
    mid = np.full(len(scores), np.inf)
    for city in np.unique(city_index):
        members = np.flatnonzero(city_index == city)
        values = scores[members]
        order = np.argsort(-values, kind="mergesort")
        ordered = values[order]
        n = len(ordered)
        start = np.zeros(n, dtype=np.int32)
        i = 0
        while i < n:
            j = i + 1
            while j < n and ordered[j] == ordered[i]:
                j += 1
            start[i:j] = i
            i = j
        length = np.empty(n, dtype=np.int32)
        i = 0
        while i < n:
            j = i + 1
            while j < n and ordered[j] == ordered[i]:
                j += 1
            length[i:j] = j - i
            i = j
        b = start + 1
        w = start + length
        best[members[order]] = b
        worst[members[order]] = w
        mid[members[order]] = (b + w) / 2.0
    return best, worst, mid


def hits(ranks: np.ndarray, event_grids: np.ndarray, keep: np.ndarray) -> float:
    used = ranks[event_grids][keep]
    return float(np.mean(used <= 10))


def main() -> None:
    frame = load_frame()
    catalog = build_catalog(frame)
    data = GraphData(frame, catalog)
    horizon_end = latest_dense_end(data.unix)
    usable = np.sort(data.unix[data.unix <= horizon_end])
    split = int(usable[int(len(usable) * 0.8)])
    if split + 30 * DAY > horizon_end:
        split = horizon_end - 30 * DAY
    rng = np.random.default_rng(7)
    head = fit_snapshot_model(data, split, rng)
    end = min(split + 30 * DAY - 1, int(data.unix.max()))
    features = data.any_features(split)
    scores = score_head(features, head)
    baseline = np.log1p(data.counts(split - 90 * DAY, split - 1))
    _, event_grids = data.labels(split, end, None)
    city_labels, city_index = np.unique(data.city_code, return_inverse=True)
    city_sizes = np.bincount(city_index)
    grid_city_size = city_sizes[city_index]
    keep = np.isfinite(city_index[event_grids]) & (grid_city_size[event_grids] >= 30)
    event_city = city_index[event_grids]
    model_best, model_worst, model_mid = tie_ranks(scores, city_index)
    base_best, base_worst, base_mid = tie_ranks(baseline, city_index)
    window = (data.unix >= split) & (data.unix <= end)
    event_type = data.types[window]
    theft = event_type == data.theft_index

    def pack(name, model_r, base_r, mask):
        m = keep & mask
        return {
            "events": int(m.sum()),
            "model_hits10": hits(model_r, event_grids, m) if m.any() else None,
            "count_hits10": hits(base_r, event_grids, m) if m.any() else None,
        }

    summary = {
        "rows": int(len(frame)),
        "decision": datetime.fromtimestamp(split, TZ8).date().isoformat(),
        "optimistic": {
            "all": pack("all", model_best, base_best, np.ones(len(event_grids), dtype=bool)),
            "theft": pack("theft", model_best, base_best, theft),
            "non_theft": pack("non", model_best, base_best, ~theft),
        },
        "worst": pack("all", model_worst, base_worst, np.ones(len(event_grids), dtype=bool)),
        "mid": pack("all", model_mid, base_mid, np.ones(len(event_grids), dtype=bool)),
    }
    # city-clustered bootstrap of optimistic paired difference, count minus model
    hit_c = (base_best[event_grids] <= 10) & keep
    hit_m = (model_best[event_grids] <= 10) & keep
    cities = event_city[keep]
    diff = hit_c[keep].astype(np.float64) - hit_m[keep].astype(np.float64)
    uniq = np.unique(cities)
    groups = [diff[cities == city] for city in uniq]
    boot = np.empty(2000, dtype=np.float64)
    rng_b = np.random.default_rng(0)
    for draw in range(2000):
        pick = rng_b.integers(0, len(groups), size=len(groups))
        parts = [groups[i] for i in pick]
        pooled = np.concatenate(parts) if parts else np.array([np.nan])
        boot[draw] = float(np.mean(pooled))
    summary["bootstrap_city"] = {
        "draws": 2000,
        "seed": 0,
        "mean_diff_count_minus_model": float(np.mean(diff)),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
    }
    # weights on this fit, standardized inside fit_head
    summary["weights"] = [float(x) for x in head["w"]]
    kept_rows = np.flatnonzero(keep)
    table = {
        "event_grid": event_grids[kept_rows],
        "city": event_city[kept_rows],
        "theft": theft[kept_rows].astype(np.int8),
        "model_best": model_best[event_grids][kept_rows],
        "count_best": base_best[event_grids][kept_rows],
        "model_mid": model_mid[event_grids][kept_rows],
        "count_mid": base_mid[event_grids][kept_rows],
        "model_worst": model_worst[event_grids][kept_rows],
        "count_worst": base_worst[event_grids][kept_rows],
    }
    import pandas as pd
    pd.DataFrame(table).to_csv(OUT.with_name("primary_event_ranks.csv"), index=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
