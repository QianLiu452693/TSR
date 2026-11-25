"""Tie-aware Protocol A audit on the extract the current loader reads.

Expected Hits@10 is the probability a cell falls inside the top 10 when
ties are broken uniformly at random. Mid-rank is not that probability.
Screening reports an arbitrary argpartition cutoff, the same random-cutoff
expectation, and the tie-inclusive list.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tkg_model.build_tkg import CLEANED
from tkg_reason.infer import (
    ANY_FEATURES,
    DAY,
    TZ8,
    GraphData,
    assemble,
    build_catalog,
    fit_head,
    load_frame,
    score_head,
)
from tkg_reason.validate import city_ranks, latest_dense_end

OUT = Path(r"E:\Cursor_code\phdstart\output\tkg\reason\tie_audit.json")
EVENT_CSV = Path(r"E:\Cursor_code\phdstart\output\tkg\reason\fold_event_ranks.csv")
SEEDS = 20


def city_groups(city_index: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(city_index, kind="mergesort")
    sorted_cities = city_index[order]
    cuts = np.flatnonzero(np.diff(sorted_cities)) + 1
    return [order[start:end] for start, end in zip(np.r_[0, cuts], np.r_[cuts, len(order)])]


def ranks_best_worst(scores: np.ndarray, groups: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    best = np.full(len(scores), np.inf)
    worst = np.full(len(scores), np.inf)
    for members in groups:
        values = scores[members]
        order = np.argsort(-values, kind="mergesort")
        ordered = values[order]
        n = len(ordered)
        start = 0
        while start < n:
            end = start + 1
            while end < n and ordered[end] == ordered[start]:
                end += 1
            best[members[order[start:end]]] = start + 1
            worst[members[order[start:end]]] = end
            start = end
    return best, worst


def inclusion_prob(scores: np.ndarray, groups: list[np.ndarray], k: int = 10) -> np.ndarray:
    prob = np.zeros(len(scores), dtype=np.float64)
    for members in groups:
        values = scores[members]
        order = np.argsort(-values, kind="mergesort")
        ordered = values[order]
        n = len(ordered)
        start = 0
        while start < n:
            end = start + 1
            while end < n and ordered[end] == ordered[start]:
                end += 1
            slots = k - start
            length = end - start
            if slots <= 0:
                p = 0.0
            elif slots >= length:
                p = 1.0
            else:
                p = slots / length
            prob[members[order[start:end]]] = p
            start = end
    return prob


def tie_diag(counts: np.ndarray, groups: list[np.ndarray], k: int = 10) -> dict:
    max_tie = 1
    cities_cross = 0
    cells_cross = 0
    values: set[int] = set()
    for members in groups:
        vals = counts[members]
        values.update(int(v) for v in np.unique(vals))
        order = np.argsort(-vals, kind="mergesort")
        ordered = vals[order]
        n = len(ordered)
        start = 0
        while start < n:
            end = start + 1
            while end < n and ordered[end] == ordered[start]:
                end += 1
            length = end - start
            if length > max_tie:
                max_tie = length
            if start < k <= end - 1 or (start < k and end > k):
                cities_cross += 1
                cells_cross += length
            start = end
    return {
        "distinct_integer_values": len(values),
        "max_tie_block": int(max_tie),
        "cities_with_tie_across_rank_10": int(cities_cross),
        "cells_inside_those_blocks": int(cells_cross),
        "cities": len(groups),
    }


def mc_hits(scores: np.ndarray, groups: list[np.ndarray], event_grids: np.ndarray, keep: np.ndarray, k: int = 10) -> dict:
    rates = []
    for seed in range(SEEDS):
        noise = np.random.default_rng(seed).random(len(scores))
        ranks = np.empty(len(scores), dtype=np.float64)
        for members in groups:
            order = np.lexsort((noise[members], -scores[members]))
            ranks[members[order]] = np.arange(1, len(members) + 1)
        rates.append(float(np.mean(ranks[event_grids][keep] <= k)))
    arr = np.asarray(rates)
    return {"seeds": SEEDS, "mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}


def screen_parts(scores: np.ndarray, fraction: float = 0.05) -> tuple[np.ndarray, np.ndarray, dict]:
    n = len(scores)
    k = max(1, int(round(n * fraction)))
    order = np.argsort(-scores, kind="mergesort")
    threshold = scores[order[k - 1]]
    above = scores > threshold
    tied = np.isclose(scores, threshold, rtol=0.0, atol=0.0)
    n_above = int(above.sum())
    n_tied = int(tied.sum())
    slots = k - n_above
    expected = np.zeros(n, dtype=np.float64)
    expected[above] = 1.0
    if n_tied:
        if slots <= 0:
            p = 0.0
        elif slots >= n_tied:
            p = 1.0
        else:
            p = slots / n_tied
        expected[tied] = p
    inclusive = above | tied
    top = np.argpartition(scores, -k)[-k:]
    arbitrary = np.zeros(n, dtype=bool)
    arbitrary[top] = True
    meta = {
        "k": k,
        "n_grids": n,
        "n_strictly_above": n_above,
        "n_tied_at_cutoff": n_tied,
        "slots_inside_tie": int(slots),
        "inclusive_length": int(inclusive.sum()),
    }
    return expected, inclusive, arbitrary, meta


def hits_from_prob(prob: np.ndarray, event_grids: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return float(np.mean(prob[event_grids][mask]))


def hits_from_rank(ranks: np.ndarray, event_grids: np.ndarray, mask: np.ndarray, k: int = 10) -> float | None:
    if not np.any(mask):
        return None
    return float(np.mean(ranks[event_grids][mask] <= k))


def capture(flag: np.ndarray, event_grids: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return float(np.mean(flag[event_grids][mask]))


def city_bootstrap(diff: np.ndarray, cities: np.ndarray, draws: int = 2000, seed: int = 0) -> dict:
    uniq, inverse = np.unique(cities, return_inverse=True)
    groups = [diff[inverse == i] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    n = len(groups)
    for draw in range(draws):
        pick = rng.integers(0, n, size=n)
        out[draw] = float(np.mean(np.concatenate([groups[i] for i in pick])))
    return {
        "mean": float(np.mean(diff)),
        "ci95": [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))],
        "draws": draws,
        "seed": seed,
        "cities": int(n),
    }


def fit_variant(data: GraphData, decision: int, rng: np.random.Generator, mode: str):
    cutoffs = [decision - DAY * 30 * (index + 1) for index in range(8)]
    features, labels = assemble(data, cutoffs, 30 * DAY, data.any_features, None, rng)
    theft_col = ANY_FEATURES.index("theft_share_90d")
    if mode == "constrained":
        nn = tuple(i for i, name in enumerate(ANY_FEATURES) if name.startswith("log_"))
        head = fit_head(features, labels, nonnegative=nn)
        return head, lambda d, fn=data.any_features: fn(d)
    if mode == "free":
        head = fit_head(features, labels, nonnegative=())
        return head, lambda d, fn=data.any_features: fn(d)
    keep = [i for i in range(features.shape[1]) if i != theft_col]
    nn = tuple(j for j, i in enumerate(keep) if ANY_FEATURES[i].startswith("log_"))
    head = fit_head(features[:, keep], labels, nonnegative=nn)
    return head, lambda d, fn=data.any_features, cols=keep: fn(d)[:, cols]


def retrieval_block(scores, groups_all, groups_eval, event_grids, masks, do_mc: bool) -> dict:
    best, worst = ranks_best_worst(scores, groups_all)
    prob = inclusion_prob(scores, groups_all, 10)
    out = {}
    for name, mask in masks.items():
        out[name] = {
            "events": int(mask.sum()),
            "optimistic": hits_from_rank(best, event_grids, mask),
            "expected_random": hits_from_prob(prob, event_grids, mask),
            "worst": hits_from_rank(worst, event_grids, mask),
        }
        if do_mc and name == "all":
            out[name]["monte_carlo"] = mc_hits(scores, groups_eval, event_grids, mask)
    return out, prob, best


def screen_block(scores, event_grids, masks) -> tuple[dict, np.ndarray]:
    expected, inclusive, arbitrary, meta = screen_parts(scores)
    block = {"cutoff": meta}
    for name, mask in masks.items():
        block[name] = {
            "events": int(mask.sum()),
            "argpartition": capture(arbitrary, event_grids, mask),
            "expected_random": capture(expected, event_grids, mask),
            "tie_inclusive": capture(inclusive, event_grids, mask),
        }
    return block, expected


def prepare(frame: pd.DataFrame):
    catalog = build_catalog(frame)
    data = GraphData(frame, catalog)
    _, city_index = np.unique(data.city_code, return_inverse=True)
    groups_all = city_groups(city_index)
    sizes = np.array([len(g) for g in groups_all])
    groups_eval = [g for g in groups_all if len(g) >= 30]
    grid_size = np.zeros(data.n_grids, dtype=np.int32)
    for g in groups_all:
        grid_size[g] = len(g)
    return data, city_index, groups_all, groups_eval, grid_size


def one_cut(data, city_index, groups_all, groups_eval, grid_size, decision, rng, label):
    end = min(decision + 30 * DAY - 1, int(data.unix.max()))
    _, event_grids = data.labels(decision, end, None)
    window = (data.unix >= decision) & (data.unix <= end)
    theft = data.types[window] == data.theft_index
    keep = grid_size[event_grids] >= 30
    masks = {"all": keep, "theft": keep & theft, "non_theft": keep & ~theft}
    screen_masks = {"all": np.ones(len(event_grids), dtype=bool), "theft": theft, "non_theft": ~theft}
    named = {}
    probs = {}
    bests = {}
    specs = {
        "constrained": True,
        "free": True,
        "no_theft": True,
    }
    for mode, do_mc in specs.items():
        head, feat = fit_variant(data, decision, rng, mode)
        scores = score_head(feat(decision), head)
        block, prob, best = retrieval_block(scores, groups_all, groups_eval, event_grids, masks, do_mc and mode == "constrained")
        screen, screen_p = screen_block(scores, event_grids, screen_masks)
        named[mode] = {"retrieval": block, "screen": screen}
        if mode in ("constrained", "free"):
            named[mode]["weights"] = {
                name: float(head["w"][i]) for i, name in enumerate(ANY_FEATURES)
            }
            named[mode]["intercept"] = float(head["w"][-1])
        if mode == "no_theft":
            kept = [name for name in ANY_FEATURES if name != "theft_share_90d"]
            named[mode]["weights"] = {
                name: float(head["w"][i]) for i, name in enumerate(kept)
            }
            named[mode]["intercept"] = float(head["w"][-1])
        if mode == "constrained":
            probs["model"] = prob
            bests["model"] = best
            named[mode]["screen_prob"] = screen_p
    for key, days in (("count7", 7), ("count30", 30), ("count90", 90)):
        raw = data.counts(decision - days * DAY, decision - 1)
        scores = np.log1p(raw)
        do_mc = key == "count90"
        block, prob, best = retrieval_block(scores, groups_all, groups_eval, event_grids, masks, do_mc)
        screen, screen_p = screen_block(scores, event_grids, screen_masks)
        named[key] = {"retrieval": block, "screen": screen}
        if key == "count90":
            eval_groups = [g for g in groups_all if len(g) >= 30]
            named[key]["tie_diag"] = tie_diag(raw, eval_groups)
            probs["count90"] = prob
            bests["count90"] = best
            named[key]["screen_prob"] = screen_p
    event_city = city_index[event_grids]
    model_screen = named["constrained"]["screen_prob"][event_grids]
    count_screen = named["count90"]["screen_prob"][event_grids]
    boot_hits = city_bootstrap(
        (probs["count90"][event_grids] - probs["model"][event_grids])[keep],
        event_city[keep],
    )
    boot_screen = city_bootstrap(
        count_screen - model_screen,
        event_city,
    )
    for item in (named["count90"], named["constrained"]):
        item.pop("screen_prob", None)
    return {
        "label": label,
        "decision": datetime.fromtimestamp(decision, TZ8).date().isoformat(),
        "test_events": int(window.sum()),
        "retrieval_events": int(keep.sum()),
        "scores": named,
        "bootstrap_expected_hits_count_minus_model": boot_hits,
        "bootstrap_expected_screen_count_minus_model": boot_screen,
    }, {
        "cut": label,
        "city": event_city,
        "theft": theft.astype(np.int8),
        "keep": keep.astype(np.int8),
        "model_expected_hit": probs["model"][event_grids],
        "count90_expected_hit": probs["count90"][event_grids],
        "model_best_rank": bests["model"][event_grids],
        "count90_best_rank": bests["count90"][event_grids],
        "model_screen_prob": model_screen,
        "count90_screen_prob": count_screen,
    }


def near_and_succession(data: GraphData, precise: np.ndarray, decision: int) -> dict:
    end = min(decision + 30 * DAY - 1, int(data.unix.max()))
    test = (data.unix >= decision) & (data.unix <= end)
    order = np.argsort(data.grid, kind="mergesort")
    # within each grid, walk precise timestamps
    same = 0
    test_precise = 0
    grids = data.grid
    unix = data.unix
    # map grid -> list of precise times
    buckets: dict[int, list[int]] = {}
    for idx in np.flatnonzero(precise):
        buckets.setdefault(int(grids[idx]), []).append(int(unix[idx]))
    for arr in buckets.values():
        arr.sort()
    pairs = 0
    for arr in buckets.values():
        left = 0
        for i, t in enumerate(arr):
            while arr[left] <= t - DAY:
                left += 1
            pairs += i - left
    for idx in np.flatnonzero(test & precise):
        test_precise += 1
        arr = buckets.get(int(grids[idx]), [])
        t = int(unix[idx])
        # any strictly earlier precise event within 7 days
        lo = t - 7 * DAY
        # binary search
        a = 0
        b = len(arr)
        while a < b:
            mid = (a + b) // 2
            if arr[mid] < lo:
                a = mid + 1
            else:
                b = mid
        hit = False
        for u in arr[a:]:
            if u >= t:
                break
            if u > t - 7 * DAY:
                hit = True
                break
        if hit:
            same += 1
    return {
        "test_events": int(test.sum()),
        "test_events_precise": test_precise,
        "precise_same_cell_near_repeat_7d": same,
        "precise_same_cell_succession_pairs_24h": pairs,
        "precise_events": int(precise.sum()),
    }


def load_with_flags() -> pd.DataFrame:
    from tkg_reason.infer import local_parts

    frame = pd.read_csv(
        CLEANED,
        usecols=[
            "crime_type_id",
            "crime_type_name",
            "city_code",
            "city_name",
            "admin_code",
            "admin_name",
            "grid_id",
            "grid_x",
            "grid_y",
            "unix_timestamp",
            "iso8601",
            "type_match_method",
            "time_precision",
        ],
    )
    frame["unix"] = pd.to_numeric(frame["unix_timestamp"], errors="coerce")
    frame = frame.dropna(subset=["unix", "grid_id", "crime_type_id", "admin_code", "city_code"]).copy()
    frame["unix"] = frame["unix"].astype(np.int64)
    for column, width in (("city_code", 6), ("admin_code", 6)):
        frame[column] = frame[column].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(width)
    frame["grid_x"] = pd.to_numeric(frame["grid_x"], errors="coerce").fillna(0).astype(np.int32)
    frame["grid_y"] = pd.to_numeric(frame["grid_y"], errors="coerce").fillna(0).astype(np.int32)
    frame["crime_type_id"] = frame["crime_type_id"].astype(str)
    frame["type_match_method"] = frame["type_match_method"].astype(str)
    frame["time_precision"] = frame["time_precision"].astype(str)
    print(f"events {len(frame)}", flush=True)
    return frame


def main() -> None:
    frame = load_with_flags()
    precise = frame["time_precision"].isin(["hour", "minute", "second"]).to_numpy()
    soft = frame["type_match_method"].isin(["tfidf", "cosine"]).to_numpy()
    align_note = "same row filter as the forecast loader, plus match method and precision"
    print("soft", int(soft.sum()), "precise", int(precise.sum()), frame["type_match_method"].value_counts().head(8).to_dict(), flush=True)
    data, city_index, groups_all, groups_eval, grid_size = prepare(frame)
    if len(precise) != len(data.unix):
        precise = np.ones(len(data.unix), dtype=bool)
        soft = np.zeros(len(data.unix), dtype=bool)
        align_note = "precision array reset after GraphData"
    horizon_end = latest_dense_end(data.unix)
    usable = np.sort(data.unix[data.unix <= horizon_end])
    split = int(usable[int(len(usable) * 0.8)])
    if split + 30 * DAY > horizon_end:
        split = horizon_end - 30 * DAY
    quantiles = [int(usable[int(len(usable) * i / 6)]) for i in range(1, 6)]
    seasonal = []
    for year, month in ((2016, 3), (2016, 6), (2017, 3), (2017, 6)):
        seasonal.append(int(datetime(year, month, 1, tzinfo=TZ8).timestamp()))
    decisions = [("primary", split)]
    for i, decision in enumerate(quantiles, start=1):
        decisions.append((f"fold{i}", decision))
    for decision in seasonal:
        decisions.append((datetime.fromtimestamp(decision, TZ8).strftime("season-%Y-%m"), decision))
    rng = np.random.default_rng(7)
    cuts = []
    rows = []
    seen = set()
    for label, decision in decisions:
        if decision in seen:
            continue
        if decision < int(data.unix.min()) + 180 * DAY or decision + 14 * DAY > horizon_end:
            continue
        seen.add(decision)
        print("cut", label, datetime.fromtimestamp(decision, TZ8).date(), flush=True)
        summary, events = one_cut(data, city_index, groups_all, groups_eval, grid_size, decision, rng, label)
        # drop large arrays already removed
        cuts.append(summary)
        part = pd.DataFrame(events)
        part.insert(0, "decision", summary["decision"])
        rows.append(part)
        c90 = summary["scores"]["count90"]["retrieval"]["all"]
        model = summary["scores"]["constrained"]["retrieval"]["all"]
        print(
            "  expected",
            round(model["expected_random"], 4),
            round(c90["expected_random"], 4),
            "optimistic",
            round(model["optimistic"], 4),
            round(c90["optimistic"], 4),
            flush=True,
        )
    clipped = (frame["grid_x"].to_numpy() < 0) | (frame["grid_x"].to_numpy() > 999) | (frame["grid_y"].to_numpy() < 0) | (frame["grid_y"].to_numpy() > 999)
    sensitivities = {"clipped_rows": int(clipped.sum()), "soft_rows": int(soft.sum()), "align": align_note, "runs": {}}
    for name, drop in (("drop_clipped", clipped), ("drop_soft", soft)):
        if int(drop.sum()) == 0:
            sensitivities["runs"][name] = {"rows_removed": 0}
            continue
        sub = frame.loc[~drop].reset_index(drop=True)
        sub_data, sub_city, sub_all, sub_eval, sub_size = prepare(sub)
        block, _ = one_cut(sub_data, sub_city, sub_all, sub_eval, sub_size, split, np.random.default_rng(7), name)
        sensitivities["runs"][name] = {
            "rows": int(len(sub)),
            "model_expected_hits": block["scores"]["constrained"]["retrieval"]["all"]["expected_random"],
            "count90_expected_hits": block["scores"]["count90"]["retrieval"]["all"]["expected_random"],
            "model_expected_screen": block["scores"]["constrained"]["screen"]["all"]["expected_random"],
            "count90_expected_screen": block["scores"]["count90"]["screen"]["all"]["expected_random"],
        }
        print(name, sensitivities["runs"][name], flush=True)
    clock = near_and_succession(data, precise, split)
    payload = {
        "rows": int(len(frame)),
        "grids": int(data.n_grids),
        "cuts": cuts,
        "sensitivities": sensitivities,
        "precise_clock": clock,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.concat(rows, ignore_index=True).to_csv(EVENT_CSV, index=False)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
