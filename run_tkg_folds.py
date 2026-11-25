"""Protocol A primary cut and five expanding folds on the current cleaned extract."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from tkg_reason.infer import DAY, TZ8, GraphData, build_catalog, load_frame
from tkg_reason.validate import evaluate_decision, fit_snapshot_model, latest_dense_end

OUT = Path(r"E:\Cursor_code\phdstart\output\tkg\reason\protocol_a_refit.json")


def slim(block: dict) -> dict:
    return {
        "decision": block["decision"],
        "events": block["model_retrieval"].get("events"),
        "model": {
            "mrr": block["model_retrieval"].get("mrr"),
            "hits_at_1": block["model_retrieval"].get("hits_at_1"),
            "hits_at_3": block["model_retrieval"].get("hits_at_3"),
            "hits_at_10": block["model_retrieval"].get("hits_at_10"),
        },
        "count90": {
            "mrr": block["baseline_retrieval"].get("mrr"),
            "hits_at_1": block["baseline_retrieval"].get("hits_at_1"),
            "hits_at_3": block["baseline_retrieval"].get("hits_at_3"),
            "hits_at_10": block["baseline_retrieval"].get("hits_at_10"),
        },
        "screen_model": {k: block["classification"].get(k) for k in ("precision", "recall", "f1", "auc", "event_recall", "flagged_grids", "positive_grids")},
        "screen_count": {k: block["baseline_classification"].get(k) for k in ("precision", "recall", "f1", "auc", "event_recall")},
    }


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
    city_labels, city_index = np.unique(data.city_code, return_inverse=True)
    city_sizes = np.bincount(city_index)
    rows = []
    for decision in [split, *[int(usable[int(len(usable) * i / 6)]) for i in range(1, 6)]]:
        if decision < int(data.unix.min()) + 180 * DAY or decision + 14 * DAY > horizon_end:
            continue
        head = fit_snapshot_model(data, decision, rng)
        block = evaluate_decision(data, head, decision, city_index, city_sizes)
        block.pop("scores", None)
        block.pop("end", None)
        rows.append(slim(block))
        print(rows[-1]["decision"], rows[-1]["model"]["hits_at_10"], rows[-1]["count90"]["hits_at_10"])
    payload = {"rows_loaded": int(len(frame)), "n_grids": int(data.n_grids), "cuts": rows}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
