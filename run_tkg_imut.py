"""在已对齐的犯罪记录上比较硬包含与 IMUT。

用法：py -3 run_tkg_imut.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tkg_pipeline.align import certainly_successive
from tkg_pipeline.config import OUTPUT_DIR
from tkg_pipeline.imut import adapted_succession_mass, imut_align_unix

CLEANED = OUTPUT_DIR / "cleaned_crime_records.csv"
REPORT_DIR = OUTPUT_DIR / "tkg" / "imut"


def _log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    if not CLEANED.exists():
        raise SystemExit("缺少 output/cleaned_crime_records.csv")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        CLEANED,
        usecols=["time_start_unix", "time_end_unix", "time_id", "time_grain", "time_entity_level", "grid_id", "grid_code"],
        low_memory=False,
    )
    frame["time_start_unix"] = pd.to_numeric(frame["time_start_unix"], errors="coerce")
    frame["time_end_unix"] = pd.to_numeric(frame["time_end_unix"], errors="coerce")
    frame = frame.dropna(subset=["time_start_unix", "time_end_unix"]).copy()
    frame["time_id"] = frame["time_id"].fillna("").astype(str).replace({"nan": "", "None": ""})
    frame["time_grain"] = frame["time_grain"].fillna("unknown").astype(str)
    _log(f"IMUT 对齐 {len(frame)} 条区间")

    retained = np.empty(len(frame), dtype=np.float64)
    support = np.empty(len(frame), dtype=np.int16)
    agree = np.empty(len(frame), dtype=np.int8)
    rescued = np.empty(len(frame), dtype=np.int8)
    starts = frame["time_start_unix"].to_numpy()
    ends = frame["time_end_unix"].to_numpy()
    hard_ids = frame["time_id"].tolist()
    cache: dict[tuple[float, float], object] = {}

    def plan_for(start: float, end: float):
        key = (float(start), float(end))
        found = cache.get(key)
        if found is None:
            found = imut_align_unix(key[0], key[1])
            cache[key] = found
        return found

    for index, (start, end, hard_id) in enumerate(zip(starts, ends, hard_ids)):
        plan = plan_for(start, end)
        retained[index] = plan.retained
        support[index] = plan.support
        agree[index] = int(plan.top_entity_id == hard_id and hard_id != "")
        rescued[index] = int(hard_id == "" and plan.retained >= 0.2 and plan.top_entity_id != "")
        if index and index % 200_000 == 0:
            _log(f"  已对齐 {index}")

    frame["retained"] = retained
    frame["support"] = support
    frame["agree"] = agree
    frame["rescued"] = rescued
    by_grain = {}
    for grain, part in frame.groupby("time_grain", sort=False):
        by_grain[str(grain)] = {
            "records": int(len(part)),
            "mean_retained": round(float(part["retained"].mean()), 4),
            "mean_support": round(float(part["support"].mean()), 3),
            "top1_matches_hard_entity": round(float(part["agree"].mean()), 4),
            "empty_hard_entity_with_partial_mass": int(part["rescued"].sum()),
        }

    ordered = frame.dropna(subset=["grid_code"]).sort_values(["grid_code", "time_start_unix"]).reset_index(drop=True)
    ordered = ordered[ordered["grid_code"].astype(str).str.len().gt(0)].reset_index(drop=True)
    same = ordered["grid_code"].eq(ordered["grid_code"].shift(-1)).to_numpy()
    positions = np.flatnonzero(same)
    rng = np.random.default_rng(0)
    chosen = rng.choice(positions, size=min(4000, int(positions.size)), replace=False) if positions.size else []
    soft_mass = []
    hard_keep = []
    for pos in chosen:
        left = ordered.iloc[int(pos)]
        right = ordered.iloc[int(pos) + 1]
        early = plan_for(float(left.time_start_unix), float(left.time_end_unix))
        late = plan_for(float(right.time_start_unix), float(right.time_end_unix))
        soft_mass.append(adapted_succession_mass(early, late))
        hard_keep.append(
            int(
                certainly_successive(
                    int(left.time_start_unix),
                    int(left.time_end_unix),
                    int(right.time_start_unix),
                    int(right.time_end_unix),
                )
            )
        )
    soft = np.array(soft_mass, dtype=float)
    hard = np.array(hard_keep, dtype=int)
    report = {
        "algorithm": "IMUT",
        "statement": "适配支撑上的重叠比例是零位移鞅耦合；保留质量是区间宽度的非平衡熵正则。",
        "eta_hours": 720.0,
        "records": int(len(frame)),
        "mean_retained": round(float(frame["retained"].mean()), 4),
        "mean_support": round(float(frame["support"].mean()), 3),
        "top1_matches_hard_entity": round(float(frame["agree"].mean()), 4),
        "hard_entity_empty": int(frame["time_id"].eq("").sum()),
        "partial_mass_where_hard_entity_empty": int(frame["rescued"].sum()),
        "by_grain": by_grain,
        "succession_pairs": int(len(hard)),
        "hard_succession_rate": round(float(hard.mean()) if len(hard) else 0.0, 4),
        "mean_adapted_succession_mass": round(float(soft.mean()) if len(soft) else 0.0, 4),
        "mean_adapted_mass_when_hard_keeps": round(float(soft[hard == 1].mean()) if (hard == 1).any() else 0.0, 4),
        "mean_adapted_mass_when_hard_drops": round(float(soft[hard == 0].mean()) if (hard == 0).any() else 0.0, 4),
        "unique_intervals": int(len(cache)),
    }
    path = REPORT_DIR / "imut_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    sample = _sample_rows(frame, plan_for)
    sample_path = REPORT_DIR / "imut_sample.csv"
    sample.to_csv(sample_path, index=False, encoding="utf-8-sig")
    _log(json.dumps(report, ensure_ascii=False, indent=2))


def _sample_rows(frame: pd.DataFrame, plan_for) -> pd.DataFrame:
    """抽出硬对齐与 IMUT 分叉的几条记录，方便对照。"""
    picks = []
    for column, limit in (("agree", 4), ("rescued", 4)):
        hits = frame.index[frame[column].eq(1)]
        if column == "agree":
            hits = frame.index[frame["agree"].eq(0) & frame["time_id"].ne("")]
        picks.extend(list(hits[:limit]))
    rows = []
    for index in picks:
        rec = frame.loc[index]
        plan = plan_for(float(rec.time_start_unix), float(rec.time_end_unix))
        rows.append(
            {
                "time_grain": rec.time_grain,
                "time_start_unix": int(rec.time_start_unix),
                "time_end_unix": int(rec.time_end_unix),
                "hard_entity_id": rec.time_id,
                "imut_level": plan.level,
                "imut_top_entity_id": plan.top_entity_id,
                "retained": round(plan.retained, 4),
                "support": plan.support,
                "mass": ";".join(f"{entity}:{value:.4f}" for entity, value in zip(plan.entity_ids, plan.mass)),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
