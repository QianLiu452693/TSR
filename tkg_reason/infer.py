"""犯罪 TKG 推理。

共享特征是多尺度空间计数、时段占比和 14 天霍克斯近重复。
三个任务共用这套特征，各自有逻辑回归输出头。
序列部分用残差门控循环预测周案件数，自门控循环作为更小的学生模型。
跨区域结果只表示行政区边界上的近重复和罪名结构相似，不表示嫌疑人轨迹。
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tkg_model.build_tkg import CLEANED
from tkg_model.holidays import SPRING_FESTIVAL
from tkg_pipeline.config import OUTPUT_DIR

REASON_DIR = OUTPUT_DIR / "tkg" / "reason"
TZ8 = timezone(timedelta(hours=8))
DAY = 86400
SEED = 7
HAWKES_TAU = 14 * DAY
HAWKES_WINDOW = 90 * DAY
FOCUS = ("C0101", "C0104", "C0201")

ANY_FEATURES = [
    "log_count_1d",
    "log_count_7d",
    "log_count_30d",
    "log_count_90d",
    "log_neighbor_7d",
    "log_neighbor_30d",
    "log_neighbor_90d",
    "log_county_30d",
    "night_share_90d",
    "holiday_share_90d",
    "weekend_share_90d",
    "log_hawkes_self",
    "log_hawkes_neighbor",
    "month_sin",
    "month_cos",
    "theft_share_90d",
]
TYPE_FEATURES = [
    "log_type_7d",
    "log_type_30d",
    "log_type_90d",
    "log_type_neighbor_30d",
    "log_count_30d",
    "log_neighbor_30d",
    "log_hawkes_type",
    "log_hawkes_self",
    "night_share_type_90d",
    "holiday_share_90d",
    "month_sin",
    "month_cos",
    "log_county_type_30d",
]
STUDENT_FEATURES = ["log_count_30d", "log_neighbor_30d"]
FEATURE_GROUPS = {
    "空间自身": ["log_count_1d", "log_count_7d", "log_count_30d", "log_count_90d"],
    "空间邻域": ["log_neighbor_7d", "log_neighbor_30d", "log_neighbor_90d", "log_county_30d"],
    "时间周期": ["night_share_90d", "holiday_share_90d", "weekend_share_90d", "month_sin", "month_cos"],
    "动态图": ["log_hawkes_self", "log_hawkes_neighbor"],
    "类型结构": ["theft_share_90d"],
}


def _log(message: str) -> None:
    print(message, flush=True)


def _epoch_day(day: date) -> int:
    return (day - date(1970, 1, 1)).days


def _holiday_days() -> np.ndarray:
    days = []
    for year in range(2000, 2020):
        days.append(date(year, 1, 1))
        days.append(date(year, 5, 1))
        days.append(date(year, 10, 1))
        spring = SPRING_FESTIVAL.get(year)
        if spring:
            days.append(spring)
    return np.array([_epoch_day(day) for day in days], dtype=np.int32)


HOLIDAY_DAYS = _holiday_days()


def local_parts(unix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = np.asarray(unix, dtype=np.int64) + 8 * 3600
    days = np.floor_divide(local, DAY).astype(np.int32)
    hour = np.floor_divide(np.remainder(local, DAY), 3600).astype(np.int16)
    weekday = np.remainder(days + 3, 7).astype(np.int8)
    return days, hour, weekday


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _nll(Xb: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float) -> float:
    eta = np.clip(Xb @ w, -20, 20)
    penalty = 0.5 * l2 * float(np.dot(w[:-1], w[:-1])) / len(y)
    return float(np.mean(np.logaddexp(0, eta) - y * eta) + penalty)


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 2.0,
    max_iter: int = 12,
    nonnegative: tuple[int, ...] = (),
) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    width = Xb.shape[1]
    reg = np.full(width, l2)
    reg[-1] = 0.0
    w = np.zeros(width)
    eye = np.eye(width)
    for _ in range(max_iter):
        eta = np.clip(Xb @ w, -20, 20)
        p = _sigmoid(eta)
        var = np.maximum(p * (1.0 - p), 1e-5)
        grad = (Xb.T @ (p - y)) / len(y) + (reg / len(y)) * w
        Xw = Xb * np.sqrt(var)[:, None]
        hessian = (Xw.T @ Xw) / len(y) + eye * (reg / len(y))
        delta = np.linalg.solve(hessian, grad)
        current = _nll(Xb, y, w, l2)
        step = 1.0
        accepted = w
        for _trial in range(6):
            trial = w - step * delta
            if _nll(Xb, y, trial, l2) <= current + 1e-9:
                accepted = trial
                break
            step *= 0.5
        if step == 1.0 and np.max(np.abs(delta)) < 1e-5:
            w = accepted
            if nonnegative:
                w[list(nonnegative)] = np.maximum(w[list(nonnegative)], 0.0)
            break
        w = accepted
        if nonnegative:
            w[list(nonnegative)] = np.maximum(w[list(nonnegative)], 0.0)
    return w


def predict_logistic(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    return _sigmoid(Xb @ w)


def standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (train - mu) / sd, (other - mu) / sd, mu, sd


def apply_standard(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def train_sequence(x: np.ndarray, y: np.ndarray, kind: str = "gru", epochs: int = 50, hidden: int = 8, lr: float = 0.03) -> dict:
    """门控循环。kind='self' 时更新门和重置门只看当前输入。"""
    rng = np.random.default_rng(SEED)
    steps, width = x.shape
    scale = 1.0 / np.sqrt(width + hidden)

    def matrix(rows: int, cols: int) -> np.ndarray:
        return rng.normal(0.0, scale, size=(rows, cols))

    Wz, Wr, Wh = matrix(hidden, width), matrix(hidden, width), matrix(hidden, width)
    bz, br, bh = np.zeros(hidden), np.zeros(hidden), np.zeros(hidden)
    Uz = np.zeros((hidden, hidden)) if kind == "self" else matrix(hidden, hidden)
    Ur = np.zeros((hidden, hidden)) if kind == "self" else matrix(hidden, hidden)
    Uh = matrix(hidden, hidden)
    wo = rng.normal(0.0, 0.1, size=hidden)
    bo = 0.0
    moment = {name: 0.0 for name in ("Wz", "Wr", "Wh", "bz", "br", "bh", "Uz", "Ur", "Uh", "wo", "bo")}
    losses = []
    for epoch in range(epochs):
        h = np.zeros(hidden)
        cache = []
        preds = np.zeros(steps)
        for t in range(steps):
            xt = x[t]
            z_pre = Wz @ xt + Uz @ h + bz
            r_pre = Wr @ xt + Ur @ h + br
            z = _sigmoid(z_pre)
            r = _sigmoid(r_pre)
            reset_h = r * h
            h_pre = Wh @ xt + Uh @ reset_h + bh
            hhat = np.tanh(h_pre)
            h_new = (1.0 - z) * h + z * hhat
            yhat = float(wo @ h_new + bo)
            preds[t] = yhat
            cache.append((xt, h, z, r, hhat, reset_h, h_new))
            h = h_new
        error = preds - y
        losses.append(float(np.mean(error**2)))
        dWz = np.zeros_like(Wz)
        dWr = np.zeros_like(Wr)
        dWh = np.zeros_like(Wh)
        dUz = np.zeros_like(Uz)
        dUr = np.zeros_like(Ur)
        dUh = np.zeros_like(Uh)
        dbz = np.zeros_like(bz)
        dbr = np.zeros_like(br)
        dbh = np.zeros_like(bh)
        dwo = np.zeros_like(wo)
        dbo = 0.0
        dh_next = np.zeros(hidden)
        for t in range(steps - 1, -1, -1):
            xt, h_prev, z, r, hhat, reset_h, h_new = cache[t]
            dy = 2.0 * error[t] / steps
            dwo += dy * h_new
            dbo += dy
            dh = dy * wo + dh_next
            dhhat = dh * z
            dz = dh * (hhat - h_prev)
            dh_prev = dh * (1.0 - z)
            dpre = dhhat * (1.0 - hhat**2)
            dWh += np.outer(dpre, xt)
            dUh += np.outer(dpre, reset_h)
            dbh += dpre
            dreset = Uh.T @ dpre
            dr = dreset * h_prev
            dh_prev += dreset * r
            dz_pre = dz * z * (1.0 - z)
            dr_pre = dr * r * (1.0 - r)
            dWz += np.outer(dz_pre, xt)
            dWr += np.outer(dr_pre, xt)
            dbz += dz_pre
            dbr += dr_pre
            dUz += np.outer(dz_pre, h_prev)
            dUr += np.outer(dr_pre, h_prev)
            dh_prev += Uz.T @ dz_pre
            dh_prev += Ur.T @ dr_pre
            dh_next = dh_prev
        grads = {
            "Wz": dWz,
            "Wr": dWr,
            "Wh": dWh,
            "bz": dbz,
            "br": dbr,
            "bh": dbh,
            "Uz": dUz,
            "Ur": dUr,
            "Uh": dUh,
            "wo": dwo,
            "bo": np.array(dbo),
        }
        total = np.sqrt(sum(float(np.sum(np.square(value))) for value in grads.values()))
        if total > 1.0:
            for key in grads:
                grads[key] = grads[key] * (1.0 / total)
        params = locals()
        for key, grad in grads.items():
            moment[key] = 0.9 * moment[key] + grad
            if key == "bo":
                bo -= lr * float(moment[key])
            else:
                params[key][...] -= lr * moment[key]
        if kind == "self":
            Uz[:] = 0.0
            Ur[:] = 0.0
    param_count = sum(value.size for value in (Wz, Wr, Wh, bz, br, bh, Uh, wo)) + 1
    if kind != "self":
        param_count += Uz.size + Ur.size
    return {
        "kind": kind,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "params": int(param_count),
        "state": {
            "Wz": Wz,
            "Wr": Wr,
            "Wh": Wh,
            "bz": bz,
            "br": br,
            "bh": bh,
            "Uz": Uz,
            "Ur": Ur,
            "Uh": Uh,
            "wo": wo,
            "bo": bo,
        },
    }


def run_sequence(state: dict, x: np.ndarray, h: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    hidden = state["wo"].shape[0]
    if h is None:
        h = np.zeros(hidden)
    preds = np.zeros(len(x))
    for t in range(len(x)):
        xt = x[t]
        z = _sigmoid(state["Wz"] @ xt + state["Uz"] @ h + state["bz"])
        r = _sigmoid(state["Wr"] @ xt + state["Ur"] @ h + state["br"])
        hhat = np.tanh(state["Wh"] @ xt + state["Uh"] @ (r * h) + state["bh"])
        h = (1.0 - z) * h + z * hhat
        preds[t] = float(state["wo"] @ h + state["bo"])
    return preds, h


def _self_check() -> None:
    moment = datetime(2016, 7, 21, 15, 0, tzinfo=TZ8)
    days, hour, weekday = local_parts(np.array([int(moment.timestamp())]))
    if int(hour[0]) != 15 or int(weekday[0]) != 3:
        raise RuntimeError("时区或星期换算错误")
    spring = datetime(2016, 2, 8, 12, 0, tzinfo=TZ8)
    spring_day, _, _ = local_parts(np.array([int(spring.timestamp())]))
    if int(spring_day[0]) not in set(HOLIDAY_DAYS.tolist()):
        raise RuntimeError("春节日期没有进入节假日集合")
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(500, 3))
    y = (X[:, 0] + 0.8 * X[:, 1] > 0).astype(np.float64)
    weights = fit_logistic(X, y, l2=0.1)
    if weights[0] <= 0 or weights[1] <= 0:
        raise RuntimeError("逻辑回归没有学到正确方向")
    t = np.arange(140, dtype=np.float64)
    signal = np.sin(2 * np.pi * t / 12.0)
    seq_x = np.column_stack([signal, np.sin(2 * np.pi * t / 12.0), np.cos(2 * np.pi * t / 12.0)])
    trained = train_sequence(seq_x[:-1], signal[1:], epochs=30, hidden=6, lr=0.05)
    if trained["loss_end"] > trained["loss_start"] * 0.7:
        raise RuntimeError("门控循环没有学到正弦序列")


def load_frame() -> pd.DataFrame:
    _log("读取标准化犯罪记录")
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
    sample = frame.sample(n=min(300, len(frame)), random_state=SEED)
    days, hour, weekday = local_parts(sample["unix"].to_numpy())
    iso_hour = sample["iso8601"].astype(str).str.slice(11, 13).astype(int).to_numpy()
    iso_day = pd.to_datetime(sample["iso8601"].astype(str).str.slice(0, 10)).dt.dayofweek.to_numpy()
    parsed_day, _, _ = local_parts(sample["unix"].to_numpy())
    iso_epoch = (pd.to_datetime(sample["iso8601"].astype(str).str.slice(0, 10)) - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    if np.mean(hour == iso_hour) < 0.99 or np.mean(parsed_day == iso_epoch) < 0.99 or np.mean(weekday == iso_day) < 0.99:
        raise RuntimeError("unix 时间与 ISO 时间不一致")
    _log(f"事件 {len(frame)} 条")
    return frame


def build_catalog(frame: pd.DataFrame) -> dict:
    catalog = (
        frame.groupby("grid_id", sort=False)
        .agg(
            city_code=("city_code", "first"),
            city_name=("city_name", "first"),
            admin_code=("admin_code", "first"),
            admin_name=("admin_name", "first"),
            grid_x=("grid_x", "first"),
            grid_y=("grid_y", "first"),
        )
        .reset_index()
    )
    lookup = {grid_id: index for index, grid_id in enumerate(catalog["grid_id"])}
    grid_index = np.fromiter((lookup[grid_id] for grid_id in frame["grid_id"]), dtype=np.int32, count=len(frame))
    n_grids = len(catalog)
    city = catalog["city_code"].to_numpy()
    gx = catalog["grid_x"].to_numpy(dtype=np.int32)
    gy = catalog["grid_y"].to_numpy(dtype=np.int32)
    key_to_index = {(city[i], int(gx[i]), int(gy[i])): i for i in range(n_grids)}
    neighbors = np.full((n_grids, 4), -1, dtype=np.int32)
    for index in range(n_grids):
        for direction, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
            neighbors[index, direction] = key_to_index.get((city[index], int(gx[index]) + dx, int(gy[index]) + dy), -1)
    type_names = frame.groupby("crime_type_id")["crime_type_name"].first().to_dict()
    _log(f"网格 {n_grids} 个，其中有相邻网格的 {(neighbors >= 0).any(axis=1).sum()} 个")
    return {
        "catalog": catalog,
        "grid_index": grid_index,
        "neighbors": neighbors,
        "type_names": {str(key): str(value) for key, value in type_names.items()},
        "n_grids": n_grids,
    }


def neighbor_sum(values: np.ndarray, neighbors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(len(values), dtype=np.float64)
    count = np.zeros(len(values), dtype=np.float64)
    for direction in range(neighbors.shape[1]):
        other = neighbors[:, direction]
        valid = other >= 0
        total[valid] += values[other[valid]]
        count[valid] += 1.0
    return total, count


def county_other_mean(values: np.ndarray, county_index: np.ndarray, n_counties: int) -> np.ndarray:
    sums = np.bincount(county_index, weights=values, minlength=n_counties)
    counts = np.bincount(county_index, minlength=n_counties).astype(np.float64)
    other = sums[county_index] - values
    denom = np.maximum(counts[county_index] - 1.0, 1.0)
    return other / denom


class GraphData:
    def __init__(self, frame: pd.DataFrame, catalog: dict):
        self.unix = frame["unix"].to_numpy(dtype=np.int64)
        self.grid = catalog["grid_index"]
        self.neighbors = catalog["neighbors"]
        self.n_grids = catalog["n_grids"]
        self.type_codes = sorted(frame["crime_type_id"].unique())
        self.type_to_i = {code: index for index, code in enumerate(self.type_codes)}
        self.types = np.fromiter((self.type_to_i[code] for code in frame["crime_type_id"]), dtype=np.int16, count=len(frame))
        admin_codes = catalog["catalog"]["admin_code"].astype(str).to_numpy()
        counties = sorted(set(admin_codes))
        self.county_codes = counties
        county_lookup = {code: index for index, code in enumerate(counties)}
        self.county_index = np.fromiter((county_lookup[code] for code in admin_codes), dtype=np.int32, count=len(admin_codes))
        self.n_counties = len(counties)
        self.admin_name = catalog["catalog"]["admin_name"].astype(str).to_numpy()
        self.city_name = catalog["catalog"]["city_name"].astype(str).to_numpy()
        self.city_code = catalog["catalog"]["city_code"].astype(str).to_numpy()
        self.grid_ids = catalog["catalog"]["grid_id"].astype(str).to_numpy()
        self.admin_code = admin_codes
        self.type_names = catalog["type_names"]
        days, hour, weekday = local_parts(self.unix)
        self.days = days
        self.night = (hour >= 20) | (hour < 6)
        self.weekend = weekday >= 5
        self.holiday = np.isin(days, HOLIDAY_DAYS)
        self.theft_index = self.type_to_i["C0101"]
        order = np.argsort(self.unix, kind="mergesort")
        self.order = order
        self.max_unix = int(self.unix.max())
        self.min_unix = int(self.unix.min())

    def _mask_range(self, start: int, end: int, type_index: int | None = None) -> np.ndarray:
        mask = (self.unix >= start) & (self.unix <= end)
        if type_index is not None:
            mask &= self.types == type_index
        return mask

    def counts(self, start: int, end: int, type_index: int | None = None) -> np.ndarray:
        mask = self._mask_range(start, end, type_index)
        if not np.any(mask):
            return np.zeros(self.n_grids, dtype=np.float64)
        return np.bincount(self.grid[mask], minlength=self.n_grids).astype(np.float64)

    def hawkes(self, decision: int, type_index: int | None = None) -> np.ndarray:
        start = decision - HAWKES_WINDOW
        mask = self._mask_range(start, decision - 1, type_index)
        if not np.any(mask):
            return np.zeros(self.n_grids, dtype=np.float64)
        weights = np.exp(-(decision - self.unix[mask]) / HAWKES_TAU)
        return np.bincount(self.grid[mask], weights=weights, minlength=self.n_grids).astype(np.float64)

    def share(self, decision: int, flag: np.ndarray, type_index: int | None = None) -> np.ndarray:
        start = decision - 90 * DAY
        base = self._mask_range(start, decision - 1, type_index)
        total = np.bincount(self.grid[base], minlength=self.n_grids).astype(np.float64) if np.any(base) else np.zeros(self.n_grids)
        selected = base & flag
        part = np.bincount(self.grid[selected], minlength=self.n_grids).astype(np.float64) if np.any(selected) else np.zeros(self.n_grids)
        global_share = float(part.sum() / total.sum()) if total.sum() else 0.0
        out = np.full(self.n_grids, global_share, dtype=np.float64)
        positive = total > 0
        out[positive] = part[positive] / total[positive]
        return out

    def any_features(self, decision: int) -> np.ndarray:
        c1 = self.counts(decision - DAY, decision - 1)
        c7 = self.counts(decision - 7 * DAY, decision - 1)
        c30 = self.counts(decision - 30 * DAY, decision - 1)
        c90 = self.counts(decision - 90 * DAY, decision - 1)
        n7, _ = neighbor_sum(c7, self.neighbors)
        n30, _ = neighbor_sum(c30, self.neighbors)
        n90, _ = neighbor_sum(c90, self.neighbors)
        county30 = county_other_mean(c30, self.county_index, self.n_counties)
        hawkes = self.hawkes(decision)
        hawkes_n, _ = neighbor_sum(hawkes, self.neighbors)
        theft90 = self.counts(decision - 90 * DAY, decision - 1, self.theft_index)
        theft_share = np.divide(theft90, c90, out=np.zeros(self.n_grids), where=c90 > 0)
        moment = datetime.fromtimestamp(decision, TZ8)
        angle = 2 * np.pi * (moment.month / 12.0)
        columns = [
            np.log1p(c1),
            np.log1p(c7),
            np.log1p(c30),
            np.log1p(c90),
            np.log1p(n7),
            np.log1p(n30),
            np.log1p(n90),
            np.log1p(county30),
            self.share(decision, self.night),
            self.share(decision, self.holiday),
            self.share(decision, self.weekend),
            np.log1p(hawkes),
            np.log1p(hawkes_n),
            np.full(self.n_grids, np.sin(angle)),
            np.full(self.n_grids, np.cos(angle)),
            theft_share,
        ]
        return np.column_stack(columns)

    def type_features(self, decision: int, type_index: int) -> np.ndarray:
        c30 = self.counts(decision - 30 * DAY, decision - 1)
        n30, _ = neighbor_sum(c30, self.neighbors)
        t7 = self.counts(decision - 7 * DAY, decision - 1, type_index)
        t30 = self.counts(decision - 30 * DAY, decision - 1, type_index)
        t90 = self.counts(decision - 90 * DAY, decision - 1, type_index)
        tn30, _ = neighbor_sum(t30, self.neighbors)
        hawkes_all = self.hawkes(decision)
        hawkes_type = self.hawkes(decision, type_index)
        county_type = county_other_mean(t30, self.county_index, self.n_counties)
        moment = datetime.fromtimestamp(decision, TZ8)
        angle = 2 * np.pi * (moment.month / 12.0)
        columns = [
            np.log1p(t7),
            np.log1p(t30),
            np.log1p(t90),
            np.log1p(tn30),
            np.log1p(c30),
            np.log1p(n30),
            np.log1p(hawkes_type),
            np.log1p(hawkes_all),
            self.share(decision, self.night, type_index),
            self.share(decision, self.holiday),
            np.full(self.n_grids, np.sin(angle)),
            np.full(self.n_grids, np.cos(angle)),
            np.log1p(county_type),
        ]
        return np.column_stack(columns)

    def labels(self, start: int, end: int, type_index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        mask = self._mask_range(start, end, type_index)
        if not np.any(mask):
            return np.zeros(self.n_grids, dtype=np.float64), np.zeros(0, dtype=np.int32)
        event_grids = self.grid[mask]
        present = np.zeros(self.n_grids, dtype=np.float64)
        present[np.unique(event_grids)] = 1.0
        return present, event_grids


def latest_dense_end(unix: np.ndarray, minimum: int = 1000) -> int:
    """最后一段仍有足够案件的 30 天。2019 年下半年文书变少，不能把收录中断当成预测窗口。"""
    end = int(unix.max())
    floor = int(unix.min()) + 180 * DAY
    while end > floor:
        start = end - 30 * DAY
        if int(np.count_nonzero((unix >= start) & (unix <= end))) >= minimum:
            return end
        end -= 7 * DAY
    return int(unix.max())


def decision_times(unix: np.ndarray) -> dict:
    horizon_end = latest_dense_end(unix)
    test_origin = horizon_end - 90 * DAY
    train_last = test_origin - 30 * DAY
    cutoffs = [train_last - index * 30 * DAY for index in range(18)]
    cutoffs = [item for item in cutoffs if item > horizon_end - 8 * 365 * DAY]
    windows = [test_origin, test_origin + 30 * DAY, test_origin + 60 * DAY]
    return {"cutoffs": cutoffs, "windows": windows, "test_origin": test_origin, "horizon_end": horizon_end}


def sample_rows(y: np.ndarray, hist: np.ndarray, rng: np.random.Generator, max_pos: int = 6000) -> np.ndarray:
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero((y == 0) & (hist > 0))
    if len(positive) > max_pos:
        positive = rng.choice(positive, size=max_pos, replace=False)
    n_neg = min(len(negative), max(len(positive) * 2, 1))
    if len(positive) == 0 or n_neg == 0:
        return np.zeros(0, dtype=np.int32)
    negative = rng.choice(negative, size=n_neg, replace=False)
    return np.concatenate([positive, negative])


def assemble(data: GraphData, cutoffs: list[int], horizon: int, feature_fn, type_index: int | None, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for decision in cutoffs:
        features = feature_fn(decision)
        labels, _ = data.labels(decision, decision + horizon - 1, type_index)
        hist = np.expm1(features[:, 2] if features.shape[1] == len(ANY_FEATURES) else features[:, 4])
        chosen = sample_rows(labels, hist, rng)
        if len(chosen) == 0:
            continue
        xs.append(features[chosen])
        ys.append(labels[chosen])
    if not xs:
        raise RuntimeError("训练样本为空")
    return np.vstack(xs), np.concatenate(ys)


def fit_head(X: np.ndarray, y: np.ndarray, nonnegative: tuple[int, ...] = ()) -> dict:
    model_x, _, mu, sd = standardize(X, X[:1])
    weights = fit_logistic(model_x, y.astype(np.float64), l2=2.0, nonnegative=nonnegative)
    return {"w": weights, "mu": mu, "sd": sd, "n": int(len(y)), "positives": int(y.sum())}


def score_head(features: np.ndarray, head: dict) -> np.ndarray:
    return predict_logistic(apply_standard(features, head["mu"], head["sd"]), head["w"])


def ranking_metrics(scores: np.ndarray, event_grids: np.ndarray, fraction: float) -> dict:
    n_grids = len(scores)
    k = max(1, int(round(n_grids * fraction)))
    top = np.argpartition(scores, -k)[-k:]
    flagged = np.zeros(n_grids, dtype=bool)
    flagged[top] = True
    n_events = int(len(event_grids))
    hits = int(flagged[event_grids].sum()) if n_events else 0
    grids_hit = int(np.intersect1d(top, event_grids).size) if n_events else 0
    capture = hits / n_events if n_events else 0.0
    return {
        "flagged_grids": k,
        "events": n_events,
        "captured_events": hits,
        "capture": capture,
        "precision": grids_hit / k,
        "pai": capture / fraction,
    }


def coefficient_groups(weights: np.ndarray, names: list[str]) -> dict:
    body = np.abs(weights[:-1])
    total = float(body.sum()) or 1.0
    shares = {}
    for group, columns in FEATURE_GROUPS.items():
        indexes = [names.index(column) for column in columns if column in names]
        shares[group] = float(body[indexes].sum() / total) if indexes else 0.0
    named = {name: float(weights[index]) for index, name in enumerate(names)}
    return {"abs_share": shares, "coefficients": named}


def high_fraction(horizon_days: int) -> float:
    if horizon_days <= 1:
        return 0.01
    if horizon_days <= 7:
        return 0.02
    return 0.05


def train_and_evaluate(data: GraphData) -> dict:
    rng = np.random.default_rng(SEED)
    times = decision_times(data.unix)
    _log(
        f"训练切分点 {len(times['cutoffs'])} 个，测试起点 {datetime.fromtimestamp(times['test_origin'], TZ8).date()}，"
        f"评价截止 {datetime.fromtimestamp(times['horizon_end'], TZ8).date()}"
    )
    heads = {}
    specs = [("any", 1, data.any_features, None, ANY_FEATURES)]
    specs.append(("any", 7, data.any_features, None, ANY_FEATURES))
    specs.append(("any", 30, data.any_features, None, ANY_FEATURES))
    for code in FOCUS:
        type_index = data.type_to_i[code]
        specs.append((code, 7, lambda decision, idx=type_index: data.type_features(decision, idx), type_index, TYPE_FEATURES))
        specs.append((code, 30, lambda decision, idx=type_index: data.type_features(decision, idx), type_index, TYPE_FEATURES))
    metrics = {}
    for name, horizon, feature_fn, type_index, feature_names in specs:
        _log(f"训练 {name} / {horizon} 天")
        X, y = assemble(data, times["cutoffs"], horizon * DAY, feature_fn, type_index, rng)
        nonnegative = tuple(index for index, column in enumerate(feature_names) if column.startswith("log_"))
        head = fit_head(X, y, nonnegative=nonnegative)
        heads[(name, horizon)] = {"head": head, "features": feature_fn, "names": feature_names}
        window_metrics = []
        for decision in times["windows"]:
            scores = score_head(feature_fn(decision), head)
            _, events = data.labels(decision, min(decision + horizon * DAY - 1, times["horizon_end"]), type_index)
            fraction = high_fraction(horizon)
            model_metric = ranking_metrics(scores, events, fraction)
            baseline_count = data.counts(decision - 30 * DAY, decision - 1, type_index)
            baseline_metric = ranking_metrics(baseline_count, events, fraction)
            window_metrics.append(
                {
                    "decision": datetime.fromtimestamp(decision, TZ8).isoformat(),
                    "model": model_metric,
                    "baseline_30d": baseline_metric,
                }
            )
        usable = [item for item in window_metrics if item["model"]["events"] > 0]
        if not usable:
            usable = window_metrics
        metrics[f"{name}_{horizon}d"] = {
            "train_rows": head["n"],
            "train_positives": head["positives"],
            "parameters": int(len(head["w"])),
            "high_tier_fraction": high_fraction(horizon),
            "windows": window_metrics,
            "capture_mean": float(np.mean([item["model"]["capture"] for item in usable])),
            "baseline_capture_mean": float(np.mean([item["baseline_30d"]["capture"] for item in usable])),
            "precision_mean": float(np.mean([item["model"]["precision"] for item in usable])),
            "baseline_precision_mean": float(np.mean([item["baseline_30d"]["precision"] for item in usable])),
            "pai_mean": float(np.mean([item["model"]["pai"] for item in usable])),
            "coefficients": coefficient_groups(head["w"], feature_names) if name == "any" and horizon == 30 else None,
        }
    student = distill_student(data, heads[("any", 30)]["head"], times)
    metrics["student_30d"] = student
    return {"heads": heads, "metrics": metrics, "times": times}


def distill_student(data: GraphData, teacher: dict, times: dict) -> dict:
    _log("蒸馏 30 天热点学生模型")
    rng = np.random.default_rng(SEED)
    columns = [ANY_FEATURES.index(name) for name in STUDENT_FEATURES]
    hard_x = []
    hard_y = []
    student_blocks = []
    logit_blocks = []
    for decision in times["cutoffs"]:
        full = data.any_features(decision)
        labels, _ = data.labels(decision, decision + 30 * DAY - 1, None)
        hist = np.expm1(full[:, ANY_FEATURES.index("log_count_30d")])
        chosen = sample_rows(labels, hist, rng)
        if len(chosen):
            hard_x.append(full[np.ix_(chosen, columns)])
            hard_y.append(labels[chosen])
        take = rng.choice(data.n_grids, size=min(8000, data.n_grids), replace=False)
        teacher_scores = score_head(full[take], teacher)
        logit_blocks.append(
            np.log(np.clip(teacher_scores, 1e-4, 1 - 1e-4) / np.clip(1 - teacher_scores, 1e-4, 1 - 1e-4))
        )
        student_blocks.append(full[np.ix_(take, columns)])
    student_x = np.vstack(student_blocks)
    logits = np.concatenate(logit_blocks)
    train_x, _, mu, sd = standardize(student_x, student_x[:1])
    design = np.concatenate([train_x, np.ones((len(train_x), 1))], axis=1)
    weights, *_ = np.linalg.lstsq(design, logits, rcond=None)
    hard_head = fit_head(np.vstack(hard_x), np.concatenate(hard_y), nonnegative=(0, 1))
    window_metrics = []
    hard_metrics = []
    teacher_captures = []
    for decision in times["windows"]:
        full = data.any_features(decision)
        student_now = apply_standard(full[:, [ANY_FEATURES.index(name) for name in STUDENT_FEATURES]], mu, sd)
        scores = student_now @ weights[:-1] + weights[-1]
        hard_scores = score_head(full[:, columns], hard_head)
        _, events = data.labels(decision, min(decision + 30 * DAY - 1, times["horizon_end"]), None)
        window_metrics.append(ranking_metrics(scores, events, 0.05))
        hard_metrics.append(ranking_metrics(hard_scores, events, 0.05))
        teacher_captures.append(ranking_metrics(score_head(full, teacher), events, 0.05)["capture"])
    capture = float(np.mean([item["capture"] for item in window_metrics]))
    hard_capture = float(np.mean([item["capture"] for item in hard_metrics]))
    teacher_capture = float(np.mean(teacher_captures))
    relative_drop = (teacher_capture - hard_capture) / teacher_capture if teacher_capture else None
    return {
        "parameters": int(len(hard_head["w"])),
        "teacher_parameters": int(len(teacher["w"])),
        "compression_ratio": float(len(teacher["w"]) / len(hard_head["w"])),
        "capture_mean": hard_capture,
        "distilled_logit_capture_mean": capture,
        "teacher_capture_mean": teacher_capture,
        "relative_capture_drop": None if relative_drop is None else float(relative_drop),
        "precision_mean": float(np.mean([item["precision"] for item in hard_metrics])),
        "windows": hard_metrics,
        "method": "轻量模型只用近 30 天自身案件和四邻域案件，直接学习 30 天是否发案。另记录了用教师 logit 蒸馏的命中率。",
    }


def mantle_haenszel(exposed: np.ndarray, outcome: np.ndarray, strata: np.ndarray) -> dict:
    numerator = 0.0
    denominator = 0.0
    crude_a = crude_b = crude_c = crude_d = 0
    for stratum in np.unique(strata):
        mask = strata == stratum
        exp = exposed[mask]
        out = outcome[mask]
        a = int(np.sum(exp & out))
        b = int(np.sum(exp & ~out))
        c = int(np.sum(~exp & out))
        d = int(np.sum(~exp & ~out))
        n = a + b + c + d
        crude_a += a
        crude_b += b
        crude_c += c
        crude_d += d
        if n == 0:
            continue
        numerator += a * (c + d) / n
        denominator += c * (a + b) / n
    crude_rate_exp = crude_a / max(crude_a + crude_b, 1)
    crude_rate_ref = crude_c / max(crude_c + crude_d, 1)
    return {
        "crude_risk_ratio": crude_rate_exp / crude_rate_ref if crude_rate_ref else None,
        "adjusted_risk_ratio": numerator / denominator if denominator else None,
        "exposed_events": crude_a,
        "reference_events": crude_c,
        "exposed": crude_a + crude_b,
        "reference": crude_c + crude_d,
    }


def causal_report(data: GraphData, decision: int) -> dict:
    _log("计算夜间盗窃的分层风险比")
    exposure_end = decision - 90 * DAY
    exposure_start = exposure_end - 90 * DAY
    total = data.counts(exposure_start, exposure_end - 1)
    night_mask = data._mask_range(exposure_start, exposure_end - 1) & data.night
    night_count = np.bincount(data.grid[night_mask], minlength=data.n_grids).astype(np.float64) if np.any(night_mask) else np.zeros(data.n_grids)
    eligible = total >= 5
    share = np.zeros(data.n_grids)
    share[eligible] = night_count[eligible] / total[eligible]
    threshold = float(np.median(share[eligible])) if np.any(eligible) else 1.0
    exposed = eligible & (share >= threshold)
    reference = eligible & (share < threshold)
    outcome, _ = data.labels(decision, decision + 30 * DAY - 1, data.theft_index)
    outcome_flag = outcome == 1
    volume = np.log1p(total)
    ranks = np.zeros(data.n_grids, dtype=np.int32)
    eligible_idx = np.flatnonzero(eligible)
    order = eligible_idx[np.argsort(volume[eligible_idx])]
    if len(order):
        cuts = np.array_split(order, 5)
        for stratum, indexes in enumerate(cuts):
            ranks[indexes] = stratum
    table = mantle_haenszel(exposed[eligible], outcome_flag[eligible], ranks[eligible])
    # 倾向得分：高夜间占比 ~ 历史案件量 + 区县均值，再按得分分层
    county_mean = county_other_mean(total, data.county_index, data.n_counties)
    design = np.column_stack([np.log1p(total[eligible]), np.log1p(county_mean[eligible])])
    propensity_y = exposed[eligible].astype(np.float64)
    propensity_head = fit_head(design, propensity_y)
    propensity = score_head(design, propensity_head)
    ps_rank = np.zeros(len(propensity), dtype=np.int32)
    ps_order = np.argsort(propensity)
    for stratum, indexes in enumerate(np.array_split(ps_order, 5)):
        ps_rank[indexes] = stratum
    propensity_table = mantle_haenszel(exposed[eligible], outcome_flag[eligible], ps_rank)
    weekly = weekly_counts(data, data.min_unix, decision)
    residual = {}
    total_week = weekly["total"]
    for code in FOCUS:
        series = weekly["types"][code]
        if len(series) < 20:
            continue
        design_w = np.column_stack([np.ones(len(series)), total_week])
        coef, *_ = np.linalg.lstsq(design_w, series, rcond=None)
        residual[code] = series - design_w @ coef
    correlations = {}
    codes = list(residual)
    for left_i, left in enumerate(codes):
        for right in codes[left_i + 1 :]:
            correlations[f"{left}-{right}"] = float(np.corrcoef(residual[left], residual[right])[0, 1])
    return {
        "exposure": "决策点前第 90 至 180 天的夜间案件占比高于同期有案网格的中位数",
        "outcome": "决策点后 30 天网格内是否出现盗窃",
        "adjustment_set": ["近 90 天案件量五分位", "区县其他网格案件量"],
        "volume_stratified": table,
        "propensity_stratified": propensity_table,
        "type_residual_correlation": correlations,
        "prior": "日常活动理论和犯罪热点理论预期夜间占比、近期案件、邻域案件与未来盗窃同向；系数符号在 30 天模型里单独列出",
    }


def weekly_counts(data: GraphData, start: int, end: int) -> dict:
    origin = _epoch_day(date(2013, 1, 7))
    mask = (data.unix >= start) & (data.unix <= end) & (data.days >= origin)
    week = (data.days[mask] - origin) // 7
    if len(week) == 0:
        return {"total": np.zeros(0), "types": {code: np.zeros(0) for code in FOCUS}, "week_index": np.zeros(0, dtype=np.int32)}
    max_week = int(week.max())
    indexes = np.arange(max_week + 1)
    total = np.bincount(week, minlength=max_week + 1).astype(np.float64)
    types = {}
    for code in FOCUS:
        type_mask = data.types[mask] == data.type_to_i[code]
        types[code] = np.bincount(week[type_mask], minlength=max_week + 1).astype(np.float64)
    holiday = np.zeros(max_week + 1, dtype=np.float64)
    holiday_weeks = (HOLIDAY_DAYS[HOLIDAY_DAYS >= origin] - origin) // 7
    holiday_weeks = holiday_weeks[(holiday_weeks >= 0) & (holiday_weeks <= max_week)]
    holiday[np.unique(holiday_weeks)] = 1.0
    return {"total": total, "types": types, "week_index": indexes, "holiday": holiday, "origin": origin}


def sequence_forecast(data: GraphData, test_origin: int) -> dict:
    _log("训练周尺度门控循环")
    observed = weekly_counts(data, data.min_unix, data.max_unix)
    origin = observed["origin"]
    test_week = int((_epoch_day(datetime.fromtimestamp(test_origin, TZ8).date()) - origin) // 7)
    reports = {}
    for code in FOCUS:
        series_all = observed["types"][code]
        total_all = observed["total"]
        if test_week < 60 or test_week >= len(series_all) - 2:
            continue
        series = series_all[:test_week]
        total = total_all[:test_week]
        holiday = observed["holiday"][:test_week]
        week = np.arange(test_week)
        months = np.array([(date(2013, 1, 7) + timedelta(days=int(item) * 7)).month for item in week], dtype=np.float64)
        x = np.column_stack(
            [
                np.log1p(series),
                np.log1p(total),
                np.sin(2 * np.pi * months / 12.0),
                np.cos(2 * np.pi * months / 12.0),
                holiday,
            ]
        )
        mu = x[:-1, :2].mean(axis=0)
        sd = np.maximum(x[:-1, :2].std(axis=0), 1e-6)
        inputs_n = x[:-1].copy()
        inputs_n[:, :2] = (x[:-1, :2] - mu) / sd
        target = np.log1p(series[1:]) - np.log1p(series[:-1])
        teacher = train_sequence(inputs_n, target, kind="gru", epochs=40, hidden=8, lr=0.03)
        student = train_sequence(inputs_n, target, kind="self", epochs=40, hidden=8, lr=0.03)
        full_month = np.array(
            [(date(2013, 1, 7) + timedelta(days=int(item) * 7)).month for item in range(len(series_all))],
            dtype=np.float64,
        )
        x_full = np.column_stack(
            [
                np.log1p(series_all),
                np.log1p(total_all),
                np.sin(2 * np.pi * full_month / 12.0),
                np.cos(2 * np.pi * full_month / 12.0),
                observed["holiday"][: len(series_all)],
            ]
        )
        x_full[:, :2] = (x_full[:, :2] - mu) / sd
        pred_teacher, _ = run_sequence(teacher["state"], x_full[:-1])
        pred_student, _ = run_sequence(student["state"], x_full[:-1])
        actual = series_all[test_week:]
        prev_log = np.log1p(series_all[test_week - 1 : len(series_all) - 1])
        pred_t = np.expm1(np.clip(prev_log + pred_teacher[test_week - 1 : len(series_all) - 1], 0, None))
        pred_s = np.expm1(np.clip(prev_log + pred_student[test_week - 1 : len(series_all) - 1], 0, None))
        naive = series_all[test_week - 1 : len(series_all) - 1]
        seasonal_index = np.arange(test_week, len(series_all)) - 52
        seasonal = series_all[np.maximum(seasonal_index, 0)]

        def mae(pred: np.ndarray) -> float:
            size = min(len(pred), len(actual))
            return float(np.mean(np.abs(pred[:size] - actual[:size]))) if size else 0.0

        month_means = []
        for month in range(1, 13):
            chosen = series[:-1][months[:-1] == month]
            month_means.append(float(chosen.mean()) if len(chosen) else 0.0)
        reports[code] = {
            "teacher_mae": mae(pred_t),
            "self_gru_mae": mae(pred_s),
            "naive_last_week_mae": mae(naive),
            "seasonal_lag52_mae": mae(seasonal),
            "teacher_params": teacher["params"],
            "self_gru_params": student["params"],
            "sequence_compression_ratio": teacher["params"] / max(student["params"], 1),
            "teacher_loss_start": teacher["loss_start"],
            "teacher_loss_end": teacher["loss_end"],
            "test_weeks": [float(value) for value in actual[:12]],
            "teacher_forecast": [float(value) for value in pred_t[:12]],
            "month_mean_weekly": month_means,
            "peak_month": int(np.argmax(month_means) + 1),
        }
    return reports


def contrastive(data: GraphData, decision: int) -> dict:
    _log("训练网格对比嵌入")
    features = data.any_features(decision)
    hist = np.expm1(features[:, ANY_FEATURES.index("log_count_90d")])
    eligible = np.flatnonzero(hist >= 3)
    rng = np.random.default_rng(SEED)
    if len(eligible) > 12000:
        eligible = rng.choice(eligible, size=12000, replace=False)
    province = np.array([code[:2] for code in data.admin_code])
    positives = []
    anchors = []
    negatives = []
    by_province: dict[str, np.ndarray] = {}
    for code in np.unique(province):
        by_province[code] = np.flatnonzero(province == code)
    for anchor in eligible:
        others = data.neighbors[anchor]
        others = others[others >= 0]
        if len(others) == 0:
            same = np.flatnonzero(data.county_index == data.county_index[anchor])
            same = same[same != anchor]
            if len(same) == 0:
                continue
            positive = int(rng.choice(same))
        else:
            positive = int(rng.choice(others))
        negative_pool = np.flatnonzero(province != province[anchor])
        negative = int(rng.choice(negative_pool))
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
    anchors_a = np.array(anchors)
    positive_a = np.array(positives)
    negative_a = np.array(negatives)
    dim = features.shape[1]
    embed = 8
    W = rng.normal(0, 1 / np.sqrt(dim), size=(dim, embed))
    losses = []
    batch = 1024
    tau = 0.2
    for _epoch in range(4):
        order = rng.permutation(len(anchors_a))
        epoch_loss = []
        for start in range(0, len(order), batch):
            chosen = order[start : start + batch]
            fa = features[anchors_a[chosen]]
            fp = features[positive_a[chosen]]
            fn = features[negative_a[chosen]]
            za = fa @ W
            zp = fp @ W
            zn = fn @ W
            pos = np.sum(za * zp, axis=1) / tau
            neg = np.sum(za * zn, axis=1) / tau
            weight = _sigmoid(neg - pos)
            epoch_loss.append(float(np.mean(np.logaddexp(0, neg - pos))))
            dza = weight[:, None] * (zn - zp) / tau
            dzp = weight[:, None] * (-za) / tau
            dzn = weight[:, None] * za / tau
            grad = (fa.T @ dza + fp.T @ dzp + fn.T @ dzn) / max(len(chosen), 1)
            W -= 0.05 * np.clip(grad, -1.0, 1.0)
        losses.append(float(np.mean(epoch_loss)))
    projected = features @ W
    county_embed = np.zeros((data.n_counties, embed))
    county_n = np.bincount(data.county_index, minlength=data.n_counties).astype(np.float64)
    for dim_i in range(embed):
        county_embed[:, dim_i] = np.bincount(data.county_index, weights=projected[:, dim_i], minlength=data.n_counties)
    county_embed /= np.maximum(county_n[:, None], 1.0)
    county_events = np.bincount(data.county_index[data.grid], minlength=data.n_counties)
    first_grid = np.full(data.n_counties, -1, dtype=np.int32)
    for grid_i, county_i in enumerate(data.county_index):
        if first_grid[county_i] < 0:
            first_grid[county_i] = grid_i
    province_of_county = np.array([data.admin_code[first_grid[index]][:2] for index in range(data.n_counties)])
    counts = np.zeros((data.n_counties, len(data.type_codes)), dtype=np.float64)
    before = data.unix < decision
    for type_index in range(len(data.type_codes)):
        selected = before & (data.types == type_index)
        if np.any(selected):
            counts[:, type_index] = np.bincount(data.county_index[data.grid[selected]], minlength=data.n_counties)
    totals = counts.sum(axis=1)
    share = counts / np.maximum(totals[:, None], 1.0)
    national = counts.sum(axis=0) / max(float(counts.sum()), 1.0)
    residual = share - national
    active = np.flatnonzero(totals >= 80)
    pairs = []
    if len(active) > 1:
        block = residual[active]
        norms = np.maximum(np.linalg.norm(block, axis=1), 1e-6)
        similarity = (block @ block.T) / (norms[:, None] * norms[None, :])
        provinces = province_of_county[active]
        for left_i in range(len(active)):
            for right_i in range(left_i + 1, len(active)):
                if provinces[left_i] == provinces[right_i]:
                    continue
                value = float(similarity[left_i, right_i])
                if value < 0.35:
                    continue
                pairs.append((value, int(active[left_i]), int(active[right_i])))
    pairs.sort(reverse=True)
    top = []
    for sim, left, right in pairs[:12]:
        left_grid = int(first_grid[left])
        right_grid = int(first_grid[right])
        top.append(
            {
                "similarity": round(sim, 4),
                "county_a": data.admin_code[left_grid],
                "name_a": data.admin_name[left_grid],
                "city_a": data.city_name[left_grid],
                "county_b": data.admin_code[right_grid],
                "name_b": data.admin_name[right_grid],
                "city_b": data.city_name[right_grid],
            }
        )
    return {
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "anchors": int(len(anchors_a)),
        "similar_counties": top,
        "meaning": "跨省名单比较的是区县罪名结构相对全国水平的残差，不是嫌疑人或团伙。对比学习只用于记录嵌入损失。",
    }


def cross_region(data: GraphData, test_origin: int) -> dict:
    _log("统计跨行政区边界的 24 小时近重复")
    train = data.unix < test_origin
    order = np.argsort(data.grid, kind="mergesort")
    grid_sorted = data.grid[order]
    starts = np.flatnonzero(np.diff(grid_sorted, prepend=-1))
    ends = np.append(starts[1:], len(grid_sorted))
    times = [np.array([], dtype=np.int64) for _ in range(data.n_grids)]
    types = [np.array([], dtype=np.int16) for _ in range(data.n_grids)]
    for grid_id, start, end in zip(grid_sorted[starts], starts, ends):
        rows = order[start:end]
        keep = train[rows]
        rows = rows[keep]
        if len(rows) == 0:
            continue
        seq = np.argsort(data.unix[rows], kind="mergesort")
        rows = rows[seq]
        times[grid_id] = data.unix[rows]
        types[grid_id] = data.types[rows]
    pair_counts: dict[tuple, int] = {}
    test_counts: dict[tuple, int] = {}
    edges = []
    for left in range(data.n_grids):
        for direction in range(4):
            right = int(data.neighbors[left, direction])
            if right < 0 or left > right:
                continue
            if data.admin_code[left] == data.admin_code[right]:
                continue
            edges.append((left, right))
    for left, right in edges:
        _accumulate_pair(times[left], types[left], times[right], types[right], data, left, right, pair_counts)
        _accumulate_pair(times[right], types[right], times[left], types[left], data, right, left, pair_counts)
    test_times = [np.array([], dtype=np.int64) for _ in range(data.n_grids)]
    test_types = [np.array([], dtype=np.int16) for _ in range(data.n_grids)]
    test_mask = data.unix >= test_origin
    for grid_id, start, end in zip(grid_sorted[starts], starts, ends):
        rows = order[start:end]
        rows = rows[test_mask[rows]]
        if len(rows) == 0:
            continue
        seq = np.argsort(data.unix[rows], kind="mergesort")
        rows = rows[seq]
        test_times[grid_id] = data.unix[rows]
        test_types[grid_id] = data.types[rows]
    for left, right in edges:
        _accumulate_pair(test_times[left], test_types[left], test_times[right], test_types[right], data, left, right, test_counts)
        _accumulate_pair(test_times[right], test_types[right], test_times[left], test_types[left], data, right, left, test_counts)
    kept = []
    for key, count in pair_counts.items():
        kind, source, target = key
        if count < 5:
            continue
        kept.append(
            {
                "kind": "全部罪名" if kind == "any" else data.type_codes[int(kind)],
                "source_admin": source,
                "target_admin": target,
                "train_count": int(count),
                "test_count": int(test_counts.get(key, 0)),
            }
        )
    kept.sort(key=lambda item: item["train_count"], reverse=True)
    name_of = {}
    city_of = {}
    for index, code in enumerate(data.admin_code):
        name_of.setdefault(code, data.admin_name[index])
        city_of.setdefault(code, data.city_name[index])
    for item in kept:
        item["source_name"] = name_of.get(item["source_admin"], item["source_admin"])
        item["target_name"] = name_of.get(item["target_admin"], item["target_admin"])
        item["source_city"] = city_of.get(item["source_admin"], "")
        item["target_city"] = city_of.get(item["target_admin"], "")
        item["cross_city"] = item["source_city"] != item["target_city"]
    any_edges = [item for item in kept if item["kind"] == "全部罪名"]
    adjacency: dict[str, list[tuple[str, int]]] = {}
    for item in any_edges:
        adjacency.setdefault(item["source_admin"], []).append((item["target_admin"], item["train_count"]))
    for source in adjacency:
        adjacency[source].sort(key=lambda pair: pair[1], reverse=True)
    strength = {}
    for item in any_edges:
        strength[item["source_admin"]] = strength.get(item["source_admin"], 0) + item["train_count"]
        strength[item["target_admin"]] = strength.get(item["target_admin"], 0) + item["train_count"]
    core = sorted(strength, key=strength.get, reverse=True)[:10]
    paths = []
    sources = sorted(adjacency, key=lambda code: strength.get(code, 0), reverse=True)[:25]
    for source in sources:
        for mid, weight_ab in adjacency.get(source, [])[:6]:
            if mid == source:
                continue
            paths.append(([source, mid], weight_ab))
            for target, weight_bc in adjacency.get(mid, [])[:6]:
                if target in (source, mid):
                    continue
                paths.append(([source, mid, target], min(weight_ab, weight_bc)))
    paths.sort(key=lambda item: item[1], reverse=True)
    unique_paths = []
    seen = set()
    for nodes, weight in paths:
        marker = tuple(nodes)
        if marker in seen:
            continue
        seen.add(marker)
        unique_paths.append(
            {
                "nodes": list(nodes),
                "names": [name_of.get(code, code) for code in nodes],
                "cities": [city_of.get(code, "") for code in nodes],
                "min_edge_count": int(weight),
                "hops": len(nodes) - 1,
            }
        )
        if len(unique_paths) >= 15:
            break
    top_any = any_edges[:20]
    recur = sum(1 for item in top_any if item["test_count"] > 0)
    any_keys = {key for key in pair_counts if key[0] == "any"} | {key for key in test_counts if key[0] == "any"}
    true_positive = false_positive = false_negative = 0
    for key in any_keys:
        predicted = pair_counts.get(key, 0) >= 5
        observed = test_counts.get(key, 0) >= 5
        if predicted and observed:
            true_positive += 1
        elif predicted:
            false_positive += 1
        elif observed:
            false_negative += 1
    return {
        "boundary_grid_pairs": len(edges),
        "edges_before_prune": len(pair_counts),
        "edges_kept_min5": len(kept),
        "top_edges": kept[:40],
        "top20_any_recurring_in_test": recur,
        "link_precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else None,
        "link_miss_rate": false_negative / (true_positive + false_negative) if true_positive + false_negative else None,
        "link_true_positive": true_positive,
        "link_false_positive": false_positive,
        "link_false_negative": false_negative,
        "core_counties": [
            {"admin_code": code, "name": name_of.get(code, code), "city": city_of.get(code, ""), "strength": int(strength[code])}
            for code in core
        ],
        "paths": unique_paths,
        "meaning": "边表示两个相邻且分属不同区县的网格，在 24 小时内先后发生案件。这是边界近重复，不是嫌疑人跨区轨迹。",
    }


def _accumulate_pair(times_a, types_a, times_b, types_b, data: GraphData, left: int, right: int, bucket: dict) -> None:
    if len(times_a) == 0 or len(times_b) == 0:
        return
    source = data.admin_code[left]
    target = data.admin_code[right]
    j = 0
    any_count = 0
    type_hits: dict[int, int] = {}
    n_b = len(times_b)
    for index, stamp in enumerate(times_a):
        while j < n_b and times_b[j] <= stamp:
            j += 1
        k = j
        matched_type = False
        matched_any = False
        this_type = int(types_a[index])
        while k < n_b and times_b[k] - stamp <= DAY:
            matched_any = True
            if int(types_b[k]) == this_type:
                matched_type = True
                break
            k += 1
        if matched_any:
            any_count += 1
        if matched_type:
            type_hits[this_type] = type_hits.get(this_type, 0) + 1
    if any_count:
        key = ("any", source, target)
        bucket[key] = bucket.get(key, 0) + any_count
    for type_index, count in type_hits.items():
        key = (type_index, source, target)
        bucket[key] = bucket.get(key, 0) + count


def export_predictions(data: GraphData, trained: dict) -> dict:
    _log("导出热点、类型和解释")
    REASON_DIR.mkdir(parents=True, exist_ok=True)
    end = trained["times"]["horizon_end"]
    latest = {
        ("any", 1): end - DAY,
        ("any", 7): end - 7 * DAY,
        ("any", 30): end - 30 * DAY,
    }
    summaries = []
    explanations = []
    for horizon, fraction, filename in ((1, 0.01, "hotspot_24h.csv"), (7, 0.02, "hotspot_7d.csv"), (30, 0.05, "hotspot_30d.csv")):
        decision = latest[("any", horizon)]
        bundle = trained["heads"][("any", horizon)]
        features = bundle["features"](decision)
        scores = score_head(features, bundle["head"])
        _, events = data.labels(decision, min(decision + horizon * DAY - 1, data.max_unix), None)
        holdout = np.bincount(events, minlength=data.n_grids) if len(events) else np.zeros(data.n_grids, dtype=np.int32)
        rows = _top_rows(data, features, scores, holdout, fraction, horizon, "全部罪名", ANY_FEATURES)
        _write_csv(REASON_DIR / filename, rows)
        summaries.append({"file": filename, "decision": datetime.fromtimestamp(decision, TZ8).isoformat(), "top_capture_check": rows[0]["grid_id"] if rows else ""})
        if horizon == 30:
            for row in rows[:15]:
                explanations.append(_explain(row, "全部罪名", 30))
    for code in FOCUS:
        decision = trained["times"]["horizon_end"] - 30 * DAY
        bundle = trained["heads"][(code, 30)]
        features = bundle["features"](decision)
        scores = score_head(features, bundle["head"])
        type_index = data.type_to_i[code]
        _, events = data.labels(decision, min(decision + 30 * DAY - 1, data.max_unix), type_index)
        holdout = np.bincount(events, minlength=data.n_grids) if len(events) else np.zeros(data.n_grids, dtype=np.int32)
        label = data.type_names.get(code, code)
        rows = _top_rows(data, features, scores, holdout, 0.05, 30, label, TYPE_FEATURES)
        _write_csv(REASON_DIR / f"type_{code}_30d.csv", rows)
        for row in rows[:5]:
            explanations.append(_explain(row, label, 30))
    _write_csv(REASON_DIR / "explanations.csv", explanations)
    return {"prediction_files": summaries, "explanation_rows": len(explanations)}


def _top_rows(data: GraphData, features: np.ndarray, scores: np.ndarray, holdout: np.ndarray, fraction: float, horizon: int, crime_name: str, names: list[str]) -> list[dict]:
    k = 200
    order = np.argpartition(scores, -k)[-k:]
    order = order[np.argsort(scores[order])[::-1]]
    n_grids = len(scores)
    high_cut = max(1, int(round(n_grids * fraction)))
    mid_cut = max(high_cut + 1, int(round(n_grids * (fraction + (0.04 if horizon == 1 else 0.08 if horizon == 7 else 0.15)))))
    rank_of = {int(index): rank for rank, index in enumerate(np.argsort(scores)[::-1], start=1)}
    rows = []
    name_index = {name: pos for pos, name in enumerate(names)}
    for index in order:
        rank = rank_of[int(index)]
        if rank <= high_cut:
            tier = "高"
        elif rank <= mid_cut:
            tier = "中"
        else:
            tier = "低"
        def feature(name: str) -> float | None:
            if name not in name_index:
                return None
            return float(features[index, name_index[name]])

        def pick(*candidates: str) -> float:
            for name in candidates:
                value = feature(name)
                if value is not None:
                    return value
            return 0.0

        rows.append(
            {
                "rank": rank,
                "tier": tier,
                "crime_type": crime_name,
                "horizon_days": horizon,
                "grid_id": data.grid_ids[index],
                "city_name": data.city_name[index],
                "admin_name": data.admin_name[index],
                "admin_code": data.admin_code[index],
                "score": float(scores[index]),
                "hist_90d": round(float(np.expm1(pick("log_count_90d", "log_type_90d"))), 2),
                "neighbor_90d": round(float(np.expm1(pick("log_neighbor_90d", "log_type_neighbor_30d"))), 2),
                "night_share": round(pick("night_share_90d", "night_share_type_90d"), 4),
                "holiday_share": round(pick("holiday_share_90d"), 4),
                "hawkes": round(float(np.expm1(pick("log_hawkes_self", "log_hawkes_type"))), 4),
                "holdout_events": int(holdout[index]),
            }
        )
    return rows


def _explain(row: dict, crime_name: str, horizon: int) -> dict:
    text = (
        f"{row['city_name']}{row['admin_name']}网格 {row['grid_id']} 被列为未来{horizon}天{crime_name}{row['tier']}风险。"
        f"依据是决策点前 90 天该网格有 {row['hist_90d']} 起相关历史案件，夜间占比 {row['night_share']:.0%}，"
        f"相邻网格同期强度 {row['neighbor_90d']}，14 天衰减后的自身霍克斯强度 {row['hawkes']}。"
        f"评分没有使用人口密度或交通流量。"
    )
    return {**row, "evidence": text}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    _self_check()
    frame = load_frame()
    catalog = build_catalog(frame)
    data = GraphData(frame, catalog)
    trained = train_and_evaluate(data)
    causal = causal_report(data, trained["times"]["cutoffs"][0])
    sequence = sequence_forecast(data, trained["times"]["test_origin"])
    embedding = contrastive(data, trained["times"]["test_origin"] - 30 * DAY)
    region = cross_region(data, trained["times"]["test_origin"])
    exported = export_predictions(data, trained)
    prior_signs = {}
    coefficients = trained["metrics"]["any_30d"]["coefficients"]["coefficients"]
    for name in ("log_count_30d", "log_neighbor_30d", "night_share_90d", "log_hawkes_self"):
        value = coefficients[name]
        if value > 1e-6:
            prior_signs[name] = "同向"
        elif value < -1e-6:
            prior_signs[name] = "反向"
        else:
            prior_signs[name] = "约束为0"
    report = {
        "built_at": datetime.now(TZ8).isoformat(),
        "decision_note": "2019年7月起裁判文书明显减少。评价截止到最近一个30天内仍有至少1000起案件的日期，并留出这之前的90天。训练不使用留出时段。",
        "data_end": datetime.fromtimestamp(data.max_unix, TZ8).isoformat(),
        "horizon_end": datetime.fromtimestamp(trained["times"]["horizon_end"], TZ8).isoformat(),
        "test_origin": datetime.fromtimestamp(trained["times"]["test_origin"], TZ8).isoformat(),
        "grids": data.n_grids,
        "events": int(len(data.unix)),
        "architecture": {
            "shared_encoder": "网格自身、四邻域、区县三档空间计数，夜间/节假日/周末占比，14 天霍克斯动态图特征",
            "heads": "全部罪名的 24 小时、7 天、30 天，以及盗窃、诈骗、故意伤害的 7 天和 30 天",
            "sequence": "周案件数的残差门控循环；自门控循环的门只依赖当前周输入",
            "distill": trained["metrics"]["student_30d"]["method"],
            "not_used": "源数据没有人口密度、交通流量、流动人口和嫌疑人标识，这些没有进入模型",
            "compute": "NumPy 向量化，单机 CPU。没有调用 GPU 集群。",
        },
        "metrics": trained["metrics"],
        "prior_signs": prior_signs,
        "causal": causal,
        "weekly_forecast": sequence,
        "contrastive": {
            "loss_start": embedding["loss_start"],
            "loss_end": embedding["loss_end"],
            "anchors": embedding["anchors"],
            "meaning": embedding["meaning"],
            "similar_counties": embedding["similar_counties"],
        },
        "cross_region": {
            "boundary_grid_pairs": region["boundary_grid_pairs"],
            "edges_before_prune": region["edges_before_prune"],
            "edges_kept_min5": region["edges_kept_min5"],
            "top20_any_recurring_in_test": region["top20_any_recurring_in_test"],
            "core_counties": region["core_counties"],
            "paths": region["paths"],
            "meaning": region["meaning"],
        },
        "export": exported,
    }
    # 系数对象太大但可接受；嵌入和路径已截断
    REASON_DIR.mkdir(parents=True, exist_ok=True)
    (REASON_DIR / "reason_report.json").write_text(json.dumps(_plain(report), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(REASON_DIR / "cross_region_edges.csv", region["top_edges"])
    _write_csv(REASON_DIR / "cross_region_paths.csv", region["paths"])
    _write_csv(REASON_DIR / "similar_counties.csv", embedding["similar_counties"])
    _log(json.dumps(_plain({key: trained["metrics"][key]["capture_mean"] if "capture_mean" in trained["metrics"][key] else trained["metrics"][key] for key in ("any_30d", "student_30d")}), ensure_ascii=False, indent=2))
    _log(f"报告已写入 {REASON_DIR / 'reason_report.json'}")


if __name__ == "__main__":
    main()
