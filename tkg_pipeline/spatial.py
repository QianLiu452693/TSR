"""高斯-克吕格投影、500 米城市网格与 DBSCAN 孤立点识别。"""

from __future__ import annotations

import math
import warnings

warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*", category=UserWarning)

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

from tkg_pipeline.config import DBSCAN_EPS_M, DBSCAN_MIN_SAMPLES, GRID_INDEX_MAX, GRID_SIZE_M


def dbscan_keep_mask(
    latitude: np.ndarray,
    longitude: np.ndarray,
    eps_m: float = DBSCAN_EPS_M,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> np.ndarray:
    """网格索引上的 DBSCAN 噪声规则。

    将平面分成边长为 eps/sqrt(2) 的格网后，同一格内任意两点距离不超过 eps。
    若某点所在格及其八邻域的点数小于 min_samples，则该点为孤立噪声。
    """
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    lat0 = math.radians(float(np.nanmedian(lat)))
    x = lon * (111_320.0 * math.cos(lat0))
    y = lat * 110_540.0
    cell = eps_m / math.sqrt(2.0)
    ix = np.floor(x / cell).astype(np.int64)
    iy = np.floor(y / cell).astype(np.int64)
    frame = pd.DataFrame({"ix": ix, "iy": iy})
    counts = frame.groupby(["ix", "iy"], sort=False).size()
    total = np.zeros(len(frame), dtype=np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            key = pd.MultiIndex.from_arrays([ix + dx, iy + dy])
            total += counts.reindex(key).fillna(0).to_numpy(dtype=np.int64)
    return total >= min_samples


def _transformer(lon0: int) -> Transformer:
    proj = (
        f"+proj=tmerc +lat_0=0 +lon_0={lon0} +k=1 +x_0=500000 +y_0=0 "
        "+ellps=GRS80 +units=m +no_defs"
    )
    return Transformer.from_crs("EPSG:4326", proj, always_xy=True)


def _city_frames(counties: gpd.GeoDataFrame) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    if counties is None or len(counties) == 0:
        return meta
    grouped = counties.groupby("city_code", sort=False)
    for city_code, sub in grouped:
        lon = float(sub.geometry.centroid.x.median())
        lon0 = int(round(lon / 3.0) * 3)
        trans = _transformer(lon0)
        xs: list[float] = []
        ys: list[float] = []
        for geom in sub.geometry:
            minx, miny, maxx, maxy = geom.bounds
            px, py = trans.transform([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
            xs.extend(px)
            ys.extend(py)
        origin_x = math.floor(min(xs) / GRID_SIZE_M) * GRID_SIZE_M
        origin_y = math.floor(min(ys) / GRID_SIZE_M) * GRID_SIZE_M
        city_name = str(sub["city_name"].dropna().iloc[0]) if sub["city_name"].notna().any() else ""
        meta[str(city_code)] = {
            "city_code": str(city_code),
            "city_name": city_name,
            "gk_lon0": lon0,
            "origin_x": float(origin_x),
            "origin_y": float(origin_y),
            "transformer": trans,
        }
    return meta


def attach_grids(df: pd.DataFrame, counties: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    print("高斯-克吕格投影并划分 500 米网格…", flush=True)
    meta = _city_frames(counties)
    out = df.copy()
    out["proj_x"] = np.nan
    out["proj_y"] = np.nan
    out["gk_lon0"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["grid_x"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["grid_y"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    overflow = 0
    negative = 0
    for city_code, index in out.groupby("city_code", sort=False).groups.items():
        code = str(city_code)
        block = out.loc[index]
        info = meta.get(code)
        if info is None:
            lon = float(block["longitude"].median())
            lon0 = int(round(lon / 3.0) * 3)
            trans = _transformer(lon0)
            px, py = trans.transform(block["longitude"].to_numpy(), block["latitude"].to_numpy())
            origin_x = math.floor(float(np.min(px)) / GRID_SIZE_M) * GRID_SIZE_M
            origin_y = math.floor(float(np.min(py)) / GRID_SIZE_M) * GRID_SIZE_M
            info = {
                "city_code": code,
                "city_name": str(block["city_name"].iloc[0]) if "city_name" in block else "",
                "gk_lon0": lon0,
                "origin_x": float(origin_x),
                "origin_y": float(origin_y),
                "transformer": trans,
            }
            meta[code] = info
        trans = info["transformer"]
        px, py = trans.transform(block["longitude"].to_numpy(), block["latitude"].to_numpy())
        gx = np.floor((px - info["origin_x"]) / GRID_SIZE_M).astype(np.int64)
        gy = np.floor((py - info["origin_y"]) / GRID_SIZE_M).astype(np.int64)
        valid = (gx >= 0) & (gy >= 0) & (gx <= GRID_INDEX_MAX) & (gy <= GRID_INDEX_MAX)
        negative += int(((gx < 0) | (gy < 0)).sum())
        overflow += int(((gx > GRID_INDEX_MAX) | (gy > GRID_INDEX_MAX)).sum())
        gx_out = pd.array(gx, dtype="Int64")
        gy_out = pd.array(gy, dtype="Int64")
        gx_out[~valid] = pd.NA
        gy_out[~valid] = pd.NA
        out.loc[index, "proj_x"] = px
        out.loc[index, "proj_y"] = py
        out.loc[index, "gk_lon0"] = info["gk_lon0"]
        out.loc[index, "grid_x"] = gx_out.to_numpy(dtype=object, na_value=pd.NA)
        out.loc[index, "grid_y"] = gy_out.to_numpy(dtype=object, na_value=pd.NA)
    kept = out["grid_x"].notna() & out["grid_y"].notna()
    out["grid_align_status"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[kept, "grid_align_status"] = "ok"
    out.loc[~kept, "grid_align_status"] = "out_of_frame"
    code = pd.Series(pd.NA, index=out.index, dtype="string")
    code.loc[kept] = (
        out.loc[kept, "city_code"].astype(str).str.zfill(6)
        + out.loc[kept, "grid_x"].astype(int).map(lambda v: f"{v:03d}")
        + out.loc[kept, "grid_y"].astype(int).map(lambda v: f"{v:03d}")
    )
    out["grid_code"] = code
    out["grid_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[kept, "grid_id"] = "GG" + code.loc[kept]
    meta_rows = []
    for info in meta.values():
        if info["city_code"] not in set(out["city_code"].astype(str)):
            continue
        meta_rows.append(
            {
                "city_code": info["city_code"],
                "city_name": info["city_name"],
                "gk_lon0": info["gk_lon0"],
                "origin_x": info["origin_x"],
                "origin_y": info["origin_y"],
                "grid_size_m": GRID_SIZE_M,
            }
        )
    meta_df = pd.DataFrame(meta_rows)
    stats = {
        "grid_out_of_frame": int((~kept).sum()),
        "grid_negative": negative,
        "grid_overflow": overflow,
        "cities": int(out["city_code"].nunique()),
    }
    occupied = int(out.loc[kept, "grid_code"].nunique()) if bool(kept.any()) else 0
    print(
        f"网格完成：{occupied} 个占用网格，{stats['cities']} 个地级单元，超出 12 位编码 {stats['grid_out_of_frame']} 条不并格",
        flush=True,
    )
    return out, meta_df, stats
