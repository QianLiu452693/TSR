"""县级行政区划代码与点面包含。

行政区划名称来自 GB/T 2260 风格的 pca-code 词典；
县级矢量面来自阿里云 DataV 公开地理边界，缓存于 data/county_geo。
"""

from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*", category=UserWarning)
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape
from shapely import make_valid

from tkg_pipeline.config import COUNTY_DIR, PCA_JSON

ALIYUN = "https://geo.datav.aliyun.com/areas_v3/bound/{code}_full.json"


def _pad_code(code: str) -> str:
    code = str(code).strip()
    if len(code) == 2:
        return code + "0000"
    if len(code) == 4:
        return code + "00"
    if len(code) < 6:
        return code.ljust(6, "0")
    return code[:6]


def load_pca_names() -> dict[str, str]:
    data = json.loads(PCA_JSON.read_text(encoding="utf-8"))
    names: dict[str, str] = {}
    for prov in data:
        pcode = _pad_code(prov["code"])
        names[pcode] = prov["name"]
        for city in prov.get("children") or []:
            ccode = _pad_code(city["code"])
            names[ccode] = city["name"]
            for county in city.get("children") or []:
                kcode = _pad_code(county["code"])
                names[kcode] = county["name"]
    return names


def city_codes() -> list[str]:
    data = json.loads(PCA_JSON.read_text(encoding="utf-8"))
    codes = []
    for prov in data:
        for city in prov.get("children") or []:
            codes.append(_pad_code(city["code"]))
    return sorted(set(codes))


def _download_one(code: str, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 800:
        return "cached"
    url = ALIYUN.format(code=code)
    try:
        with urlopen(url, timeout=30) as resp:
            payload = resp.read()
        if len(payload) < 800 or b'"features"' not in payload:
            return "empty"
        dest.write_bytes(payload)
        return "ok"
    except (HTTPError, URLError, TimeoutError, OSError):
        return "fail"


def ensure_county_geojsons() -> list[Path]:
    COUNTY_DIR.mkdir(parents=True, exist_ok=True)
    codes = city_codes()
    print(f"准备县级矢量面：{len(codes)} 个地级单元", flush=True)
    status = {"ok": 0, "cached": 0, "fail": 0, "empty": 0}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_download_one, code, COUNTY_DIR / f"{code}.json"): code for code in codes
        }
        for fut in as_completed(futures):
            status[fut.result()] = status.get(fut.result(), 0) + 1
    files = sorted(COUNTY_DIR.glob("*.json"))
    print(
        f"矢量面缓存 {len(files)} 个（新下载 {status['ok']}，已缓存 {status['cached']}，失败 {status['fail'] + status['empty']}）",
        flush=True,
    )
    if len(files) < 200:
        raise RuntimeError("县级矢量面不足，无法做点面包含。")
    return files


def _parent_code(prop: dict) -> str:
    parent = prop.get("parent") or {}
    if isinstance(parent, dict) and parent.get("adcode"):
        return _pad_code(str(parent["adcode"]))
    ad = prop.get("adcode")
    return _pad_code(str(ad)[:4])


def load_county_gdf() -> gpd.GeoDataFrame:
    files = ensure_county_geojsons()
    names = load_pca_names()
    rows = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for feat in payload.get("features") or []:
            geom = feat.get("geometry")
            prop = feat.get("properties") or {}
            if not geom or prop.get("adcode") is None:
                continue
            try:
                geometry = make_valid(shape(geom))
            except Exception:
                continue
            if geometry.is_empty:
                continue
            level = str(prop.get("level") or "")
            if level in {"province", "city", "country"}:
                continue
            ad = _pad_code(str(prop["adcode"]))
            if len(str(prop["adcode"])) > 6:
                # 东莞、中山等以镇街为要素时，归到地级代码
                ad = _pad_code(str(prop["adcode"])[:6])
            city = _parent_code(prop)
            province = ad[:2] + "0000"
            city_name = names.get(city, "")
            if city_name in {"市辖区", "县", "省直辖县级行政区划", "自治区直辖县级行政区划"}:
                city_name = names.get(province, city_name)
            rows.append(
                {
                    "admin_code": ad,
                    "admin_name": prop.get("name") or names.get(ad, ""),
                    "city_code": city,
                    "city_name": city_name or names.get(city, ""),
                    "province_code": province,
                    "province_name": names.get(province, ""),
                    "geometry": geometry,
                }
            )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf["geometry"] = gdf.geometry.simplify(0.001, preserve_topology=True)
    gdf = gdf[~gdf.geometry.is_empty].copy()
    # 同一区划被重复收录时保留顶点更多的面
    gdf["_n"] = gdf.geometry.length
    gdf = gdf.sort_values("_n", ascending=False).drop_duplicates("admin_code", keep="first")
    gdf = gdf.drop(columns="_n").reset_index(drop=True)
    print(f"县级面要素 {len(gdf)} 个", flush=True)
    return gdf


def build_name_index(gdf: gpd.GeoDataFrame) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}
    for rec in gdf.drop(columns="geometry").to_dict("records"):
        name = str(rec["admin_name"])
        index.setdefault(name, []).append(rec)
    return index


def _cell(value) -> str:
    if value is None or value is pd.NA:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "<na>", "none"}:
        return ""
    return text


def court_admin(court_name: str, city_name: str, name_index: dict[str, list[dict]]) -> dict | None:
    court_name = _cell(court_name)
    city_name = _cell(city_name)
    text = f"{court_name}{city_name}"
    if "人民法院" not in text:
        return None
    token = text.split("人民法院", 1)[0]
    token = token.replace(" ", "")
    found = ""
    for i in range(len(token)):
        piece = token[i:]
        if piece in name_index:
            found = piece
            break
    if not found:
        return None
    cands = name_index[found]
    city_name = city_name or ""
    ranked = []
    for cand in cands:
        score = 0
        if city_name and (city_name in str(cand["city_name"]) or str(cand["city_name"]) in city_name):
            score += 2
        if str(cand["city_name"]) and str(cand["city_name"]) in text:
            score += 2
        if str(cand["province_name"]) and str(cand["province_name"])[:2] in text:
            score += 1
        ranked.append((score, cand))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def assign_admin(df: pd.DataFrame, counties: gpd.GeoDataFrame) -> pd.DataFrame:
    print("点面包含匹配行政区划…", flush=True)
    name_index = build_name_index(counties)
    right = counties[
        ["admin_code", "admin_name", "city_code", "city_name", "province_code", "province_name", "geometry"]
    ]
    joined_parts = []
    batch = 250_000
    for start in range(0, len(df), batch):
        stop = min(start + batch, len(df))
        block = df.iloc[start:stop]
        points = gpd.GeoDataFrame(
            {"row_id": block.index.to_numpy()},
            geometry=gpd.points_from_xy(block["longitude"], block["latitude"]),
            crs="EPSG:4326",
        ).set_index("row_id")
        part = gpd.sjoin(points, right, how="left", predicate="within")
        part = part[~part.index.duplicated(keep="first")]
        joined_parts.append(part)
        print(f"  点面批次 {stop}/{len(df)}", flush=True)
    joined = pd.concat(joined_parts)
    points = gpd.GeoDataFrame(
        {"row_id": df.index.to_numpy()},
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    ).set_index("row_id")
    out = df.copy()
    for col in ["admin_code", "admin_name", "city_code", "city_name", "province_code", "province_name"]:
        out[col] = pd.NA
    out.loc[joined.index, "admin_code"] = joined["admin_code"].to_numpy()
    out.loc[joined.index, "admin_name"] = joined["admin_name"].to_numpy()
    out.loc[joined.index, "city_code"] = joined["city_code"].to_numpy()
    out.loc[joined.index, "city_name"] = joined["city_name"].to_numpy()
    out.loc[joined.index, "province_code"] = joined["province_code"].to_numpy()
    out.loc[joined.index, "province_name"] = joined["province_name"].to_numpy()
    out["admin_match"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["admin_distance_m"] = pd.Series(pd.NA, index=out.index, dtype="Float64")
    hit = out["admin_code"].notna()
    out.loc[hit, "admin_match"] = "pip"
    out.loc[hit, "admin_distance_m"] = 0.0

    missing = out.index[~hit]
    print(f"点面直接命中 {int(hit.sum())}，未命中 {len(missing)}，改走最近面与法院名称", flush=True)
    if len(missing):
        near = gpd.sjoin_nearest(
            points.loc[missing],
            right,
            how="left",
            max_distance=0.05,
            distance_col="dist_deg",
        )
        near = near[~near.index.duplicated(keep="first")]
        near = near[near["admin_code"].notna()]
        for col in ["admin_code", "admin_name", "city_code", "city_name", "province_code", "province_name"]:
            out.loc[near.index, col] = near[col].to_numpy()
        out.loc[near.index, "admin_match"] = "nearest"
        out.loc[near.index, "admin_distance_m"] = np.round(near["dist_deg"].to_numpy(dtype=float) * 111_320.0, 1)

    still = out.index[out["admin_code"].isna()]
    if len(still):
        codes, names, cities, city_names, provs, prov_names, methods = [], [], [], [], [], [], []
        keep_idx = []
        pending = out.loc[still, ["court_name", "city"]]
        for rec in pending.itertuples():
            cand = court_admin(rec.court_name, rec.city, name_index)
            if cand is None:
                continue
            idx = rec.Index
            keep_idx.append(idx)
            codes.append(cand["admin_code"])
            names.append(cand["admin_name"])
            cities.append(cand["city_code"])
            city_names.append(cand["city_name"])
            provs.append(cand["province_code"])
            prov_names.append(cand["province_name"])
            methods.append("court")
        if keep_idx:
            out.loc[keep_idx, "admin_code"] = codes
            out.loc[keep_idx, "admin_name"] = names
            out.loc[keep_idx, "city_code"] = cities
            out.loc[keep_idx, "city_name"] = city_names
            out.loc[keep_idx, "province_code"] = provs
            out.loc[keep_idx, "province_name"] = prov_names
            out.loc[keep_idx, "admin_match"] = methods
    return out
