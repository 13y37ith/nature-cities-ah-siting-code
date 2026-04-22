import argparse
import math
import os
import re
import time

import pandas as pd
import requests
from tqdm import tqdm


CAT_NAME = {
    "culture": "Culture",
    "sports": "Sports",
    "health": "Healthcare",
    "commerce": "Commerce",
    "green": "GreenSpace",
    "rail": "RailTransit",
    "road": "SurfaceTransport",
}

KEYWORD_MAP = {
    "culture": "文化中心|图书馆|少年宫|青少年活动中心|文化活动中心|文化活动室|社区文化",
    "sports": "体育馆|体育场|健身馆|健身中心|游泳馆|游泳池|运动场|球场|儿童游乐|游乐场|健身点",
    "health": "医院|医疗中心|社区卫生服务中心|卫生服务中心|卫生服务站|卫生服务点|妇幼保健院|护理院|药店|诊所",
    "commerce": "菜市场|农贸市场|室内菜市场|便利店|超市|商场|社区食堂|生活服务",
    "green": "公园|公共绿地|绿地|城市绿地|广场",
    "rail": "地铁站|轨道交通站|轻轨站|火车站|铁路车站|市域铁路|城际铁路|有轨电车站|有轨车站",
    "road": "公交站|公交枢纽|公交换乘|长途汽车站|客运站|公交首末站",
}

BAD_PATTERNS = {
    "culture": r"公司|传媒|广告|培训|学校|机构|工作室|设计|咨询|科技",
    "sports": r"用品|批发|公司|销售|专卖|维修|租赁|体育用品",
    "green": r"公司|置业|营销中心|售楼处|地产|物业",
}

AMAP_AROUND_URL = "https://restapi.amap.com/v3/place/around"
AMAP_DRIVING_URL = "https://restapi.amap.com/v3/direction/driving"
AMAP_WALKING_URL = "https://restapi.amap.com/v3/direction/walking"


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def parse_lonlat(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None, None
    text = str(value).strip().replace("，", ",").replace(" ", ",")
    text = re.sub(r",+$", "", text)
    parts = [p for p in text.split(",") if p]
    if len(parts) < 2:
        return None, None
    return safe_float(parts[0]), safe_float(parts[1])


def is_valid_poi(category, poi_name, poi_type):
    text = f"{poi_name or ''} {poi_type or ''}".strip()
    pattern = BAD_PATTERNS.get(category)
    return not (pattern and re.search(pattern, text))


def amap_place_around(key, location_lonlat, keywords, radius_m, page, offset):
    params = {
        "key": key,
        "location": location_lonlat,
        "radius": radius_m,
        "keywords": keywords,
        "page": page,
        "offset": offset,
        "extensions": "base",
        "sortrule": "distance",
    }
    response = requests.get(AMAP_AROUND_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def amap_route_distance_m(key, origin_lonlat, dest_lonlat, mode="driving", retry=3):
    url = AMAP_DRIVING_URL if mode == "driving" else AMAP_WALKING_URL
    params = {
        "key": key,
        "origin": origin_lonlat,
        "destination": dest_lonlat,
        "extensions": "base",
    }
    if mode == "driving":
        params["strategy"] = 0

    for attempt in range(retry):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if str(data.get("status")) != "1":
                time.sleep(attempt + 1)
                continue
            paths = data.get("route", {}).get("paths", [])
            if not paths:
                return None
            return safe_float(paths[0].get("distance"))
        except Exception:
            time.sleep(attempt + 1)
    return None


def effective_distance(distances_m, beta):
    values = [d for d in distances_m if d is not None and not (isinstance(d, float) and math.isnan(d))]
    if not values:
        return None
    x = [-beta * d for d in values]
    m = max(x)
    mean_exp = sum(math.exp(v - m) for v in x) / len(x)
    return float(-(m + math.log(mean_exp)) / beta)


def shannon_diversity(strengths):
    total = sum(strengths)
    if total <= 0:
        return 0.0
    shares = [v / total for v in strengths if v > 0]
    if not shares:
        return 0.0
    return float(-sum(p * math.log(p) for p in shares) / math.log(len(strengths)))


def detect_columns(df):
    col_coord = col_name = col_addr = None
    for c in df.columns:
        cs = str(c)
        if col_coord is None and ("坐标" in cs or "经纬度" in cs or "lon" in cs.lower() or "lng" in cs.lower()):
            col_coord = c
        if col_name is None and ("名称" in cs or "项目" in cs or "小区" in cs or "点位" in cs):
            col_name = c
        if col_addr is None and ("地址" in cs or "位置" in cs):
            col_addr = c
    if col_coord is None:
        raise ValueError("No coordinate column was detected. Expected a column containing lon,lat.")
    return col_coord, col_name, col_addr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="案例列表-坐标更新.xlsx")
    parser.add_argument("--output", default="amap_affordable_housing_indicators.xlsx")
    parser.add_argument("--amap-key", default=os.getenv("AMAP_KEY"))
    parser.add_argument("--radius-m", type=int, default=3000)
    parser.add_argument("--offset", type=int, default=25)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--topk-per-cat", type=int, default=15)
    parser.add_argument("--route-mode", choices=["driving", "walking"], default="driving")
    parser.add_argument("--beta", type=float, default=0.004)
    parser.add_argument("--sleep-sec", type=float, default=0.25)
    parser.add_argument("--route-sleep-sec", type=float, default=0.25)
    parser.add_argument("--route-retry", type=int, default=3)
    args = parser.parse_args()

    if not args.amap_key:
        raise ValueError("AMAP key is required. Set AMAP_KEY or pass --amap-key.")

    df_in = pd.read_excel(args.input)
    col_coord, col_name, col_addr = detect_columns(df_in)

    df = df_in.copy()
    df["housing_name"] = df[col_name].astype(str) if col_name is not None else [f"H{i+1}" for i in range(len(df))]
    df["housing_addr"] = df[col_addr].astype(str) if col_addr is not None else ""
    df["coord_raw"] = df[col_coord]

    parsed = [parse_lonlat(v) for v in df["coord_raw"].tolist()]
    df["lon"] = [v[0] for v in parsed]
    df["lat"] = [v[1] for v in parsed]
    df["coord_ok"] = [v[0] is not None and v[1] is not None for v in parsed]

    df_valid = df[df["coord_ok"]].copy().reset_index(drop=True)
    if df_valid.empty:
        raise ValueError("No valid coordinates were parsed from the input file.")
    df_valid["housing_id"] = df_valid.index.astype(int)

    poi_rows = []
    for _, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="POI candidates"):
        origin = f"{row['lon']},{row['lat']}"
        for category, keywords in KEYWORD_MAP.items():
            page = 1
            fetched_any = False
            while page <= args.max_pages:
                data = None
                for attempt in range(3):
                    try:
                        data = amap_place_around(args.amap_key, origin, keywords, args.radius_m, page, args.offset)
                        break
                    except Exception:
                        time.sleep(attempt + 1)

                if data is None:
                    poi_rows.append({
                        "housing_id": int(row["housing_id"]),
                        "housing_name": row["housing_name"],
                        "housing_addr": row["housing_addr"],
                        "origin_lon": row["lon"],
                        "origin_lat": row["lat"],
                        "category": category,
                        "keywords": keywords,
                        "status": "request_failed",
                    })
                    break

                if str(data.get("status")) != "1":
                    poi_rows.append({
                        "housing_id": int(row["housing_id"]),
                        "housing_name": row["housing_name"],
                        "housing_addr": row["housing_addr"],
                        "origin_lon": row["lon"],
                        "origin_lat": row["lat"],
                        "category": category,
                        "keywords": keywords,
                        "status": f"api_error:{data.get('info', '')}",
                    })
                    break

                pois = data.get("pois", []) or []
                if not pois:
                    if not fetched_any:
                        poi_rows.append({
                            "housing_id": int(row["housing_id"]),
                            "housing_name": row["housing_name"],
                            "housing_addr": row["housing_addr"],
                            "origin_lon": row["lon"],
                            "origin_lat": row["lat"],
                            "category": category,
                            "keywords": keywords,
                            "status": "no_pois",
                        })
                    break

                fetched_any = True
                for p in pois:
                    location = p.get("location", "")
                    poi_lon = poi_lat = None
                    if location and "," in location:
                        try:
                            poi_lon = float(location.split(",")[0])
                            poi_lat = float(location.split(",")[1])
                        except Exception:
                            pass

                    poi_rows.append({
                        "housing_id": int(row["housing_id"]),
                        "housing_name": row["housing_name"],
                        "housing_addr": row["housing_addr"],
                        "origin_lon": row["lon"],
                        "origin_lat": row["lat"],
                        "category": category,
                        "poi_id": p.get("id"),
                        "poi_name": p.get("name"),
                        "poi_type": p.get("type"),
                        "poi_typecode": p.get("typecode"),
                        "poi_lon": poi_lon,
                        "poi_lat": poi_lat,
                        "distance_m_straight": safe_float(p.get("distance")),
                        "kept_after_filter": is_valid_poi(category, p.get("name"), p.get("type")),
                        "status": "ok",
                    })

                if len(pois) < args.offset:
                    break
                page += 1
                time.sleep(args.sleep_sec)
            time.sleep(args.sleep_sec)

    df_poi_all = pd.DataFrame(poi_rows)
    df_poi_ok = df_poi_all[
        (df_poi_all["status"] == "ok")
        & (df_poi_all["kept_after_filter"] == True)
        & (df_poi_all["poi_id"].notna())
    ].copy()

    if not df_poi_ok.empty:
        df_poi_ok["distance_m_straight_num"] = pd.to_numeric(df_poi_ok["distance_m_straight"], errors="coerce")
        df_poi_ok = df_poi_ok.sort_values(["housing_id", "category", "poi_id", "distance_m_straight_num"])
        df_poi_ok = df_poi_ok.drop_duplicates(subset=["housing_id", "category", "poi_id"], keep="first")
        df_poi_ok = df_poi_ok.drop(columns=["distance_m_straight_num"])

    df_poi_ok["distance_m_straight_num"] = pd.to_numeric(df_poi_ok["distance_m_straight"], errors="coerce")
    df_poi_ok = df_poi_ok.sort_values(["housing_id", "category", "distance_m_straight_num"])
    df_topk = df_poi_ok.groupby(["housing_id", "category"], as_index=False, group_keys=False).head(args.topk_per_cat).copy()

    route_rows = []
    for row in tqdm(df_topk.itertuples(index=False), total=len(df_topk), desc="Route distances"):
        origin = f"{row.origin_lon},{row.origin_lat}"
        if pd.isna(row.poi_lon) or pd.isna(row.poi_lat):
            network_distance = None
        else:
            destination = f"{row.poi_lon},{row.poi_lat}"
            network_distance = amap_route_distance_m(args.amap_key, origin, destination, mode=args.route_mode, retry=args.route_retry)

        route_rows.append({
            "housing_id": int(row.housing_id),
            "housing_name": row.housing_name,
            "category": row.category,
            "poi_id": row.poi_id,
            "poi_name": row.poi_name,
            "poi_type": row.poi_type,
            "poi_lon": row.poi_lon,
            "poi_lat": row.poi_lat,
            "distance_m_straight": row.distance_m_straight,
            "distance_m_network": network_distance,
        })
        time.sleep(args.route_sleep_sec)

    df_route = pd.DataFrame(route_rows)

    a_rows = []
    for (housing_id, category), group in df_route.groupby(["housing_id", "category"]):
        dist_list = group["distance_m_network"].tolist()
        a_rows.append({
            "housing_id": int(housing_id),
            "category": category,
            "A_effective_m": effective_distance(dist_list, args.beta),
            "n_routes_used": sum(d is not None for d in dist_list),
        })

    df_a = pd.DataFrame(a_rows)
    a_wide = df_a.pivot_table(index="housing_id", columns="category", values="A_effective_m", aggfunc="first").reset_index()
    for cat in CAT_NAME:
        if cat not in a_wide.columns:
            a_wide[cat] = None

    s_wide = a_wide.copy()
    for cat in CAT_NAME:
        s_wide[cat] = s_wide[cat].apply(
            lambda x: math.exp(-args.beta * float(x)) if x is not None and not (isinstance(x, float) and math.isnan(x)) else 0.0
        )

    cats7 = list(CAT_NAME.keys())
    s_wide["diversity_shannon_0_1"] = s_wide.apply(lambda row: shannon_diversity([float(row[c]) for c in cats7]), axis=1)

    rename_a = {cat: f"A_eff_{CAT_NAME[cat]}_m" for cat in cats7}
    rename_s = {cat: f"S_{CAT_NAME[cat]}" for cat in cats7}
    a_wide = a_wide.rename(columns=rename_a)
    s_wide = s_wide.rename(columns=rename_s)

    df_out = df_valid.merge(a_wide, on="housing_id", how="left").merge(
        s_wide[["housing_id"] + list(rename_s.values()) + ["diversity_shannon_0_1"]],
        on="housing_id",
        how="left",
    )

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        df_valid.to_excel(writer, sheet_name="Input_Clean", index=False)
        df_poi_all.to_excel(writer, sheet_name="POI_All_Raw", index=False)
        df_poi_ok.to_excel(writer, sheet_name="POI_Candidates_3km", index=False)
        df_route.to_excel(writer, sheet_name="OD_RouteDistances", index=False)
        df_a.to_excel(writer, sheet_name="A_Effective_Long", index=False)
        df_out.to_excel(writer, sheet_name="Indicators_Final", index=False)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
