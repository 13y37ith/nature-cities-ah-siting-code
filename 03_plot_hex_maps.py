import argparse
import math
import os
import urllib.request

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from sklearn.neighbors import NearestNeighbors

TYPE_COLORS = {
    "Type A: Resource-Balanced": "#004873",
    "Type B: Transit-Oriented": "#F08228",
    "Type C: Culture-Oriented": "#009300",
    "Type D: Resource-Poor": "#CB0505",
}


def gcj02_to_wgs84(lon, lat):
    pi = math.pi
    a = 6378245.0
    ee = 0.00669342162296594323

    def out_of_china(x, y):
        return not (73.66 < x < 135.05 and 3.86 < y < 53.55)

    def transform_lat(x, y):
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * pi) + 40.0 * math.sin(y / 3.0 * pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y / 12.0 * pi) + 320.0 * math.sin(y * pi / 30.0)) * 2.0 / 3.0
        return ret

    def transform_lon(x, y):
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * pi) + 40.0 * math.sin(x / 3.0 * pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * pi) + 300.0 * math.sin(x / 30.0 * pi)) * 2.0 / 3.0
        return ret

    if out_of_china(lon, lat):
        return lon, lat

    dlat = transform_lat(lon - 105.0, lat - 35.0)
    dlon = transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlon = (dlon * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglon = lon + dlon
    return lon * 2 - mglon, lat * 2 - mglat


def make_full_hex_grid(boundary_geom, edge_length_m):
    xmin, ymin, xmax, ymax = boundary_geom.bounds
    a = edge_length_m
    hex_h = math.sqrt(3) * a
    dx = 1.5 * a
    dy = hex_h

    polys = []
    col = 0
    x = xmin - 2 * a
    while x < xmax + 2 * a:
        y_offset = 0 if col % 2 == 0 else hex_h / 2
        y = ymin - 2 * hex_h
        while y < ymax + 2 * hex_h:
            cy = y + y_offset
            coords = [
                (x + a, cy),
                (x + a / 2, cy + hex_h / 2),
                (x - a / 2, cy + hex_h / 2),
                (x - a, cy),
                (x - a / 2, cy - hex_h / 2),
                (x + a / 2, cy - hex_h / 2),
            ]
            poly = Polygon(coords)
            if boundary_geom.contains(poly.centroid):
                polys.append(poly)
            y += dy
        x += dx
        col += 1

    gdf_hex = gpd.GeoDataFrame({"geometry": polys}, crs="EPSG:3857")
    gdf_hex["hex_id"] = np.arange(len(gdf_hex))
    return gdf_hex


def short_label(text):
    if str(text).startswith("Type A"):
        return "A"
    if str(text).startswith("Type B"):
        return "B"
    if str(text).startswith("Type C"):
        return "C"
    if str(text).startswith("Type D"):
        return "D"
    return str(text)


def base_hex_background(ax, hex_plot):
    base = hex_plot[hex_plot["n_points"].isna()].copy()
    if len(base):
        base.plot(ax=ax, color="#d9d9d9", edgecolor="white", linewidth=0.22)
    ax.set_axis_off()


def save_type_figure(path, hex_plot):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    base_hex_background(ax, hex_plot)
    hit = hex_plot.dropna(subset=["cluster_type"]).copy()
    for ctype, color in TYPE_COLORS.items():
        sub = hit[hit["cluster_type"] == ctype]
        if len(sub):
            sub.plot(ax=ax, color=color, edgecolor="white", linewidth=0.22)

    ax.set_title("Type", fontsize=15, pad=10)
    legend_ax = fig.add_axes([0.84, 0.18, 0.12, 0.62])
    legend_ax.axis("off")
    handles = [mpatches.Patch(color=color, label=short_label(name)) for name, color in TYPE_COLORS.items()]
    legend_ax.legend(handles=handles, title="Type", loc="center", frameon=True, fontsize=11, title_fontsize=12)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_continuous_figure(path, hex_plot, col, title, cmap="viridis", diverging=False, cbar_label="Value"):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    base_hex_background(ax, hex_plot)
    hit = hex_plot.dropna(subset=[col]).copy()
    if not len(hit):
        ax.set_title(title, fontsize=15, pad=10)
        fig.savefig(path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        return

    if diverging:
        vmax = np.nanmax(np.abs(hit[col]))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        hit.plot(ax=ax, column=col, cmap="RdBu_r", edgecolor="white", linewidth=0.22, vmin=-vmax, vmax=vmax)
        norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    else:
        hit.plot(ax=ax, column=col, cmap=cmap, edgecolor="white", linewidth=0.22)
        vmin = np.nanmin(hit[col])
        vmax = np.nanmax(hit[col])
        if not np.isfinite(vmin):
            vmin = 0
        if not np.isfinite(vmax):
            vmax = 1
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    ax.set_title(title, fontsize=15, pad=10)
    cax = fig.add_axes([0.88, 0.18, 0.03, 0.62])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(cbar_label, fontsize=11)
    cb.ax.tick_params(labelsize=10)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results_lqi_pipeline/Final_Results.xlsx")
    parser.add_argument("--sheet", default="Data_All")
    parser.add_argument("--output-dir", default="hex_map_outputs")
    parser.add_argument("--target-hex-count", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_excel(args.input, sheet_name=args.sheet).copy()
    score_col = "lqi_score" if "lqi_score" in df.columns else "maturity_score"
    pred_col = "lqi_pred_gnn" if "lqi_pred_gnn" in df.columns else "maturity_pred_gnn"

    if "residual_gnn" not in df.columns and pred_col in df.columns:
        df["residual_gnn"] = df[score_col] - df[pred_col]

    if "spillover_x_gnn" not in df.columns and "residual_gnn" in df.columns:
        coords_tmp = df[["lon", "lat"]].values
        knn_tmp = NearestNeighbors(n_neighbors=min(8, len(df))).fit(coords_tmp)
        neighbors = knn_tmp.kneighbors(coords_tmp, return_distance=False)
        resid = df["residual_gnn"].to_numpy()
        spill = np.zeros(len(df))
        for i, row in enumerate(neighbors):
            nb = row[1:] if len(row) > 1 else row
            spill[i] = np.nanmean(resid[nb]) if len(nb) else np.nan
        df["spillover_x_gnn"] = spill

    wgs = df.apply(lambda r: gcj02_to_wgs84(r["lon"], r["lat"]), axis=1)
    df["lon_wgs84"] = [x[0] for x in wgs]
    df["lat_wgs84"] = [x[1] for x in wgs]

    gdf_points = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["lon_wgs84"], df["lat_wgs84"]), crs="EPSG:4326").to_crs("EPSG:3857")
    gdf_sh = ox.geocode_to_gdf("Shanghai, China")
    gdf_sh = gdf_sh.set_crs("EPSG:4326") if gdf_sh.crs is None else gdf_sh.to_crs("EPSG:4326")

    land_zip = os.path.join(args.output_dir, "ne_10m_land.zip")
    if not os.path.exists(land_zip):
        urllib.request.urlretrieve("https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip", land_zip)

    land_gdf = gpd.read_file(f"zip://{land_zip}")
    land_gdf = land_gdf.set_crs("EPSG:4326") if land_gdf.crs is None else land_gdf.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf_sh.total_bounds
    land_clip = land_gdf.cx[minx - 1.0:maxx + 1.0, miny - 1.0:maxy + 1.0].copy()

    geom_shanghai_admin = gdf_sh.to_crs("EPSG:3857").unary_union
    geom_land_mask = unary_union(land_clip.to_crs("EPSG:3857").geometry)
    geom_shanghai_land = geom_shanghai_admin.intersection(geom_land_mask)

    components = list(geom_shanghai_land.geoms) if isinstance(geom_shanghai_land, MultiPolygon) else [geom_shanghai_land]
    components = [g for g in components if g.area > 1e6]

    seed_wgs = {
        "mainland": (121.47, 31.23),
        "chongming": (121.40, 31.62),
        "changxing": (121.80, 31.43),
        "hengsha": (121.85, 31.33),
    }
    seed_gdf = gpd.GeoDataFrame({"name": list(seed_wgs.keys())}, geometry=[Point(v[0], v[1]) for v in seed_wgs.values()], crs="EPSG:4326").to_crs("EPSG:3857")

    chosen = {}
    for name, seed_pt in zip(seed_gdf["name"], seed_gdf.geometry):
        dists = [geom.distance(seed_pt) for geom in components]
        chosen[name] = components[int(np.argmin(dists))]

    target_geom = unary_union([chosen["mainland"], chosen["chongming"], chosen["changxing"], chosen["hengsha"]])

    candidate_edges = [4600, 4400, 4200, 4000, 3800, 3600, 3400, 3200, 3000, 2800, 2600]
    best_hex = None
    best_diff = 10**9
    for edge in candidate_edges:
        gtest = make_full_hex_grid(target_geom, edge)
        diff = abs(len(gtest) - args.target_hex_count)
        if diff < best_diff:
            best_diff = diff
            best_hex = gtest

    hex_gdf = best_hex.copy()
    join = gpd.sjoin(gdf_points, hex_gdf[["hex_id", "geometry"]], how="left", predicate="within")
    if join["hex_id"].isna().any():
        miss = join["hex_id"].isna()
        pts_miss = join.loc[miss].drop(columns=["index_right", "hex_id"], errors="ignore")
        near = gpd.sjoin_nearest(pts_miss, hex_gdf[["hex_id", "geometry"]], how="left")
        join.loc[miss, "hex_id"] = near["hex_id"].values

    agg = join.groupby("hex_id").agg(
        n_points=("lon_wgs84", "size"),
        mean_lqi=(score_col, "mean"),
        mean_predicted_lqi=(pred_col, "mean") if pred_col in join.columns else (score_col, "mean"),
        mean_gnn_residual=("residual_gnn", "mean") if "residual_gnn" in join.columns else (score_col, "mean"),
        mean_latent_dim1=("gnn_x", "mean") if "gnn_x" in join.columns else (score_col, "mean"),
        mean_latent_dim2=("gnn_y", "mean") if "gnn_y" in join.columns else (score_col, "mean"),
    ).reset_index()

    dominant_type = join.dropna(subset=["hex_id", "cluster_type"]).groupby(["hex_id", "cluster_type"]).size().reset_index(name="count").sort_values(["hex_id", "count"], ascending=[True, False]).drop_duplicates("hex_id")[["hex_id", "cluster_type"]]
    hex_plot = hex_gdf.merge(agg, on="hex_id", how="left").merge(dominant_type, on="hex_id", how="left")

    save_type_figure(os.path.join(args.output_dir, "Fig_Type.png"), hex_plot)
    save_continuous_figure(os.path.join(args.output_dir, "Fig_Calculated_LQI.png"), hex_plot, "mean_lqi", "Calculated LQI", cbar_label="LQI")
    save_continuous_figure(os.path.join(args.output_dir, "Fig_Predicted_LQI_GNN.png"), hex_plot, "mean_predicted_lqi", "Predicted LQI by GNN", cbar_label="LQI")
    save_continuous_figure(os.path.join(args.output_dir, "Fig_GNN_Residual.png"), hex_plot, "mean_gnn_residual", "GNN Residual", cmap="RdBu_r", diverging=True, cbar_label="Residual")
    save_continuous_figure(os.path.join(args.output_dir, "Fig_Latent_Dim1.png"), hex_plot, "mean_latent_dim1", "Latent dim 1", cbar_label="Dim 1")
    save_continuous_figure(os.path.join(args.output_dir, "Fig_Latent_Dim2.png"), hex_plot, "mean_latent_dim2", "Latent dim 2", cbar_label="Dim 2")

    print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
