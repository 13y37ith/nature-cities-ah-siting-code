import argparse
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    GNN_AVAILABLE = True
except Exception:
    GNN_AVAILABLE = False


TYPE_A = "Type A: Resource-Balanced"
TYPE_B = "Type B: Transit-Oriented"
TYPE_C = "Type C: Culture-Oriented"
TYPE_D = "Type D: Resource-Poor"
TYPE_ORDER = [TYPE_A, TYPE_B, TYPE_C, TYPE_D]

FEATURES = [
    "A_eff_Commerce_m",
    "A_eff_Culture_m",
    "A_eff_GreenSpace_m",
    "A_eff_Healthcare_m",
    "A_eff_RailTransit_m",
    "A_eff_SurfaceTransport_m",
    "A_eff_Sports_m",
    "diversity_shannon_0_1",
]
DIST_COLS = [c for c in FEATURES if c != "diversity_shannon_0_1"]
LATEX_LABEL = {
    "A_eff_Commerce_m": r"$A_{\mathrm{comm}}$",
    "A_eff_Culture_m": r"$A_{\mathrm{cult}}$",
    "A_eff_GreenSpace_m": r"$A_{\mathrm{green}}$",
    "A_eff_Healthcare_m": r"$A_{\mathrm{health}}$",
    "A_eff_RailTransit_m": r"$A_{\mathrm{rail}}$",
    "A_eff_SurfaceTransport_m": r"$A_{\mathrm{road}}$",
    "A_eff_Sports_m": r"$A_{\mathrm{sport}}$",
    "diversity_shannon_0_1": r"$D_{\mathrm{Sh}}$",
}
PALETTE = {
    TYPE_A: "#004873",
    TYPE_B: "#F08228",
    TYPE_C: "#009300",
    TYPE_D: "#CB0505",
}


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def calc_metrics(y_true, y_pred, label):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    ratio = np.where(np.abs(y_true) > 1e-12, y_pred / y_true, np.nan)
    ratio = ratio[np.isfinite(ratio)]
    mean_ratio = float(np.mean(ratio)) if len(ratio) else np.nan
    std_ratio = float(np.std(ratio, ddof=1)) if len(ratio) > 1 else np.nan
    cov_ratio = float(std_ratio / mean_ratio) if len(ratio) > 1 and abs(mean_ratio) > 1e-12 else np.nan
    return {
        "Dataset": label,
        "N": len(y_true),
        "R2": r2,
        "MeanRatio_pred_true": mean_ratio,
        "CoVRatio_pred_true": cov_ratio,
        "RMSE": rmse,
        "MAE": mae,
    }


def load_data(filepath, sheet_name=None, aff_n=148, com_n=12):
    xls = pd.ExcelFile(filepath)
    df = xls.parse(xls.sheet_names[-1] if sheet_name is None else sheet_name).copy()

    if "source_type" not in df.columns:
        df["source_type"] = "Unknown"
        df.iloc[:aff_n, df.columns.get_loc("source_type")] = "Affordable Housing"
        df.iloc[aff_n:aff_n + com_n, df.columns.get_loc("source_type")] = "Commodity Housing"

    X = df[FEATURES].copy()
    mask_aff = df["source_type"].eq("Affordable Housing")
    for c in DIST_COLS:
        base = X.loc[mask_aff, c].replace(0, np.nan)
        penalty = base.quantile(0.95) * 1.2
        X[c] = X[c].replace(0, penalty)

    X_log = X.copy()
    for c in DIST_COLS:
        X_log[c] = np.log1p(X_log[c])

    return df, X, X_log


def compute_lqi_scores(X_log, mask_aff):
    aff_min = X_log.loc[mask_aff, FEATURES].min()
    aff_max = X_log.loc[mask_aff, FEATURES].max()
    X_norm = (X_log[FEATURES] - aff_min) / (aff_max - aff_min)
    X_norm = X_norm.clip(0, 1)

    X_good = X_norm.copy()
    for c in DIST_COLS:
        X_good[c] = 1 - X_good[c]

    X_good = X_good.clip(lower=1e-6, upper=1.0)
    lqi_score = (np.prod(X_good.values, axis=1) ** (1 / len(FEATURES))) * 100
    return X_norm, X_good, lqi_score


def assign_type_names(df, X_raw, mask_aff):
    profile = X_raw.loc[mask_aff].copy()
    profile["cluster_id"] = df.loc[mask_aff, "cluster_id"].values
    profile = profile.groupby("cluster_id").mean()

    prof_norm = (profile - profile.min()) / (profile.max() - profile.min())
    prof_good = prof_norm.copy()
    for c in DIST_COLS:
        prof_good[c] = 1 - prof_good[c]

    overall = prof_good.mean(axis=1)
    id_a = overall.idxmax()
    id_d = overall.idxmin()
    remaining = [i for i in prof_good.index if i not in [id_a, id_d]]

    id_b = prof_good.loc[remaining, "A_eff_RailTransit_m"].idxmax()
    remaining = [i for i in remaining if i != id_b]
    id_c = prof_good.loc[remaining, "A_eff_Culture_m"].idxmax()

    return {
        id_a: TYPE_A,
        id_b: TYPE_B,
        id_c: TYPE_C,
        id_d: TYPE_D,
    }


def run_xgboost(df_all, X_log, mask_aff, outdir, random_state):
    df_aff = df_all.loc[mask_aff].copy()
    X_model = X_log.loc[mask_aff, FEATURES].rename(columns=LATEX_LABEL)
    y = df_aff["lqi_score"].values

    train_idx, test_idx = train_test_split(
        np.arange(len(df_aff)),
        test_size=0.30,
        random_state=random_state,
        stratify=df_aff["cluster_type"],
    )

    Xtr = X_model.iloc[train_idx].copy()
    Xte = X_model.iloc[test_idx].copy()
    ytr = y[train_idx]
    yte = y[test_idx]

    params = {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 4,
        "reg_alpha": 0.10,
        "reg_lambda": 1.5,
        "gamma": 0.02,
        "objective": "reg:squarederror",
        "random_state": random_state,
        "tree_method": "hist",
    }

    model = xgb.XGBRegressor(**params)
    model.fit(Xtr, ytr)

    df_all["lqi_pred_xgb"] = np.nan
    df_all.loc[mask_aff, "lqi_pred_xgb"] = model.predict(X_model)
    df_all["split_aff_xgb"] = "Not_Affordable"
    df_all.loc[df_aff.index[train_idx], "split_aff_xgb"] = "Train_Affordable"
    df_all.loc[df_aff.index[test_idx], "split_aff_xgb"] = "Test_Affordable"
    df_all["residual_xgb"] = df_all["lqi_score"] - df_all["lqi_pred_xgb"]
    df_all["abs_residual_xgb"] = np.abs(df_all["residual_xgb"])

    metrics_df = pd.DataFrame([
        calc_metrics(ytr, model.predict(Xtr), "Train (Affordable)"),
        calc_metrics(yte, model.predict(Xte), "Test (Affordable)"),
        calc_metrics(y, model.predict(X_model), "Total (Affordable)"),
    ])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_model)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_model, show=False)
    savefig(os.path.join(outdir, "Fig_SHAP_Beeswarm.png"))

    shap_global_df = pd.DataFrame({
        "feature": list(X_model.columns),
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    return df_all, metrics_df, shap_global_df


def run_pca(df_all, X_log, outdir):
    X_pca = StandardScaler().fit_transform(X_log[FEATURES])
    coords = PCA(n_components=2, random_state=42).fit_transform(X_pca)
    df_all["PC1"] = coords[:, 0]
    df_all["PC2"] = coords[:, 1]

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df_all, x="PC1", y="PC2", hue="cluster_type", style="source_type", palette=PALETTE, s=80)
    plt.title("PCA embedding")
    savefig(os.path.join(outdir, "Fig_PCA_Embedding.png"))
    return df_all


def run_gnn(df_all, X_std, mask_aff, outdir, random_state, knn_k=8):
    if not GNN_AVAILABLE:
        return df_all, pd.DataFrame()

    coords = df_all[["lon", "lat"]].to_numpy()
    knn = NearestNeighbors(n_neighbors=min(knn_k, len(df_all))).fit(coords)
    graph = knn.kneighbors_graph(coords, mode="connectivity").tocoo()
    mask_no_self = graph.row != graph.col
    rows = graph.row[mask_no_self]
    cols = graph.col[mask_no_self]

    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    x_t = torch.tensor(X_std[FEATURES].values, dtype=torch.float)
    y_t = torch.tensor(df_all["lqi_score"].values, dtype=torch.float).view(-1, 1)

    aff_pos = np.where(mask_aff)[0]
    train_idx, test_idx = train_test_split(
        aff_pos,
        test_size=0.30,
        random_state=random_state,
        stratify=df_all.loc[mask_aff, "cluster_type"],
    )
    train_idx_t = torch.tensor(train_idx, dtype=torch.long)

    class GNNReg(nn.Module):
        def __init__(self, in_dim, hidden_dim=24, dropout=0.05):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, 2)
            self.out = nn.Linear(2, 1)
            self.dropout = dropout

        def forward(self, x, edge_index):
            h = F.relu(self.conv1(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.conv2(h, edge_index)
            y = self.out(h)
            return y, h

    model = GNNReg(len(FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    for _ in range(400):
        model.train()
        optimizer.zero_grad()
        pred, emb = model(x_t, edge_index)
        mse_loss = F.mse_loss(pred[train_idx_t], y_t[train_idx_t])
        smooth_loss = ((emb[edge_index[0]] - emb[edge_index[1]]) ** 2).sum(dim=1).mean()
        loss = mse_loss + 0.005 * smooth_loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred, emb = model(x_t, edge_index)
        pred_np = pred.cpu().numpy().flatten()
        emb_np = emb.cpu().numpy()

    df_all["lqi_pred_gnn"] = pred_np
    df_all["residual_gnn"] = df_all["lqi_score"] - df_all["lqi_pred_gnn"]
    df_all["abs_residual_gnn"] = np.abs(df_all["residual_gnn"])
    df_all["gnn_x"] = emb_np[:, 0]
    df_all["gnn_y"] = emb_np[:, 1]
    df_all["split_aff_gnn"] = "Not_Affordable"
    df_all.loc[train_idx, "split_aff_gnn"] = "Train_Affordable"
    df_all.loc[test_idx, "split_aff_gnn"] = "Test_Affordable"

    metrics_df = pd.DataFrame([
        calc_metrics(df_all.loc[train_idx, "lqi_score"], df_all.loc[train_idx, "lqi_pred_gnn"], "Train (Affordable)"),
        calc_metrics(df_all.loc[test_idx, "lqi_score"], df_all.loc[test_idx, "lqi_pred_gnn"], "Test (Affordable)"),
        calc_metrics(df_all.loc[mask_aff, "lqi_score"], df_all.loc[mask_aff, "lqi_pred_gnn"], "Total (Affordable)"),
    ])

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df_all, x="gnn_x", y="gnn_y", hue="cluster_type", style="source_type", palette=PALETTE, s=80)
    plt.title("GraphSAGE embedding")
    savefig(os.path.join(outdir, "Fig_GNN_Embedding.png"))

    plt.figure(figsize=(7, 7))
    plt.scatter(df_all.loc[mask_aff, "lqi_pred_gnn"], df_all.loc[mask_aff, "lqi_score"], alpha=0.7)
    low = min(df_all.loc[mask_aff, "lqi_score"].min(), df_all.loc[mask_aff, "lqi_pred_gnn"].min())
    high = max(df_all.loc[mask_aff, "lqi_score"].max(), df_all.loc[mask_aff, "lqi_pred_gnn"].max())
    plt.plot([low, high], [low, high], "k--", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.title("GNN fit")
    savefig(os.path.join(outdir, "Fig_GNN_Fit.png"))

    return df_all, metrics_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="amap_affordable_housing_indicators.xlsx")
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--output-dir", default="results_lqi_pipeline")
    parser.add_argument("--affordable-n", type=int, default=148)
    parser.add_argument("--commodity-n", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--run-gnn", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["figure.dpi"] = 300
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("whitegrid")

    df_all, X_raw, X_log = load_data(args.input, args.sheet, args.affordable_n, args.commodity_n)
    mask_aff = df_all["source_type"].eq("Affordable Housing").values

    scaler = StandardScaler()
    X_std = pd.DataFrame(scaler.fit_transform(X_log[FEATURES]), columns=FEATURES, index=df_all.index)

    X_std_aff = X_std.loc[mask_aff].copy()
    mcd = MinCovDet(random_state=args.random_state, support_fraction=0.9).fit(X_std_aff)
    md = mcd.mahalanobis(X_std_aff)
    outlier_index = X_std_aff.index[int(np.argmax(md))]
    X_fit = X_std_aff.drop(index=outlier_index)

    kmeans = KMeans(n_clusters=4, n_init=100, random_state=args.random_state)
    kmeans.fit(X_fit)
    df_all["cluster_id"] = kmeans.predict(X_std)
    df_all["cluster_type"] = df_all["cluster_id"].map(assign_type_names(df_all, X_raw, mask_aff))

    _, _, lqi_score = compute_lqi_scores(X_log, mask_aff)
    df_all["lqi_score"] = lqi_score

    df_all, xgb_metrics_df, shap_global_df = run_xgboost(df_all, X_log, mask_aff, args.output_dir, args.random_state)
    df_all = run_pca(df_all, X_log, args.output_dir)

    gnn_metrics_df = pd.DataFrame()
    if args.run_gnn:
        df_all, gnn_metrics_df = run_gnn(df_all, X_std, mask_aff, args.output_dir, args.random_state)

    metrics_list = [xgb_metrics_df.assign(Model="XGBoost")]
    if not gnn_metrics_df.empty:
        metrics_list.append(gnn_metrics_df.assign(Model="GNN"))
    metrics_summary_df = pd.concat(metrics_list, axis=0, ignore_index=True)

    out_xlsx = os.path.join(args.output_dir, "Final_Results.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="Data_All", index=False)
        df_all.loc[mask_aff].to_excel(writer, sheet_name="Data_Affordable", index=False)
        shap_global_df.to_excel(writer, sheet_name="SHAP_Global_MeanAbs", index=False)
        metrics_summary_df.to_excel(writer, sheet_name="Metrics_Summary", index=False)

    print(f"Saved: {out_xlsx}")


if __name__ == "__main__":
    main()
