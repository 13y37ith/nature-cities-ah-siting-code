import argparse

import pandas as pd

TYPE_ORDER = [
    "Type A: Resource-Balanced",
    "Type B: Transit-Oriented",
    "Type C: Culture-Oriented",
    "Type D: Resource-Poor",
]

VARS = [
    "A_eff_Commerce_m",
    "A_eff_Culture_m",
    "A_eff_GreenSpace_m",
    "A_eff_Healthcare_m",
    "A_eff_RailTransit_m",
    "A_eff_SurfaceTransport_m",
    "A_eff_Sports_m",
    "diversity_shannon_0_1",
]

SHAP_ORDER = [
    "A_eff_Healthcare_m",
    "diversity_shannon_0_1",
    "A_eff_RailTransit_m",
    "A_eff_Culture_m",
    "A_eff_Sports_m",
    "A_eff_Commerce_m",
    "A_eff_SurfaceTransport_m",
    "A_eff_GreenSpace_m",
]


def calc_stats(series):
    desc = series.describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    return pd.Series({
        "Min": desc["min"],
        "P25": desc["25%"],
        "Median (P50)": desc["50%"],
        "P75": desc["75%"],
        "P90": desc["90%"],
        "Max": desc["max"],
        "Mean": desc["mean"],
        "Std": desc["std"],
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results_lqi_pipeline/Final_Results.xlsx")
    parser.add_argument("--sheet", default="Data_All")
    parser.add_argument("--output", default="MedianBoundary_Rule_Analysis.xlsx")
    args = parser.parse_args()

    df = pd.read_excel(args.input, sheet_name=args.sheet)
    score_col = "lqi_score" if "lqi_score" in df.columns else "maturity_score"

    if "source_type" in df.columns:
        df = df[df["source_type"].astype(str).str.strip() == "Affordable Housing"].copy()
    df = df[df["cluster_type"].isin(TYPE_ORDER)].copy()

    higher_better = {score_col: True, "diversity_shannon_0_1": True}
    for var in VARS:
        if var not in higher_better:
            higher_better[var] = False

    stats_data = []
    for var in [score_col] + VARS:
        s = calc_stats(df[var])
        s.name = var
        s["Group"] = "All Affordable"
        stats_data.append(s)

    for t in TYPE_ORDER:
        dft = df[df["cluster_type"] == t]
        for var in [score_col] + VARS:
            s = calc_stats(dft[var])
            s.name = var
            s["Group"] = t
            stats_data.append(s)

    stats_df = pd.DataFrame(stats_data).reset_index().rename(columns={"index": "Variable"})
    stats_df = stats_df[["Group", "Variable", "Min", "P25", "Median (P50)", "P75", "P90", "Max", "Mean", "Std"]]

    cutpoints = []
    threshold_dict = {}
    for var in [score_col] + VARS:
        medians = {t: df[df["cluster_type"] == t][var].median() for t in TYPE_ORDER}
        mA = medians[TYPE_ORDER[0]]
        mB = medians[TYPE_ORDER[1]]
        mC = medians[TYPE_ORDER[2]]
        mD = medians[TYPE_ORDER[3]]

        cut_dc = (mD + mC) / 2
        cut_cb = (mC + mB) / 2
        cut_ba = (mB + mA) / 2
        cutpoints.append({
            "Variable": var,
            "Median_D": mD,
            "Median_C": mC,
            "Median_B": mB,
            "Median_A": mA,
            "Cut_D_C": cut_dc,
            "Cut_C_B (Threshold)": cut_cb,
            "Cut_B_A": cut_ba,
        })
        threshold_dict[var] = cut_cb

    cutpoints_df = pd.DataFrame(cutpoints)

    def check_pass(row, selected_vars):
        if row[score_col] < threshold_dict[score_col]:
            return False
        for var in selected_vars:
            if higher_better[var]:
                if row[var] < threshold_dict[var]:
                    return False
            else:
                if row[var] > threshold_dict[var]:
                    return False
        return True

    rule_results = []
    current_vars = []
    for i in range(len(SHAP_ORDER) + 1):
        if i > 0:
            current_vars.append(SHAP_ORDER[i - 1])

        df["Pass_Current_Rule"] = df.apply(lambda row: check_pass(row, current_vars), axis=1)
        result = {
            "Num_Vars": f"LQI + {i}",
            "Vars_Added": " + ".join([
                "Diversity" if v == "diversity_shannon_0_1" else v.replace("A_eff_", "").replace("_m", "")
                for v in current_vars
            ]) if current_vars else "Only LQI",
        }

        for t in TYPE_ORDER:
            sub = df[df["cluster_type"] == t]
            result[f"PassRate_{t.split(':')[0]}"] = sub["Pass_Current_Rule"].mean()

        passed = df[df["Pass_Current_Rule"]].copy()
        if len(passed):
            result["AB_Purity"] = passed["cluster_type"].isin(TYPE_ORDER[:2]).mean()
            result["Total_Passed_Count"] = len(passed)
        else:
            result["AB_Purity"] = 0.0
            result["Total_Passed_Count"] = 0

        rule_results.append(result)

    rule_df = pd.DataFrame(rule_results)

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        stats_df.to_excel(writer, sheet_name="Statistics", index=False)
        cutpoints_df.to_excel(writer, sheet_name="Cutpoints", index=False)
        rule_df.to_excel(writer, sheet_name="Rule_Analysis", index=False)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
