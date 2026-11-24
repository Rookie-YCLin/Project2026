"""用 MATLAB 的正式資料表產生逐項報告的可核對數字。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def clean(a):
    if isinstance(a, pd.DataFrame):
        return json.loads(a.to_json(orient="records", double_precision=14))
    if isinstance(a, pd.Series):
        return json.loads(a.to_json(double_precision=14))
    if isinstance(a, (np.int64, np.int32)):
        return int(a)
    if isinstance(a, np.floating):
        return float(a)
    raise TypeError(type(a))


def collect(run):
    d = run/"05_統計分析"
    py = run/"07_Python驗證"
    stats = pd.read_csv(d/"statistics_matlab.csv")
    data = stats.query("condition != 'identity'").copy()
    data["QF"] = data.condition.str[2:].astype(int)
    means = data.pivot(index="QF",columns=["scope","metric"],values="mean")
    rep = [1, 10, 25, 50, 75, 90, 99, 100]
    all_mean = means["All"]
    dfl = means.xs("D_F",level=1,axis=1)[["Y","U","V","All"]]
    significant = np.diff(dfl.All) > 32*np.finfo(float).eps*np.maximum(1,dfl.All.to_numpy()[:-1])
    df_changes = pd.DataFrame({"from_QF":dfl.index[:-1],"to_QF":dfl.index[1:],
                               "delta":np.diff(dfl.All)})
    f_ranges = {}
    for c in dfl.columns:
        y = dfl[c]
        f_ranges[c] = dict(argmax=y.idxmax(), maximum=y.max(), argmin=y.idxmin(), minimum=y.min(),
                           up=int(np.count_nonzero(y.diff()>1e-12)), down=int(np.count_nonzero(y.diff()<-1e-12)))
    dfl.to_csv(d/"matrix_means_for_report.csv")
    ranking = []
    for c in ("Y","U","V","All"):
        t = means[c]
        for family in ("abs_delta_sigma","theta_u","theta_v"):
            names = [f"{family}_{j}" for j in range(1,5)]
            group = t[names]
            values = group.to_numpy()
            largest = np.isclose(values,values.max(axis=1,keepdims=True),rtol=1e-12,atol=1e-12)
            smallest = np.isclose(values,values.min(axis=1,keepdims=True),rtol=1e-12,atol=1e-12)
            for j,key in enumerate(names):
                ranking.append([c,family,j+1,group[key].mean(),int(largest[:,j].sum()),int(smallest[:,j].sum()),
                                int((group[key].diff()>1e-12).sum()),int((group[key].diff()<-1e-12).sum())])
    ranking = pd.DataFrame(ranking,columns=["scope","family","j","mean_across_qf","max_qf_count",
                                            "min_qf_count","up_steps","down_steps"])
    ranking.to_csv(d/"component_rankings_for_report.csv",index=False)
    corr = pd.read_csv(d/"correlations_matlab.csv").query("condition != 'identity'")
    corr_report = corr.groupby(["scope","j","side","feature"],as_index=False).agg(
        pearson_median=("pearson","median"),spearman_median=("spearman","median"),
        pearson_min=("pearson","min"),pearson_max=("pearson","max"),
        spearman_min=("spearman","min"),spearman_max=("spearman","max"),
        valid_qf=("spearman","count"),min_count=("count","min"),max_count=("count","max"))
    corr_report.to_csv(d/"correlation_across_qf_for_report.csv",index=False)
    identifiable = pd.read_csv(d/"identifiable_angles_matlab.csv")
    identifiable = identifiable.query("condition != 'identity'").copy()
    summary_for_sensitivity = identifiable.query("scope == 'All'").groupby(["side","j"],as_index=False).agg(
        mean_angle=("mean","mean"), mean_n=("count","mean"), min_n=("count","min"),max_n=("count","max"))
    paired = pd.read_csv(d/"paired_angles_and_signs_matlab.csv")
    per_j = paired.query("scope == 'All'").groupby("j",as_index=False).agg(
        paired_Pearson_median=("pearson_left_right","median"),
        paired_Pearson_min=("pearson_left_right","min"),
        paired_Pearson_max=("pearson_left_right","max"),
        mean_absolute_angle_difference=("mean_absolute_paired_angle_difference","mean"),
        positive=("positive","sum"),negative=("negative","sum"),near_zero=("near_zero","sum"),
        total=("count","sum"))
    diag = pd.read_csv(d/"diagnostics_matlab.csv")
    dg = diag[(diag.scope=="All") & (diag.condition.isin([f"QF{q:03d}" for q in rep]))]
    diag_keys = ["original_near_rank_deficient","jpeg_near_rank_deficient","original_zero_matrix",
                 "jpeg_zero_matrix","original_near_repeated_3","jpeg_near_repeated_3",
                 "original_near_repeated_4","jpeg_near_repeated_4"]
    dgt = dg[dg.flag.isin(diag_keys)].pivot(index="condition",columns="flag",values="true_count").reset_index()
    pstats = pd.read_csv(py/"statistics_python.csv")
    cmp = stats.merge(pstats,on=["condition","scope","metric"],suffixes=("_matlab","_python"),validate="one_to_one")
    cmp = cmp[cmp.metric.str.startswith(("theta_u","theta_v"))].copy()
    cmp["absolute_mean_difference_deg"] = abs(cmp.mean_matlab-cmp.mean_python)
    cmp = cmp[["condition","scope","metric","mean_matlab","mean_python","absolute_mean_difference_deg"]]
    cmp.to_csv(run/"08_交叉驗證"/"angle_mean_differences_by_condition.csv",index=False)
    corr_focus = corr_report[(corr_report.scope=="All")&
                             corr_report.feature.isin(["gap","gap_norm","perturbation_over_gap"])]
    image_meta = json.loads((run/"10_執行紀錄"/"environment_python.json").read_text(encoding="utf-8"))
    original_tables = image_meta["original"]["quantization_tables"]
    jpeg_meta = json.loads((run/"01_JPEG影像"/"jpeg_metadata.json").read_text(encoding="utf-8"))
    table_changes = []
    for jpeg in jpeg_meta:
        for table, entries in original_tables.items():
            changed = sum(a != b for a,b in zip(entries,jpeg["quantization_tables"][table],strict=True))
            table_changes.append((jpeg["qf"],table,changed))
    table_changes = pd.DataFrame(table_changes,columns=["QF","table_id","entries_different_from_original"])
    table_changes.to_csv(d/"quantization_table_comparison.csv",index=False)
    bundle = dict(frobenius_selected=dfl.loc[rep].reset_index(), frobenius_ranges=f_ranges,
                  channel_largest_count=dfl[["Y","U","V"]].idxmax(axis=1).value_counts(),
                  frobenius_biggest_rises=df_changes[significant].nlargest(10,"delta"),
                  frobenius_biggest_drops=df_changes.nsmallest(10,"delta"),
                  all_qf_component_ranking=ranking.query("scope == 'All'"),
                  qf50_component_ranking_by_scope=means.loc[50].unstack(0).loc[
                      [f"{s}_{j}" for s in ("abs_delta_sigma","theta_u","theta_v","delta_sigma") for j in range(1,5)]
                  ].reset_index(),
                  selected_all_means=all_mean.loc[rep,[f"{s}_{j}" for s in
                      ("abs_delta_sigma","theta_u","theta_v","delta_sigma") for j in range(1,5)]
                      +["D_F","D_sigma","R","nuclear_delta","nuclear_abs"]].reset_index(),
                  signs_and_paired=per_j, correlations_focus=corr_focus, sensitivity=summary_for_sensitivity,
                  diagnostics_selected=dgt, top_angle_mean_error=cmp.nlargest(10,"absolute_mean_difference_deg"),
                  exact_original_quantization_table_matches=table_changes[
                      table_changes.entries_different_from_original==0])
    out = run/"09_實驗報告"/"report_evidence.json"
    out.write_text(json.dumps(bundle,default=clean,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    collect(p.parse_args().run)
