"""Collect observed Phase02 results into auditable tables for the report."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

METHODS=("G0","S1","S2","S3","S4","S5","S6","S7")


def observations(root):
    s=root/"07_統計分析"
    M=root/"06_嵌入與擾動數據"
    frames={}
    for family in ("ber_psnr","matrix_stats","allocation_stats","decision_stats","spectral_stats","association"):
        frames[family]=pd.concat([pd.read_csv(s/f"{m}_{family}.csv").assign(method=m) for m in METHODS],ignore_index=True)
    outcomes=frames["ber_psnr"]
    attacked=outcomes[outcomes.qf.notna()].copy()
    attacked["qf"]=attacked.qf.astype(int)
    assert len(attacked)==800 and len(outcomes)==808
    assert all(set(t.qf)==set(range(1,101)) for _,t in attacked.groupby("method"))
    assert (outcomes[outcomes.qf.isna()].bit_errors==0).all()
    assert outcomes[outcomes.qf.isna()].psnr_original_db.between(39.9,40.1).all()
    outcomes.to_csv(s/"all_methods_ber_psnr.csv",index=False,float_format="%.17g")
    baseline=attacked[attacked.method=="G0"].set_index("qf").bit_errors
    alloc=frames["allocation_stats"].query("scope == 'All'").pivot(index="method",columns="metric",values="mean")
    mat=frames["matrix_stats"].query("scope == 'All'")
    q=frames["decision_stats"].query("scope == 'All'")
    f00=mat.query("condition == 'unattacked' and metric == 'D_F'").pivot(
             index="method",columns="comparison",values="mean")
    defense=mat.query("comparison == 'B0_to_C' and metric in ['D_F','nuclear_delta','nuclear_abs']").copy()
    defense["qf"]=defense.condition.str[2:].astype(int)
    F=defense.pivot(index=["method","qf"],columns="metric",values="mean")
    signed=defense.query("metric == 'nuclear_delta'")
    mv=mat.query("comparison == 'B0_to_C' and metric in ['abs_delta_1','abs_delta_2','abs_delta_3','abs_delta_4']")
    mvp=mv.pivot(index=["method","condition"],columns="metric",values="mean")
    thresholds=[]
    for method,t in attacked.groupby("method",sort=False):
        t=t.sort_values("qf")
        eb=t.bit_errors.to_numpy()
        ec=eb-baseline.loc[t.qf].to_numpy()
        nuclear=F.loc[method,"nuclear_delta"].reindex(t.qf)
        df=F.loc[method,"D_F"].reindex(t.qf)
        related=pd.DataFrame(dict(qf=t.qf.to_numpy(),bit_errors=eb,block_ber=t.block_ber.to_numpy(),
            nuclear=nuclear.to_numpy())).set_index("qf")
        corr=related.corr(method="spearman")
        thresholds.append(dict(method=method,mean_final_errors=float(eb.mean()),total_errors=int(eb.sum()),
            zero_error_qf=int((eb==0).sum()),worst_errors=int(eb.max()),
            win_qf=int((ec<0).sum()),tie_qf=int((ec==0).sum()),loss_qf=int((ec>0).sum()),
            win_qualities=[int(x) for x in t.qf[ec<0]],mean_block_ber=float(t.block_ber.mean()),
            max_win_margin=int(ec.min()),qf_max_win=int(t.qf.iloc[ec.argmin()]),
            qf1=int(eb[0]), qf10=int(eb[9]),qf30=int(eb[29]),qf50=int(eb[49]),
            errors_qf79_80_81=[int(x) for x in eb[78:81]],qf90=int(eb[89]),qf100=int(eb[-1]),
            d_f_qf79_80_81=[float(x) for x in df.to_numpy()[78:81]],
            nuclear_negative_qf=int((nuclear<0).sum()),
            sigma4_min_qf=int((mvp.loc[method].idxmin(axis=1)=="abs_delta_4").sum()),
            nuclear_ber_spearman=float(corr.loc["nuclear","bit_errors"]),block_ber_spearman=float(corr.loc["block_ber","bit_errors"])))
    summary=pd.DataFrame(thresholds)
    summary.to_csv(s/"cross_qf_method_summary.csv",index=False,float_format="%.17g")
    calibration=pd.read_csv(root/"02_步長校準"/"selected.csv").set_index("method")
    counts={m:len(pd.read_csv(root/"02_步長校準"/m/"candidates.csv")) for m in METHODS}
    validation={}
    for m in METHODS:
        c=pd.read_csv(root/"10_交叉驗證"/f"{m}_matlab_checks.csv")
        v=pd.read_csv(root/"09_Python驗證"/f"{m}_independent_checks.csv")
        assert len(c)==100 and len(v)==203
        validation[m]={"matlab":c.select_dtypes("number").max().to_dict(),
                       "python":v.select_dtypes("number").max().to_dict(),
                       "block_bit_mismatch_sum":int(c.block_bit_mismatches.sum())}
    e0=q.query("condition == 'unattacked' and metric in ['e0','abs_e','e0_s1','e0_s2']").pivot(
        index="method",columns="metric",values="mean")
    margin=q.query("condition != 'unattacked' and metric in ['target_margin','nearest_boundary','abs_e']").copy()
    margin["qf"]=margin.condition.str[2:].astype(int)
    margin=margin[margin.qf.isin([1,50,80,100])]
    payload=dict(
        calibration=calibration.reset_index().to_dict("records"),candidate_counts=counts,
        ber=summary.to_dict("records"),alloc=json.loads(alloc.to_json(orient="index")),
        squared_embedding=json.loads(f00.to_json(orient="index")),
        center_png=json.loads(e0.to_json(orient="index")),selected_center_offsets=margin.to_dict("records"),
        validation=validation)
    target=root/"11_研究報告"/"report_evidence.json"
    target.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(summary.drop(columns=["win_qualities"]).to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    observations(p.parse_args().run)
