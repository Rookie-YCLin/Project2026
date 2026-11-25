"""驗收完整 QF 資料，整理摘要、CSV 分割索引與本機資料雜湊。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pandas as pd

from image_experiment import CALIB,DATA,LABELS,STATS,phase01
from jpeg_experiment import json_file

GATE="01_數學驗證"
FIGURES="06_MATLAB圖表"
REPORT="07_研究報告"
PUBLISH="08_獨立審查"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_index(paths,root):
    return [dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=phase01.sha(p))
            for p in sorted(paths)]


def split(source,destination,limit=499999):
    """保留 CSV 標頭、行結尾、欄位文字精度與原始資料順序。"""
    destination.mkdir(parents=True,exist_ok=False)
    files=[]
    n=0
    with source.open("rb") as f:
        header=f.readline()
        names=next(csv.reader([header.decode("utf-8-sig")]))
        col=names.index("qf") if "qf" in names else None
        stream=None
        try:
            for line in f:
                assert len(line)+len(header)<=limit
                if stream is None or stream.tell()+len(line)>limit:
                    if stream: stream.close()
                    path=destination/f"part_{len(files)+1:03d}.csv"
                    stream=path.open("xb")
                    stream.write(header)
                    files.append(dict(path=path.name,row_start=n+1,row_count=0,qf_min=None,qf_max=None))
                stream.write(line)
                n+=1
                row=files[-1]
                row["row_count"]+=1
                if col is not None:
                    value=next(csv.reader([line.decode("utf-8")]))[col]
                    if value not in ("","NaN"):
                        q=int(value)
                        row["qf_min"]=q if row["qf_min"] is None else min(row["qf_min"],q)
                        row["qf_max"]=q if row["qf_max"] is None else max(row["qf_max"],q)
        finally:
            if stream: stream.close()
    digest=hashlib.sha256(header)
    for row in files:
        path=destination/row["path"]
        row.update(bytes=path.stat().st_size,sha256=phase01.sha(path))
        with path.open("rb") as f:
            assert f.readline()==header
            for line in f:
                digest.update(line)
    pin=phase01.sha(source)
    assert digest.hexdigest()==pin
    return dict(original_filename=source.name,bytes=source.stat().st_size,sha256=pin,
                row_count=n,merged_sha256=digest.hexdigest(),parts=files)


def audit(run):
    freeze=read_json(run/CALIB/"frozen.json")
    assert set(freeze["methods"])==set(LABELS)
    gates=read_json(run/GATE/"stress_gate"/"gate.json")
    assert gates["passed"]
    sample=pd.read_csv(run/GATE/"frozen_p_matlab_audit.csv")
    assert sample.method.nunique()==8 and len(sample)>=8*60
    assert sample[["branch_value_difference","matlab_gap","python_gap","matlab_max_angle_error"]].max().max()<=1e-8
    data=run/DATA
    stats=run/STATS
    by_method=[]
    by_qf=[]
    for label in LABELS:
        folder=data/label
        method=freeze["methods"][label]
        png=folder/"unattacked.png"
        assert phase01.sha(png)==method["png_sha256"]
        result=pd.read_csv(stats/f"{label}_ber_psnr.csv",float_precision="round_trip")
        assert len(result)==101 and result.qf.dropna().tolist()==list(range(1,101))
        assert result.block_count.eq(38088).all() and result.bit_count.eq(512).all()
        assert result.classification.eq(method["classification"]).all()
        assert int(result.final_errors.iloc[0])==method["errors"]
        assert float(result.psnr_original_db.iloc[0])==method["psnr"]
        checks=pd.read_csv(stats/f"{label}_checks_matlab.csv")
        assert checks.qf.tolist()==list(range(101))
        with h5py.File(folder/"attacks.h5","r") as h:
            assert len(h["jpeg"])==100
            for quality in range(1,101):
                q=f"QF{quality:03d}"
                group=h["jpeg"][q]
                jpeg=folder/"JPEG"/f"{q}.jpg"
                assert group.attrs["complete"] and group["matrices"].shape==(3,12696,4,4)
                assert all(group[key].shape==(3,12696,4,4) for key in ("U","V"))
                assert group["sigma"].shape==(3,12696,4)
                assert phase01.sha(jpeg)==group.attrs["jpeg_sha256"]
                assert json.loads(group.attrs["jpeg_metadata"])["sampling_factors"] == [
                    [1,1,1,0],[2,1,1,1],[3,1,1,1]]
                by_qf.append(dict(method=label,qf=quality,hdf5=(folder/"attacks.h5").relative_to(run).as_posix(),
                    dataset=group.name,matlab_hdf5=(folder/"matlab_checks"/f"part{(quality-1)//10+1:02d}.h5").relative_to(run).as_posix(),
                    jpeg=jpeg.relative_to(run).as_posix(),jpeg_bytes=jpeg.stat().st_size,jpeg_sha256=group.attrs["jpeg_sha256"]))
        x=result.iloc[1:]
        by_method.append(dict(method=label,classification=method["classification"],q_rad=method["q_rad"],
            q_deg=method["q_degrees"],png_psnr=method["psnr"],png_final_errors=method["errors"],
            png_block_ber=float(result.block_ber.iloc[0]),mean_ber=float(x.ber.mean()),
            mean_low20_ber=float(x.loc[x.qf<=20,"ber"].mean()),zero_qf=int(x.final_errors.eq(0).sum()),
            qf_worsen_with_higher_quality=int((x.ber.diff()>0).sum()),
            cross_platform_bit_disagreements=int(checks.bit_disagreement.sum()),
            unexplained_disagreements=int(checks.unexplained_bit_disagreement.sum()),
            seconds_embed_and_write_png=method["seconds_embed_png"],seconds_png_blind=method["seconds_blind"],
            seconds_jpeg_blind_total=float(x.seconds_blind.sum()),
            seconds_jpeg_encode_decode_total=float(x.seconds_jpeg_encode_decode.sum())))
    for j in range(1,25):
        for ext in ("fig","png","svg","pdf","csv"):
            f=run/FIGURES/f"Fig{j:02d}.{ext}"
            assert f.stat().st_size>100
    return pd.DataFrame(by_method),gates,pd.DataFrame(by_qf),sample


def overview(run,phase02,phase01_root=None):
    methods,gate,index,sample=audit(run)
    report=run/REPORT
    report.mkdir(exist_ok=True)
    methods.to_csv(report/"method_summary.csv",index=False)
    index.to_csv(report/"qf_data_index.csv",index=False)
    f=methods[methods.classification=="formal"].sort_values("mean_ber")
    e=methods[methods.classification=="exploratory"]
    ref=pd.read_csv(phase02/"07_統計分析"/"all_methods_ber_psnr.csv")
    ref_means=ref[ref.condition!="unattacked"].groupby("method").ber.mean().sort_values()
    math_csv=pd.read_csv(run/GATE/"stress_gate"/"matlab_cases.csv")
    matrix=pd.concat([pd.read_csv(run/STATS/f"{m}_matrix_matlab.csv") for m in LABELS],ignore_index=True)
    diag=matrix[(matrix.scope=="All")&(matrix.comparison=="embedding")]
    png_matrix=matrix[(matrix.scope=="All")&(matrix.qf==0)&(matrix.metric=="D_F")]
    pvals=png_matrix.pivot(index="method",columns="comparison",values="mean")
    pvals.to_csv(report/"unattacked_frobenius_by_transition.csv")
    thresholds=diag[diag.metric.isin(["embed_success","reselected","B_angle_target_error_success","certificate_gap"])]
    thresholds.pivot(index="method",columns="metric",values="mean").to_csv(report/"embedding_diagnostics.csv")
    relation=pd.concat([pd.read_csv(run/STATS/f"{m}_relation_matlab.csv") for m in LABELS],ignore_index=True)
    sweep=relation[(relation.scope=="All")&(relation.rank_class==0)&(relation.qf>0)]
    rho=sweep.groupby("method").pearson_gap_angle.agg(["min","median","max"])
    rho.to_csv(report/"gap_angle_correlation_by_method.csv")
    rank=relation[(relation.scope=="All")&(relation.rank_class>0)&(relation.qf.isin([0,10,50,90]))].copy()
    rank.to_csv(report/"rank_and_recovery_selected_conditions.csv",index=False)
    paired=math_csv[math_csv.category.str.startswith("platelet")]
    matched=paired[paired.D_G.notna()]
    ran=paired.D_R>1e-12
    reduction=(1-paired.loc[ran,"D_P"]/paired.loc[ran,"D_R"]).median()
    f_comp="沒有符合正式校準門檻的第三階段組別。"
    if len(f):
        lead=f.iloc[0]
        f_comp=(f"正式組最低的 QF 1～100 平均最終 BER 出現在 {lead.method}，為 {lead.mean_ber:.3%}。"
                f"第二階段同一品質範圍最低值為 {ref_means.iloc[0]:.3%}（{ref_means.index[0]}）。")
    lines=[
        "# Project2026 第三階段研究報告","",
        "## 研究問題","",
        "本文以相同血小板影像檢驗奇異向量的自然穩定性、正交限制、矩陣嵌入失真及 JPEG 後盲式提取之間的關係。"
        "對每一組方向，本文分別比較單欄旋轉 S、Givens 旋轉 G、最短正交旋轉 R 與受限正交 Procrustes P。","",
        "## 數學驗證","",
        f"MATLAB 與 Python 獨立驗證 {gate['case_count']:,} 組，其中 {gate['phase01_fixtures']:,} 組取自正式影像。"
        f"P 的正規化平方失真在兩套實作間最大差為 {gate['independent_objective_errors']['D_P']:.3g}。"
        "指定反例的未分支對偶上界為 22。行列式正分支的可行及對偶目標皆為 21.5，對應平方 Frobenius 失真 17。","",
        f"凍結後從 P 的八組抽取 {len(sample)} 個實際角度目標，再由 MATLAB 獨立重算四分支。"
        f"與 Python 的正規化分支跡目標最大差為 {sample.branch_value_difference.max():.3g}。","",
        f"實際影像數學案例共 {len(paired)} 個相同目標角度的配對，其中 {len(matched)} 個具有可行 G 平面。"
        f"在 R 正規化平方失真大於 10⁻¹² 的 {ran.sum()} 個配對，P 相對 R 的失真改善中位數為 {reduction:.3%}。"
        "P 保持因子正交與奇異值，重根子空間內用重新 SVD 找到的個別方向仍受選基影響。"
        "S 的單欄操作之後會重新分解 B。","",
        "## 影像與校準","",
        f"每組均經凍結與完整 100 個品質條件驗收。正式校準 {len(f)} 組，探索性校準 {len(e)} 組。{f_comp}","",
        "| 方法 | 校準 | 步長（度） | PNG PSNR（dB） | PNG 錯誤/512 | QF 1～100 平均 BER | 零 BER 的 QF 數 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in methods.itertuples():
        lines.append(f"| {r.method} | {'正式' if r.classification=='formal' else '探索'} | "
                     f"{r.q_deg:.6g} | {r.png_psnr:.6f} | {r.png_final_errors} | {r.mean_ber:.4%} | {r.zero_qf} |")
    lines+=["","## 自然擾動與可用性",""]
    if phase01_root is not None:
        natural=pd.read_csv(phase01_root/"05_統計分析"/"component_rankings_for_report.csv")
        natural=natural[(natural.scope=="All")&natural.family.isin(["theta_u","theta_v"])].copy()
        natural["vector"]=natural.family.str[-1].str.upper()+natural.j.astype(str)
        natural=natural[["vector","mean_across_qf"]].sort_values("mean_across_qf")
        natural=natural.rename(columns={"mean_across_qf":"phase01_mean_angle_deg"})
        formal=f.assign(vector=f.method.str[1:]).groupby("vector").mean_ber.min()
        angles=natural.merge(formal.rename("best_phase03_formal_mean_ber"),how="left",on="vector")
        angles.to_csv(report/"phase01_orientation_comparison.csv",index=False)
        v=angles.iloc[0]
        lines.append(f"第一階段自然角度擾動最小的成分為 {v.vector}，全 QF 合併通道平均 {v.phase01_mean_angle_deg:.4f} 度。"
                     f"第三階段有正式組可供比較的方向共有 {formal.size} 種。"
                     "逐方向資料表同列第一階段自然擾動與第三階段對應正式組的最低平均 BER。")
        lines.append("")
    error=thresholds[thresholds.metric=="B_angle_target_error_success"].copy()
    error["family"]=error.method.str[0]
    by_family=error.groupby("family")["mean"].mean()
    lines.append("理論 B 重新 SVD 後的目標方向絕對誤差，先對已嵌入區塊平均，再對各家族八種方向平均。"
                 +"、".join(f"{k} 為 {x:.4g} rad" for k,x in by_family.items())+"。")
    if len(f)>0:
        chosen=f.iloc[0]
        vals=pvals.loc[chosen.method]
        lines.append(f"正式組平均 BER 最低的 {chosen.method} 在 A→B、B→B₀、A→B₀ 的平均平方 Frobenius 失真依序為 "
                     f"{vals.A_to_B:.6g}、{vals.B_to_B0:.6g}、{vals.A_to_B0:.6g}。"
                     "三階段量值可用來定位重建前後的改變。兩段平方量與總平方量還相差誤差交叉項。")
    lines+=["","## 矩陣擾動與判決","",
            f"全方法中有 {(methods.qf_worsen_with_higher_quality>0).sum()} 組在某些相鄰 QF 提高時出現最終 BER 上升。"
            "這些變動保留在原始曲線。整體平均 BER 以 100 個離散 QF 同權計算。","",
            "PNG 與各 QF 的區塊級方向、間距、奇異值、位元與矩陣失真分別保存在 HDF5。"
            f"關係表保留 Y、U、V 與合併通道的逐條件 Pearson 相關及光譜分類。"
            f"各方法 100 個 QF 的合併通道 Pearson 相關中位數介於 {rho['median'].min():.4f} 和 {rho['median'].max():.4f}。"
            "秩類別的分母、區塊 BER 與方向偏移均在同一表保留，零筆類別的統計以缺值標示。","",
            f"Python 與 MATLAB 逐區塊判決相差 {methods.cross_platform_bit_disagreements.sum():,} 次，"
            f"其中 {methods.unexplained_disagreements.sum():,} 次未由精確重根分類或數值近界解釋。"
            "正式盲式 BER 以研究設定所指定的 Python 提取流程為準。","",
            "## 解讀限制","",
            "本文使用單張影像與單一浮水印種子。重複嵌入與影像區塊結構會造成觀察值相依。"
            "相關係數描述這一張影像的量測，BER 差異同時包含失真、合法中心、正交補全、逆轉換與 JPEG 的作用。"
            "探索組因 PNG 起始條件不同，解讀時需參考各組的 PNG PSNR 與錯誤數。",""]
    (report/"Phase03_Report_zh-TW.md").write_text("\n".join(lines),encoding="utf-8")


def package(run,output):
    assert not output.exists()
    output.mkdir(parents=True)
    items=[]
    for rel in (STATS,FIGURES,GATE,GATE+"/stress_gate",REPORT):
        for source in sorted((run/rel).glob("*.csv")):
            path=output/rel/source.name
            path.parent.mkdir(exist_ok=True,parents=True)
            if source.stat().st_size<500000:
                shutil.copy2(source,path)
                continue
            relative=(Path(rel)/source.stem).as_posix()
            result=split(source,output/relative)
            result.update(destination=relative,source_path=source.relative_to(run).as_posix())
            items.append(result)
    for source in sorted((run/FIGURES).glob("*.svg")):
        shutil.copy2(source,output/FIGURES/source.name)
    for rel in (CALIB,GATE,GATE+"/stress_gate",REPORT,"02_研究規格"):
        target=output/rel
        target.mkdir(parents=True,exist_ok=True)
        for f in (run/rel).iterdir():
            if f.is_file() and f.suffix in (".json",".sha256",".md",".log") and f.stat().st_size<500000:
                shutil.copy2(f,target/f.name)
    for label in LABELS:
        target=output/CALIB/label
        target.mkdir(exist_ok=True)
        for name in ("candidates.csv","selection.json"):
            shutil.copy2(run/CALIB/label/name,target/name)
    json_file(output/"SPLIT_INDEX.json",items)
    raw=[p for p in run.rglob("*") if p.is_file() and PUBLISH not in p.relative_to(run).parts]
    records=file_index(raw,run)
    index=run/PUBLISH
    index.mkdir(exist_ok=True)
    json_file(index/"local_files.json",records)
    large=[r for r in records if r["path"].endswith(".h5")]
    json_file(output/"LOCAL_HDF5_INDEX.json",large)
    retained=list(output.rglob("*"))
    json_file(output/"PUBLISHED_FILES.json",file_index((p for p in retained if p.is_file()),output))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("phase02",type=Path)
    p.add_argument("--phase01",type=Path)
    p.add_argument("--publish-to",type=Path)
    args=p.parse_args()
    overview(args.run,args.phase02,args.phase01)
    if args.publish_to:
        package(args.run,args.publish_to)
