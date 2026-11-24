"""Render a Traditional Chinese Phase02 report from verified observations."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

FIGURES=(
"八組方法的最終 BER","原圖參考 PSNR","正式步長及初始 PSNR",
"理論嵌入平方 Frobenius 失真","理論、存檔與實際嵌入失真",
"非負限制、投影及重新選中心","次序反轉與數值近似相等",
"JPEG 對 B₀ 的平方 Frobenius 擾動","JPEG 對四個奇異值的絕對擾動",
"JPEG 核範數平均有號差","JPEG 核範數平均絕對差","QIM 正規化中心偏移",
"QIM 判決裕度與區塊正確率","區塊與 512-bit BER","嵌入前後的相鄰間距",
"S5 中央 Y 區塊四矩陣頻譜","QF 1–30 的 BER","各總和 QIM 組相對 G0 的錯誤位元數")


def runs(nums):
    if not nums:
        return "—"
    b=a=nums[0]
    spans=[]
    for x in nums[1:]:
        if x==b+1:
            b=x
            continue
        spans.append(str(a) if a==b else f"{a}–{b}")
        a=b=x
    spans.append(str(a) if a==b else f"{a}–{b}")
    return "、".join(spans)


def make(run):
    folder=run/"11_研究報告"
    p=json.loads((folder/"report_evidence.json").read_text(encoding="utf-8"))
    stat=run/"07_統計分析"
    archived=pd.read_csv(stat/"all_methods_ber_psnr.csv")
    c={d["method"]:d for d in p["calibration"]}
    b={d["method"]:d for d in p["ber"]}
    alloc=p["alloc"]
    embedding=p["squared_embedding"]
    base=b["G0"]
    best=min(b.values(),key=lambda r:r["mean_final_errors"])
    best_s=min((d for d in b.values() if d["method"]!="G0"),key=lambda d:d["mean_final_errors"])
    easy=min((m for m in embedding if m!="G0"),key=lambda m:embedding[m]["A_to_B"])
    initial=archived[archived.qf.isna()]
    best_p=min(c.values(),key=lambda x:x["psnr_db"])["psnr_db"]
    highest_p=max(c.values(),key=lambda x:x["psnr_db"])["psnr_db"]
    model=json.loads((run/"09_Python驗證"/"matlab_allocation_gate.json").read_text(encoding="utf-8"))["model_checks"]
    model_err=max(x["allocation_max_absolute"] or 0 for x in model.values())
    maxReconstruction=max(r["matlab"]["matlab_reconstruction"] for r in p["validation"].values())
    maxAngle=max(r["python"]["max_identifiable_angle_deg"] for r in p["validation"].values())
    maxAmbig=max(r["python"]["max_unresolved_angle_deg"] for r in p["validation"].values())
    statsErr=max(r["python"]["max_statistic_scaled_same_source"] for r in p["validation"].values())
    nMismatch=sum(r["block_bit_mismatch_sum"] for r in p["validation"].values())
    lines=["# 奇異值修改量分配與 JPEG 浮水印強健性","",
           "## 1. 研究目的","",
           "我在固定的血小板影像上比較七種奇異值修改量分配，並以前人郭飛的 Common/Slow 法為對照。"
           "S1～S7 共用奇異值總和量化與同一盲式提取方式。每組先將未攻擊 8-bit PNG 的完整彩色 PSNR 校準至 40±0.1 dB，"
           "同時要求未攻擊錯誤數為 0/512，再測量各組的全部 JPEG QF 1–100。本文討論同一影像上的配對結果。",
           "","## 2. 矩陣與非負可行性","",
           r"令 $A=U\operatorname{diag}(\sigma)V^\mathsf{T}$。S1～S7 對 $S=\sum_j\sigma_j$ 選取最近的合法中心 $T=kq$，其中 $k\ge0$ 且 $k\bmod 2=b$。同距依 Huang 等原式的邊界規則取較大中心。從收到影像重做 SVD，再以 $\lfloor\widehat S/q+1/2\rfloor\bmod2$ 作逐塊判決。[Huang et al., 2015](https://doi.org/10.1186/s13640-015-0073-6)",
           "",
           r"固定權重時令 $y=T-S$，$x=\sigma+yw$。每個 $w_j>0$ 給出 $y\ge-\sigma_j/w_j$，取交集得到 $T_{\min}=S-\min_{w_j>0}(\sigma_j/w_j)$。中心若落在下界之外，沿同奇偶格點選最近可行中心。S1 的 $x$ 屬於 $x_j\ge0$ 且 $\sum_j x_j=T$ 的單純形。平方距離目標為嚴格凸，KKT 條件給出 $x_j=\max(\sigma_j+\tau,0)$，其中 $\tau$ 使總和等於 $T$。$T=0$ 時 $x=0$。",
           "",
           r"令 $B=U\operatorname{diag}(x)V^\mathsf{T}$。正交不變性給出 $\|B-A\|_F^2=\|x-\sigma\|_2^2$。固定權重的平方失真為 $y^2\|w\|_2^2$。四分量平均的係數為 $1/4$，單一分量為 $1$，兩種比例分配皆為 $\sum_j\sigma_j^2/S^2$。比例分配的等值結論要求同一 $\sigma$ 與同一 $y$；不同可行區及個別校準步長會改變實際的 $y$。使用單純形投影時，直接計算 $\|x-\sigma\|_2^2$。重建按原分量保留 $x_j$ 與 $u_j,v_j$ 的配對。",
           "",
           r"$B_0$ 是 PNG 還原後的矩陣，$C(Q)$ 是該 PNG 經固定 JPEG 條件後的矩陣。$C-A=(B_0-A)+(C-B_0)$，因此平方失真還有 $2\langle B_0-A,C-B_0\rangle_F$。本文分別量測 A→B、B→B₀、A→B₀、B₀→C、A→C。重根即使依序排列仍對應不唯一基底。主統計保留全部排序索引，退化區塊與嚴格次序反轉另有標記。",
           "","## 3. 資料、驗收與正式步長","",
           "全彩 738×1108 原圖與第一階段相同。A 經原始矩陣逐元素核對。每個條件包含 Y、U、V 各 12,696 塊。"
           "三通道共享係數排列和位置對應的加密位元，各 JPEG 均由該方法的正式 PNG 獨立編碼，採 Pillow 4:4:4。"
           "G0 原版分數是 $(3b_1+b_2)/4$，區塊硬判決門檻為 0.5。各方法先逐位元跨位置取分數平均，再套用郭飛原版一維 K-means，最後解開位元排列。",
           "",
           f"共實測 {sum(p['candidate_counts'].values())} 個未攻擊候選。正式參數凍結後才進行 JPEG。八組 PSNR 介於 {best_p:.6f} 與 {highest_p:.6f} dB。",
           "",
           "| 方法 | 正式步長或倍率 | 未攻擊 PSNR | 未攻擊錯誤數 | 未攻擊區塊 BER |",
           "|---|---:|---:|---:|---:|"]
    for m,d in c.items():
        z=initial[initial.method==m].iloc[0]
        param=f"α={d['parameter']:.8f}，d₁={36*d['parameter']:.6f}，d₂={20*d['parameter']:.6f}" if m=="G0" else f"q={d['parameter']:.8f}"
        lines.append(f"| {m} | {param} | {d['psnr_db']:.6f} | 0/512 | {100*z.block_ber:.3f}% |")
    lines+=["","## 4. 正式 JPEG 結果","",
            "表中的平均 BER 以 100 個品質因子等權平均，同時保留下列逐 QF 代表值。最後三欄依相同 QF 的錯誤位元數與 G0 比較。圖 1、2、17、18 展示完整曲線及低 QF 區段。","",
            "| 方法 | 平均 BER | QF1 錯誤 | QF10 | QF30 | QF50 | QF90 | QF100 | 低於 G0 | 等於 G0 | 高於 G0 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m,r in b.items():
        lines.append(f"| {m} | {100*r['mean_final_errors']/512:.3f}% | {r['qf1']} | {r['qf10']} | {r['qf30']} | {r['qf50']} | {r['qf90']} | {r['qf100']} | {r['win_qf']} | {r['tie_qf']} | {r['loss_qf']} |")
    lines+=["",f"全 QF 等權平均最低的是 {best['method']}，平均每個 QF 有 {best['mean_final_errors']:.2f} 個錯誤；"
            f"七種總和量化組中最低的是 {best_s['method']}，每個 QF 平均有 {best_s['mean_final_errors']:.2f} 個錯誤。"
            f"G0 平均為 {base['mean_final_errors']:.2f} 個錯誤。每個 QF 是同一影像的重複條件，這些平均值是描述性比較。",
            "","相對於 G0，出現較少錯誤位元的實測 QF 如下。",""]
    for m,r in b.items():
        if m!="G0":
            peak=f"，最大優勢是在 QF {r['qf_max_win']} 少 {abs(r['max_win_margin'])} 位元" if r["win_qf"] else ""
            lines.append(f"- {m}｜{runs(r['win_qualities'])}{peak}。")
    lines+=["","## 5. 非負限制、局部失真與奇異值擾動","",
            "| 方法 | A→B 平均平方失真 | B→B₀ | A→B₀ | 非負觸發 | 投影 | 改中心 | 嚴格次序反轉 | 負／零／正修改 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def pct(value):
        return "—" if value is None else f"{100*value:.2f}%"
    for m in c:
        z=embedding[m]; d=alloc[m]
        sign="—"
        if m!="G0":
            with h5py.File(run/"06_嵌入與擾動數據"/f"{m}_embed.h5") as h:
                y=h["embedding/y"][:]
                scale=np.maximum(1,np.maximum(h["embedding/S"][:],h["embedding/T"][:]))
                neg=np.mean(y < -64*np.finfo(float).eps*scale)
                pos=np.mean(y > 64*np.finfo(float).eps*scale)
                sign=f"{100*neg:.2f}%／{100*(1-neg-pos):.2f}%／{100*pos:.2f}%"
        lines.append(f"| {m} | {z['A_to_B']:.4f} | {z['B_to_B0']:.4f} | {z['A_to_B0']:.4f} | {pct(d.get('nonnegative_trigger'))} | {pct(d.get('projected'))} | {pct(d.get('reselected'))} | {pct(d['rank_reversal'])} | {sign} |")
    lines+=["",f"在七組的正式步長下，平均理論平方失真最低者為 {easy}。各組的局部失真係數、正式步長與中心重新選擇率一併決定實測失真。"
            f"S1 的正式 q 為 {c['S1']['parameter']:.4f}，S5 為 {c['S5']['parameter']:.4f}。"
            f"正向 S6 和反向 S7 的重新選中心率各為 {pct(alloc['S6']['reselected'])} 與 {pct(alloc['S7']['reselected'])}。"
            "比例向量的平方長度相同，這項關係只控制相同 y 時的理論擾動。",
            "",f"S5 的整體第四奇異值絕對擾動在 {b['S5']['sigma4_min_qf']}/100 個 QF 是同組四個成分中最小者。"
            f"它對應的全 QF 等權 BER 為 {100*b['S5']['mean_final_errors']/512:.3f}%。本階段提取使用四個奇異值的總和，"
            "因此可將第四成分本身的擾動和整體的判決錯誤一起核對。S1～S7 的次序反轉比例保留於圖 7；"
            "由於組別的步長、影像重建誤差與非負觸發一同變動，單張影像的這些比例無法辨認次序反轉的因果作用。",
            "","## 6. 核範數、判決及 QF 80","",
            r"總和量化組的 PNG 中心偏差為 $e_0=S_0-T$，JPEG 核範數改變為 $\delta_Q=S_Q-S_0$，總偏移 $e_Q=e_0+\delta_Q$。"
            r"目標裕度為 $q/2-|e_Q|$，其正值給出目標格點內的充分判決條件。實際判決依整數索引差的奇偶而定，"
            r"偶數格的偏移仍可給出同一位元。距最近邊界的距離為 $q/2-|S_Q-k_Qq|$。G0 另外保存兩個成分各自的中心偏移與原版判決。",
            "",
            "| 方法 | PNG 平均絕對中心偏差 | JPEG 核範數平均為負的 QF 數 | 核範數有號平均與最終錯誤的 Spearman | 區塊 BER 與最終錯誤的 Spearman | QF 79／80／81 錯誤 |",
            "|---|---:|---:|---:|---:|---:|"]
    for m,r in b.items():
        q=p["center_png"][m]["abs_e"]
        errors="／".join(map(str,r["errors_qf79_80_81"]))
        lines.append(f"| {m} | {'—' if q is None else format(q,'.4f')} | {r['nuclear_negative_qf']} | {r['nuclear_ber_spearman']:.3f} | {r['block_ber_spearman']:.3f} | {errors} |")
    lines+=["", "每列相關在同一組方法內對 100 個 QF 的摘要值計算。逐 QF、逐通道的區塊中心偏移與判決關聯另有 MATLAB 資料表。"
            "高階頻譜可能隨嵌入與重新排序改變，中心偏移及最大可用 q 可以解釋部分結果。相同 QF 也同時影響像素變動、區塊特徵及投票分數，未以相關係數辨識因果。",
            "",
            "| 方法 | QF 79 的 B₀→C 平方擾動 | QF 80 | QF 81 | 跨 QF 平均區塊 BER | 跨 QF 平均 512-bit BER |",
            "|---|---:|---:|---:|---:|---:|"]
    for m,r in b.items():
        a,d,e=r["d_f_qf79_80_81"]
        lines.append(f"| {m} | {a:.4f} | {d:.4f} | {e:.4f} | {100*r['mean_block_ber']:.3f}% | {100*r['mean_final_errors']/512:.3f}% |")
    dip=[m for m,r in b.items() if r["d_f_qf79_80_81"][1]<min(r["d_f_qf79_80_81"][0],r["d_f_qf79_80_81"][2])]
    lines+=["",f"以相鄰 QF 的 B₀→C 指標比較，QF 80 低於 QF 79 及 81 的組別為 {'、'.join(dip) or '無'}。"
            "未嵌入原圖的 QF 80 低谷是第一階段的研究動機；此處各 JPEG 的直接輸入是重新建成的含浮水印 PNG。"
            "重複位元平均與 K-means 的作用可從兩種 BER 的實際差異、圖 14 和存檔的 512 個投票值檢查。",
            "","## 7. 驗證與解讀範圍","",
            f"MATLAB 由 A、B、B₀、C 重新計算頁面 SVD、Frobenius 範數、核範數與方向角度。七組非負分配的最大 MATLAB／Python 修改值重算差為 {model_err:.3g}。"
            f"JPEG 矩陣的最大 MATLAB 尺度化 SVD 重建殘差為 {maxReconstruction:.3g}。"
            f"Python 對可辨識個別方向的跨平台最大角度差為 {maxAngle:.3g} 度，其他方向最大為 {maxAmbig:.3g} 度。"
            f"同一 MATLAB 逐塊資料重新彙整的最大尺度化統計差為 {statsErr:.3g}。Python 使用各方法的盲式入口結果作正式 BER；"
            f"MATLAB 從重算 SVD 的區塊判決與正式結果在 800 組中共相差 {nMismatch} 個區塊，逐條件數量見交叉驗證 CSV。",
            "",
            "本文的影像條件為固定單張血小板原圖、同一 payload 與相同 Pillow 色度取樣。"
            "各 QF 重複使用同一組嵌入位置，圖表的分位數描述完整區塊分布。若要把相關性、最有利的分量或步長外推到其他影像，需要新資料及方法固定後的獨立比較。",
            "","## 8. 約三頁正文的素材建議","",
            "- 第一頁放四矩陣定義、可行中心下界及固定 y 的平方失真推導。附簡表列 G0 比例、七組 q 及未攻擊驗收。",
            "- 第二頁放完整 BER、低 QF 曲線與相對 G0 的錯誤差，標出實際勝出的 QF。把比較的主數字限定在同一 PSNR 條件。",
            "- 第三頁放非負觸發、B→B₀ 誤差、核範數與 QIM 判決裕度。用 QF 79–81 及區塊／最終 BER 的差異連結第一階段結果。",
            "","[圖表索引](圖表索引.md)","",
            "執行識別碼為 `"+run.name+"`。",""]
    strata=pd.read_csv(stat/"rank_reversal_strata.csv")
    strata=strata[(strata["scope"]=="All")&(strata.qf==50)]
    reversal=["QF 50 的逐塊組內對照如下。表中使用嵌入時的嚴格反轉旗標分組，再計算 B₀→C 的平方 Frobenius 擾動平均。","",
              "| 方法 | 無反轉塊數 | 無反轉平均 | 反轉塊數 | 反轉平均 |",
              "|---|---:|---:|---:|---:|"]
    for m,t in strata.groupby("method",sort=False):
        a,d=list(t.sort_values("rank_reversal").itertuples(index=False))
        reversal.append(f"| {m} | {a.n} | {a.mean_D_F:.4f} | {d.n} | {d.mean_D_F:.4f} |".replace("nan","—"))
    reversal+=["","完整分層表另存每個 QF、通道及旗標分組的矩陣擾動、核範數與第四成分角度。"
               "分組由嵌入後的頻譜結構形成，也可能反映原始區塊結構差異。",""]
    ranking=["下表列出第四奇異值的平均絕對擾動落在四個成分最低值的 QF 數。"
             "第一階段未嵌入圖在各通道與整體均為 100。在第二階段，基準改為各方法各自的含水印 PNG。","",
             "| 方法 | Y | U | V | 整體 |","|---|---:|---:|---:|---:|"]
    keys=[f"abs_delta_{i}" for i in range(1,5)]
    for m in c:
        t=pd.read_csv(stat/f"{m}_matrix_stats.csv")
        t=t[(t.comparison=="B0_to_C")&t.metric.isin(keys)]
        x=t.pivot(index=["scope","condition"],columns="metric",values="mean")
        count=x["abs_delta_4"].eq(x.min(axis=1)).groupby(level=0).sum()
        ranking.append(f"| {m} | {count['Y']} | {count['U']} | {count['V']} | {count['All']} |")
    ranking+=["",f"以各組正式步長比較，{easy} 的平均理論嵌入失真最低，{best_s['method']} 的全 QF 平均 BER 最低。"
             f"S1 的 q={c['S1']['parameter']:.5f} 為七組中最大。這張圖的結果須連同非負可行區和 PNG 重建誤差解讀。",""]
    at=lines.index("## 6. 核範數、判決及 QF 80")
    lines[at:at]=ranking
    lines[at:at]=reversal
    subset=pd.read_csv(run/"10_交叉驗證"/"representative_subspace_checks.csv")
    vpos=lines.index("## 8. 約三頁正文的素材建議")
    lines[vpos:vpos]=[f"我另外核對全部方法的 B、B₀ 與代表性 JPEG 品質因子 1、50、80、100，共 {len(subset)} 組。"
         f"以相鄰近重根分組比較投影矩陣，最大 Frobenius 差為 {subset.max_projector_error.max():.3g}，"
         f"最大誤差與頻譜分隔尺度界的比值為 {subset.max_bound_ratio.max():.3g}。", ""]
    (folder/"第二階段研究報告.md").write_text("\n".join(lines),encoding="utf-8")
    idx=["# 第二階段圖表索引","","下列 18 張圖均由 MATLAB 產生。PNG 以 350 dpi 匯出，FIG 可編輯。圖中曲線採逐 QF 資料。","",
         "| 圖號 | 內容 | 本機成果 |","|---:|---|---|"]
    for i,t in enumerate(FIGURES,1):
        links=" · ".join(f"[{e.upper()}](../08_MATLAB圖表/Fig{i:02d}.{e})" for e in ("fig","svg","pdf","png","csv"))
        idx.append(f"| {i} | {t} | {links} |")
    (folder/"圖表索引.md").write_text("\n".join(idx)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    make(p.parse_args().run)
