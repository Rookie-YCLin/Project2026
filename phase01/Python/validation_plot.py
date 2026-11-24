"""以 Matplotlib 檢視兩套軟體的 D_F 平均值及數值殘差。"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(root):
    a = pd.read_csv(root/"05_統計分析"/"statistics_matlab.csv")
    b = pd.read_csv(root/"07_Python驗證"/"statistics_python.csv")
    keys = ["condition","scope","metric"]
    s = a.merge(b, on=keys, suffixes=("_matlab","_python"), validate="one_to_one")
    s = s[(s.metric=="D_F") & (s.condition!="identity")].copy()
    s["QF"] = s.condition.str[2:].astype(int)
    s["mean_difference"] = s.mean_python - s.mean_matlab
    directory = root/"07_Python驗證"
    fields = ["QF","scope","mean_matlab","mean_python","mean_difference"]
    s[fields].to_csv(directory/"frobenius_validation_plot.csv",index=False)
    with plt.rc_context({"font.family":"Times New Roman","font.size":10}):
        fig, axes = plt.subplots(2,1,figsize=(6.4,5.0),sharex=True,layout="constrained")
        for scope,color,style in zip(("Y","U","V","All"),("#0D529E","#B94A14","#087559","#7A3D8A"),
                                     ("-","--","-.",":")):
            t = s[s.scope==scope].sort_values("QF")
            axes[0].plot(t.QF,t.mean_python,color=color,linestyle=style,label=scope,linewidth=1.35)
            axes[1].plot(t.QF,t.mean_difference,color=color,linestyle=style,linewidth=1.15)
        axes[0].legend(ncol=4,loc="upper center",bbox_to_anchor=(.5,1.18),frameon=False)
        axes[0].set_ylabel("Mean squared Frobenius norm")
        axes[1].set_ylabel("Python − MATLAB mean")
        axes[1].set_xlabel("JPEG quality factor (QF)")
        count = s.QF.nunique()
        if count == 100:
            axes[1].set_xticks([1,20,40,60,80,100])
        for ax in axes:
            ax.set_xlim(1,100)
            ax.grid(alpha=.23)
        for suffix in ("svg","pdf","png"):
            fig.savefig(directory/f"frobenius_validation.{suffix}",dpi=350,facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    plot(p.parse_args().run)
