"""用同一份逐塊角度，獨立重算整個 Pearson、Spearman 及分箱資料表。"""
import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from image_pipeline import json_file
from numerics import correlations, bins
from validate_run import BIN_KEYS, CORR_KEYS, fields, read_summary


def verify(run):
    validation = run/"08_交叉驗證"
    python_dir = run/"07_Python驗證"
    ref_dir = run/"05_統計分析"
    specs = {
        "correlations": (CORR_KEYS, ("count","pearson","spearman"), correlations),
        "gap_bins": (BIN_KEYS, ("lower","upper","count","median","p10","p90"), bins),
    }
    paths = {name: python_dir/f"{name}_from_matlab_blocks.csv" for name in specs}
    file = run/"04_逐區塊擾動數據"/"perturbation_matlab.h5"
    with (h5py.File(file) as h5, paths["correlations"].open("x",newline="",encoding="utf-8") as corr,
          paths["gap_bins"].open("x",newline="",encoding="utf-8") as gap):
        writers = dict(correlations=csv.writer(corr),gap_bins=csv.writer(gap))
        for name,(keys, cols, _) in specs.items():
            writers[name].writerow((*keys,*cols))
        for cond, group in (("identity","identity"),*[(q,"jpeg/"+q) for q in sorted(h5["jpeg"])]):
            m=fields(h5[f"{group}/metrics"])
            for name,(_,_,calc) in specs.items():
                writers[name].writerows(calc(m,cond))
    summary = {}
    for name,(keys,cols,_) in specs.items():
        a,b = (read_summary(path,keys) for path in (paths[name],ref_dir/f"{name}_matlab.csv"))
        assert a.index.equals(b.index)
        x,y = a[list(cols)].to_numpy(dtype=float),b[list(cols)].to_numpy(dtype=float)
        assert np.array_equal(np.isfinite(x),np.isfinite(y))
        assert np.array_equal(a["count"],b["count"])
        finite = np.isfinite(x)
        e = np.abs(x[finite]-y[finite])
        err=float(np.max(e/np.maximum(1,np.maximum(np.abs(x[finite]),np.abs(y[finite]))),initial=0))
        assert err<=2e-12, (name,err)
        independently_computed = read_summary(python_dir/f"{name}_python.csv",keys)
        raw = a.join(independently_computed,lsuffix="_mat_source",rsuffix="_py_source").reset_index()
        for col in [c for c in cols if c not in {"count","lower","upper"}]:
            raw["abs_error_"+col]=(raw[col+"_mat_source"]-raw[col+"_py_source"]).abs()
        errors=[k for k in raw if k.startswith("abs_error")]
        report = raw.melt(id_vars=list(keys)+["count_mat_source"],value_vars=errors,
                          var_name="field",value_name="absolute_difference")
        report.nlargest(50,"absolute_difference").to_csv(validation/f"{name}_top_independent_differences.csv",index=False)
        summary[name]=dict(rows=len(a),max_scaled_same_input=err,
                           max_abs_same_input=float(e.max(initial=0)))
    json_file(validation/"association_reaggregation.json",summary)
    print(summary)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    verify(p.parse_args().run)
