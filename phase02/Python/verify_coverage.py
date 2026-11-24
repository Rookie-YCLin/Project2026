"""Check end-to-end coverage and reconcile exported plot inputs."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

from allocation import METHODS
from calibrate import FOLDERS
from watermark import phase01

def verify(run):
    stats=run/FOLDERS[6]
    plots=run/FOLDERS[7]
    data=run/FOLDERS[5]
    expected={f"QF{q:03d}" for q in range(1,101)}
    matrices=[]
    for m in METHODS:
        with h5py.File(data/f"{m}_jpeg.h5") as raw,h5py.File(data/f"{m}_matlab.h5") as mat:
            assert set(raw["jpeg"])==expected and set(mat["jpeg"])==expected
            for k in expected:
                a=raw["jpeg/"+k]
                b=mat["jpeg/"+k]
                assert a["A"].shape==b["U"].shape==b["V"].shape==(3,12696,4,4)
                assert b["sigma"].shape==(3,12696,4)
                assert b["B0_to_C"].shape==b["A_to_C"].shape==(3,12696,21)
                assert a["final_bits"].shape==a["votes"].shape==(512,)
            matrices.append((m,len(raw["jpeg"])))
    expected_table=pd.read_csv(stats/"all_methods_ber_psnr.csv").dropna(subset="qf").set_index(["method","qf"]).sort_index()
    assert len(expected_table)==800 and expected_table.index.is_unique
    for fig,x,y in [(1,"bit_errors","bit_errors"),(2,"psnr_original_db","psnr_original_db")]:
        T=pd.read_csv(plots/f"Fig{fig:02d}.csv").set_index(["method","qf"]).sort_index()
        assert T.index.equals(expected_table.index)
        np.testing.assert_allclose(T[x],expected_table[y],rtol=1e-14,atol=1e-14)
    T=pd.read_csv(plots/"Fig18.csv").set_index(["method","qf"]).sort_index()
    baseline=expected_table.xs("G0")["bit_errors"]
    for m in METHODS[1:]:
        np.testing.assert_array_equal(T.xs(m).error_count_minus_G0,expected_table.xs(m).bit_errors-baseline)
    for fig,metric in [(8,"D_F"),(10,"nuclear_delta"),(11,"nuclear_abs")]:
        T=pd.read_csv(plots/f"Fig{fig:02d}.csv").set_index(["method","condition"]).sort_index()
        for method in METHODS:
            s=pd.read_csv(stats/f"{method}_matrix_stats.csv")
            x=s.query("comparison == 'B0_to_C' and scope == 'All' and metric == @metric").set_index("condition").sort_index()
            np.testing.assert_allclose(T.xs(method)["mean"],x["mean"],atol=1e-13,rtol=1e-14)
    original=run/FOLDERS[0]/"frozen_parameters.json"
    sha=(original.parent/"frozen_parameters.sha256").read_text().split()[0]
    assert phase01.sha(original)==sha
    with h5py.File(run/FOLDERS[4]/"common.h5") as h:
        assert h["original/A"].shape==(3,12696,4,4)
    output=dict(all_methods=dict(matrices),total_conditions=len(expected_table),qf_1_through_100=True,
                figure_data_compared=[1,2,8,10,11,18],frozen_hash=sha)
    (run/FOLDERS[9]/"end_to_end_coverage.json").write_text(
        json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(output)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    verify(p.parse_args().run)
