"""Validate the accelerator's objective and every branch against the MATLAB fixtures."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from image_experiment import feature
from procrustes_batch import solve_batch

def check(out):
    cases=json.loads((out/"cases.json").read_text(encoding="utf-8"))["cases"]
    m=pd.read_csv(out/"matlab_cases.csv").set_index("id")
    timings=[]
    errors=[]
    bounds=[]
    for j in range(4):
        x=[q for q in cases if q["j"]==j]
        ids=[q["id"] for q in x]
        S=np.asarray([q["sigma"] for q in x])
        W=np.divide(S,S[:,:1],out=np.zeros_like(S),where=S[:,:1]>0)**2
        C=np.asarray([q["c"] for q in x])
        phi=np.asarray([q["phi"] for q in x])
        start=perf_counter()
        R,d=solve_batch(W,C,j,phi)
        timings.append(perf_counter()-start)
        assert np.max(abs(feature(C,R[:,:,j])-phi))<=1e-8
        reference=m.loc[ids,["F1","F2","F3","F4"]].to_numpy()
        err=abs(reference-d["branch_primal"])/np.maximum(1,abs(reference))
        differences=np.sum((R-np.eye(4))**2*W[:,None,:],axis=(1,2))-m.loc[ids,"D_P"].to_numpy()
        assert np.max(abs(differences))<=1e-8
        errors.append(float(err.max()))
        bounds.append(float(np.max((d["dual"]-d["primal"])/np.maximum(1,W.sum(axis=1)))))
    assert max(errors)<=1e-8
    data=dict(passed=True,fixtures=len(cases),max_branch_error=max(errors),
              max_global_primal_dual_gap=max(bounds),seconds=sum(timings))
    (out/"batch_gate.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    print(data,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    check(p.parse_args().out)
