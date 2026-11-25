"""Check the numerical gate across the two independently written implementations."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def run(out):
    p=json.loads((out/"python_summary.json").read_text(encoding="utf-8"))
    m=json.loads((out/"matlab_summary.json").read_text(encoding="utf-8"))
    assert p["passed"] and m["passed"] and p["case_count"]==m["case_count"]>=1000
    a=pd.read_csv(out/"python_cases.csv")
    b=pd.read_csv(out/"matlab_cases.csv")
    categories=a.category.value_counts().to_dict()
    assert (a.id==b.id).all() and (a.planes==b.planes).all()
    keys=["D_S","D_G","D_R","D_P","F1","F2","F3","F4"]
    maxima={}
    for name in keys:
        x,y=a[name].to_numpy(),b[name].to_numpy()
        assert np.array_equal(np.isnan(x),np.isnan(y))
        maxima[name]=float(np.nanmax(abs(x-y)/np.maximum(1,np.maximum(abs(x),abs(y)))))
        assert maxima[name]<=1e-8,(name,maxima[name])
    result=dict(passed=True,case_count=len(a),categories=categories,independent_objective_errors=maxima,
                phase01_fixtures=sum(n for k,n in categories.items() if k.startswith("platelet")),
                example=p["example"],limits={
                "objective_agreement":1e-8,"angle_rad":1e-8,"orthogonality":1e-10})
    (out/"gate.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.out)
