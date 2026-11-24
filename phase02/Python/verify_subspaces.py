"""Record cross-platform singular subspace checks for representative Phase02 conditions."""
import argparse
from pathlib import Path

import h5py
import numpy as np

from allocation import METHODS
from calibrate import FOLDERS,write_csv
from validate_run import geom
from numerics import subspace_errors


def check(run):
    data=run/FOLDERS[5]
    rows=[]
    for method in METHODS:
        with (h5py.File(data/f"{method}_embed.h5") as embed,
              h5py.File(data/f"{method}_jpeg.h5") as jpeg,
              h5py.File(data/f"{method}_matlab.h5") as mat):
            sources=[("B",embed),("B0",embed)]+[(f"jpeg/QF{q:03d}",jpeg) for q in (1,50,80,100)]
            for key,src in sources:
                u,s,vh=geom(src[key+"/A"][:])
                uM,sM,vM=(mat[key+"/"+name][:] for name in ("U","sigma","V"))
                _,error,bound,by_size,counts=subspace_errors((u,s,vh.swapaxes(-1,-2)),(uM,sM,vM),1e-6)
                assert bound<=1, (method,key,bound)
                rows.append(dict(method=method,condition=key,max_projector_error=error,max_bound_ratio=bound,
                                 pairs_1=int(counts[0]),pairs_2=int(counts[1]),pairs_3=int(counts[2]),
                                 pairs_4=int(counts[3]),max_dim2=float(by_size[1]),max_dim3=float(by_size[2])))
    write_csv(run/FOLDERS[9]/"representative_subspace_checks.csv",rows)
    print("Representative singular subspaces checked:",len(rows))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    args=p.parse_args()
    check(args.run)
