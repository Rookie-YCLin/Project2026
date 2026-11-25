"""以凍結的 P 目標比較 S、G、R、P，G 不重選中心。"""
import argparse
from pathlib import Path

import h5py
import numpy as np

from image_experiment import DATA, EYE, LABELS, REF, geometry, put


def run(run_dir):
    directory=run_dir/DATA
    output=directory/"matched_targets.h5"
    with h5py.File(directory/"common.h5","r") as common:
        s=common["/original/sigma"][:]
        factors={k:common["/original/"+k][:] for k in ("U","V")}
    layout=s.shape[:2]
    s=s.reshape(-1,4)
    weight=np.divide(s,s[:,:1],out=np.zeros_like(s),where=s[:,:1]>0)**2
    normalization=np.maximum(1.,weight.sum(axis=1))
    with h5py.File(output.with_suffix(".pending"),"w") as h:
        h.attrs.update(scope="All blocks with frozen P targets",G_infeasible="NaN, no center reselection",
                       unit="squared Frobenius loss normalized by original sigma_1 squared")
        for label in (x for x in LABELS if x.startswith("P")):
            j=int(label[2])-1
            c=np.einsum("ijk,j->ik",factors[label[1]].reshape(-1,4,4),REF)
            c/=np.linalg.norm(c,axis=1)[:,None]
            with h5py.File(directory/label/"embedding.h5","r") as p:
                valid=p["/geometry/embed_success"][:].ravel()
                phi=p["/geometry/target_phi"][:].ravel()
                loss=p["/geometry/theoretical_loss"][:].ravel()
                G=p["/geometry/dual"][:].ravel()
                F=p["/geometry/primal"][:].ravel()
            data=np.full((len(s),4),np.nan)
            data[valid,3]=np.divide(loss[valid],s[valid,0]**2,out=np.zeros(valid.sum()),where=s[valid,0]>0)
            for k,m in enumerate("SGR"):
                T,details=geometry(weight[valid],c[valid],j,phi[valid],m)
                val=np.sum((T-EYE)**2*weight[valid,None,:],axis=(1,2))
                if m=="G":
                    val[details["auxiliary0"]<0]=np.nan
                data[valid,k]=val
            P=data[:,3]
            delta=np.column_stack((data[:,0]-P,P-data[:,1],P-data[:,2]))
            cert_up=2*np.maximum(0,G-F)
            assert np.nanmax(delta[:,0]/normalization)<=1e-10
            assert np.nanmax((delta[:,1:]-cert_up[:,None])/normalization[:,None])<=1e-10
            group=h.create_group(label)
            put(group,"loss",data.reshape(*layout,4)).attrs["fields"]="D_S,D_G,D_R,D_P"
            put(group,"comparison_violation",delta.reshape(*layout,3)).attrs["fields"]="D_S-D_P,D_P-D_G,D_P-D_R"
            put(group,"certificate_loss_tolerance",cert_up.reshape(layout))
            print(label,int(valid.sum()),int(np.isfinite(data[:,1]).sum()),float(np.nanmax(delta)),flush=True)
    output.with_suffix(".pending").replace(output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    run(parser.parse_args().run)
