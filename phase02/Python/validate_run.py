"""用 NumPy 重算 SVD、擾動與分布，核對正式 JPEG 與存檔判決。"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin

from allocation import METHODS,index
from calibrate import FOLDERS,write_csv
from watermark import CONFIG,phase01,psnr,upstream_module,wm_order

NAMES=["D_F","D_sigma","R","nuclear_delta","nuclear_abs",
       *[f"delta_{j}" for j in range(1,5)],*[f"abs_delta_{j}" for j in range(1,5)],
       *[f"theta_u_{j}" for j in range(1,5)],*[f"theta_v_{j}" for j in range(1,5)]]
PROBS=[.50,.10,.25,.75,.90,.95]
STAT=["count","mean","median","std","min","max","p10","p25","p75","p90","p95"]


def geom(A):
    u,s,vh=np.linalg.svd(A,full_matrices=False)
    rec=(u*s[...,None,:])@vh
    re=np.max(np.linalg.norm(rec-A,axis=(-2,-1))/np.maximum(1,np.linalg.norm(A,axis=(-2,-1))))
    orth=max(np.max(np.linalg.norm(v@v.swapaxes(-1,-2)-np.eye(4),axis=(-2,-1))) for v in (u,vh))
    assert re<128*np.finfo(float).eps and orth<128*np.finfo(float).eps
    assert np.all(s>=0) and np.all(np.diff(s,axis=-1)<=0)
    return u,s,vh


def angle(u,v):
    return np.degrees(np.arccos(np.clip(abs((u*v).sum(axis=-2)),0,1)))


def pair(A,B,F,G):
    u,s,vh=F
    U,S,Vh=G
    delta=S-s
    df=((B-A)**2).sum(axis=(-2,-1))
    ds=(delta**2).sum(axis=-1)
    nuclear=S.sum(axis=-1)-s.sum(axis=-1)
    values=np.concatenate([np.stack([df,ds,df-ds,nuclear,abs(nuclear)],axis=-1),
                           delta,abs(delta),angle(u,U),angle(vh.swapaxes(-1,-2),Vh.swapaxes(-1,-2))],axis=-1)
    def stable(s):
        d=-np.diff(s,axis=-1)
        nearest=np.stack([d[...,0],np.minimum(d[...,0],d[...,1]),
                          np.minimum(d[...,1],d[...,2]),d[...,2]],axis=-1)
        return (nearest>1e-6*s[...,:1])&(s>1e-6*s[...,:1])
    return values,stable(s)&stable(S)


def compare(X,identified,dataset,tab,label,trans):
    ref=dataset[:]
    assert ref.shape==X.shape
    err=abs(ref-X)
    base=float(np.max(err[...,:13]/np.maximum(1,np.maximum(abs(ref[...,:13]),abs(X[...,:13])))))
    assert base<2e-9, (label,trans,base)
    angles=err[...,13:]
    ok=np.concatenate([identified,identified],axis=-1)
    precise=float(np.max(angles[ok],initial=0))
    unresolved=float(np.max(angles[~ok],initial=0))
    assert precise<.002, (label,trans,precise)
    mDiff=0.0
    reaggregate=0.0
    for c,scope in enumerate(("Y","U","V","All")):
        q=X[c] if c<3 else X.reshape(-1,21)
        qm=ref[c] if c<3 else ref.reshape(-1,21)
        expected=tab.loc[(label,trans,scope)].loc[NAMES,STAT].to_numpy()
        def statistics(q):
            p=np.quantile(q,PROBS,axis=0,method="linear")
            return np.vstack([np.full(q.shape[1],len(q)),q.mean(axis=0),p[0],
                              q.std(axis=0),q.min(axis=0),q.max(axis=0),p[1:]]).T
        diff=abs(statistics(qm)-expected)/np.maximum(1,abs(expected))
        reaggregate=max(reaggregate,float(diff.max()))
        assert diff.max()<2e-11, (label,trans,scope,diff.max())
        mDiff=max(mDiff,float(abs(q.mean(axis=0)[13:]-expected[13:,1]).max()))
    return dict(condition=label,comparison=trans,max_scaled_nonangular=base,max_identifiable_angle_deg=precise,
                max_unresolved_angle_deg=unresolved,max_angle_mean_difference_deg=mDiff,
                max_statistic_scaled_same_source=reaggregate,identifiable_angles=int(ok.sum()))


def process(task):
    run,source,upstream_path,method,wait=task
    if wait:
        while not (run/FOLDERS[9]/f"{method}_matlab_checks.csv").is_file():
            time.sleep(4)
    final_checks=run/FOLDERS[8]/f"{method}_independent_checks.csv"
    final_attacks=run/FOLDERS[9]/f"{method}_attack_checks.csv"
    if final_checks.is_file() and final_attacks.is_file():
        assert len(pd.read_csv(final_checks))==203 and len(pd.read_csv(final_attacks))==100
        return method
    upstream=upstream_module(upstream_path)
    origin=phase01.decode(source.read_bytes())
    data=run/FOLDERS[5]
    imgDir=run/FOLDERS[3]/method
    meta=json.loads((imgDir/"jpeg_metadata.json").read_text(encoding="utf-8"))
    assert [x["qf"] for x in meta]==list(range(1,101)) and len(list(imgDir.glob("*.jpg")))==100
    ber=pd.read_csv(run/FOLDERS[6]/f"{method}_ber_psnr.csv").set_index("condition")
    tab=pd.read_csv(run/FOLDERS[6]/f"{method}_matrix_stats.csv",float_precision="round_trip").set_index(
        ["condition","comparison","scope","metric"]).sort_index()
    out=[]
    records=[]
    with (h5py.File(run/FOLDERS[4]/"common.h5") as common,
          h5py.File(data/f"{method}_embed.h5") as e,
          h5py.File(data/f"{method}_jpeg.h5") as h,
          h5py.File(data/f"{method}_matlab.h5") as matlab):
        A=common["original/A"][:]; F=geom(A)
        B=e["B/A"][:]; G=geom(B)
        B0=e["B0/A"][:]; H=geom(B0)
        bits=common["payload/original"][:]
        truth=common["payload/expected_blocks"][:]
        step=float(e.attrs["parameter"])
        order=wm_order(len(bits),CONFIG["password_wm"])
        for name,X,Y,K,L in [("A_to_B",A,B,F,G),("A_to_B0",A,B0,F,H),("B_to_B0",B,B0,G,H)]:
            vals,identified=pair(X,Y,K,L)
            out.append(compare(vals,identified,matlab["unattacked/"+name],tab,"unattacked",name))
        for q,m in enumerate(meta,1):
            label=f"QF{q:03d}"
            path=imgDir/f"{label}.jpg"
            assert phase01.sha(path)==m["sha256"]
            with Image.open(path) as jpg:
                assert JpegImagePlugin.get_sampling(jpg)==0 and [list(a) for a in jpg.layer]==m["sampling_factors"]
                assert {str(k):v for k,v in jpg.quantization.items()}==m["quantization_tables"]
                im=np.asarray(jpg)[:,:,::-1]
            group=h[f"jpeg/{label}"]
            C=group["A"][:]; J=geom(C)
            math_group=matlab[f"jpeg/{label}"]
            assert np.max(abs(J[1]-math_group["sigma"][:])/np.maximum(1,J[1][...,:1]))<128*np.finfo(float).eps
            if method=="G0":
                s=np.linalg.svd(C.astype(np.float32),compute_uv=False)
                scores=(3*((s[...,0]%(36*step))>18*step).astype(float)+((s[...,1]%(20*step))>10*step))/4
            else:
                scores=index(J[1].sum(axis=-1),step)%2
            assert np.array_equal(scores,group["block_scores"][:]), (method,label)
            votes=np.array([scores[:,j::len(bits)].mean() for j in range(len(bits))])
            assert np.array_equal(votes,group["votes"][:])
            recovered=np.empty_like(bits)
            recovered[order]=upstream.one_dim_kmeans(votes)
            assert np.array_equal(recovered,group["final_bits"][:])
            err=np.count_nonzero(recovered!=bits)
            berr=np.count_nonzero((scores>.5)!=truth)
            assert err==ber.loc[label,"bit_errors"] and berr==ber.loc[label,"block_errors"]
            assert abs(psnr(origin,im)-ber.loc[label,"psnr_original_db"])<1e-12
            for name,X,K in [("B0_to_C",B0,H),("A_to_C",A,F)]:
                vals,identified=pair(X,C,K,J)
                out.append(compare(vals,identified,math_group[name],tab,label,name))
            records.append(dict(method=method,qf=q,jpeg_hash_ok=True,jpeg_444=True,
                bit_errors=int(err),block_errors=int(berr),bits_match=True))
            if q%20==0:
                print(method,"Python cross-check",q,flush=True)
    write_csv(final_checks,out)
    write_csv(final_attacks,records)
    return method


def main(run,source,upstream,workers,methods,wait):
    failed=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks={pool.submit(process,(run,source,upstream,m,wait)):m for m in methods}
        for task in as_completed(tasks):
            method=tasks[task]
            try:
                task.result()
                print(method,"Python full validation passed",flush=True)
            except Exception as e:
                print(method,"Python validation failed",repr(e),flush=True)
                failed.append((method,repr(e)))
    if failed:
        raise RuntimeError(failed)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("run","source","upstream"):
        p.add_argument(name,type=Path)
    p.add_argument("--workers",type=int,default=3)
    p.add_argument("--wait-for-matlab",action="store_true")
    p.add_argument("--methods",choices=METHODS,nargs="+",default=METHODS)
    a=p.parse_args()
    main(a.run,a.source,a.upstream,a.workers,a.methods,a.wait_for_matlab)
