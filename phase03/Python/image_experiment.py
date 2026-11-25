"""Vectorized S/G/R embedding and reproducible eight-bit PNG calibration."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import cv2
import h5py
import numpy as np

from procrustes import EPS,HALF_PI,feature,qim_indices

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(1,str(ROOT/"phase02"/"Python"))
from watermark import PreparedImage,phase01,png_roundtrip,psnr,wm_order,CONFIG as P2_CONFIG,upstream_module

LABELS=tuple(f"{m}{s}{j}" for m in "SGRP" for s in "UV" for j in range(1,5))
REF=np.full(4,.5)
EYE=np.eye(4)
CANDIDATES=13
LEVELS=2
SUBINTERVALS=8
CALIB="03_校準與凍結"
DATA="04_影像與矩陣"
STATS="05_統計分析"


def gate(run):
    g=json.loads((run/"01_數學驗證"/"gate.json").read_text(encoding="utf-8"))
    assert g["passed"] and g["case_count"]>=1000
    return g


def target_many(c,x,t,sn):
    z=x-(c*x).sum(axis=1)[:,None]*c
    z-=(c*z).sum(axis=1)[:,None]*c
    n=np.linalg.norm(z,axis=1)
    bad=n<=16*EPS
    if np.any(bad):
        idx=np.argmin(abs(c[bad]),axis=1)
        cb=c[bad]
        z[bad]=EYE[idx]-cb[np.arange(len(cb)),idx,None]*cb
        n[bad]=np.linalg.norm(z[bad],axis=1)
    return t[:,None]*c+(sn/n)[:,None]*z


def outer(a,b):
    return a[...,None]*b[...,None,:]


def geometry(w,c,j,phi,method):
    """Return a per-block change of coordinates and the selected auxiliary plane."""
    sign=np.where(c[:,j]>=0,1.,-1.)
    desired=np.where(phi==HALF_PI,0.,np.cos(phi))
    if method in ("S","R"):
        x=target_many(c,np.broadcast_to(EYE[j],c.shape),sign*desired,np.sin(phi))
        out=np.broadcast_to(EYE,(len(c),4,4)).copy()
        if method=="S":
            out[:,:,j]=x
        else:
            e=EYE[j]
            u=x.copy(); u[:,j]=0
            norm=np.linalg.norm(u,axis=1)
            u=np.divide(u,norm[:,None],out=np.zeros_like(u),where=norm[:,None]!=0)
            out+=(x[:,j]-1)[:,None,None]*(outer(e,e)+outer(u,u))
            out+=norm[:,None,None]*(outer(u,e)-outer(e,u))
        return out,{}
    if method=="P":
        from procrustes_batch import solve_batch
        return solve_batch(w,c,j,phi)
    n=len(c)
    cost=np.full(n,np.inf)
    aux=np.full(n,-1,np.int8)
    angles=np.zeros(n)
    planes=np.zeros(n,np.int8)
    for k in range(4):
        if j==k:
            continue
        amplitude=np.hypot(c[:,j],c[:,k])
        residual=np.linalg.norm(c[:,[x for x in range(4) if x not in (j,k)]],axis=1)
        feasible=(phi+16*EPS>=np.arctan2(residual,amplitude))
        planes+=feasible
        seed=np.arctan2(c[:,k],c[:,j])
        beta=np.arctan2(np.sqrt(np.maximum(0.,(np.sin(phi)-residual)*(np.sin(phi)+residual))),desired)
        for s in [-1,1]:
            for t in [0.,np.pi]:
                th=np.where(amplitude>0,seed+s*beta+t,0.)
                d=4*(w[:,j]+w[:,k])*np.sin(th/2)**2
                better=feasible & (d<cost)
                aux[better]=k
                cost[better]=d[better]
                angles[better]=th[better]
    out=np.broadcast_to(EYE,(n,4,4)).copy()
    loc=np.flatnonzero(aux>=0)
    k=aux[loc]
    C,S=np.cos(angles[loc]),np.sin(angles[loc])
    out[loc,j,j]=out[loc,k,k]=C
    out[loc,k,j]=S; out[loc,j,k]=-S
    return out,{"auxiliary0":aux,"feasible_planes":planes}


def put(f,name,x):
    a=np.asarray(x)
    d=f.create_dataset(name,data=a,compression="gzip",compression_opts=1,shuffle=True,fletcher32=True)
    d.attrs["dtype"]=str(a.dtype)
    d.attrs["layout"]=("channel,block,row,column" if a.ndim==4 else
                       "channel,block,field" if a.ndim==3 else
                       "channel,block" if a.ndim==2 else "sequence")
    return d


def rank_info(s,j):
    d=abs(s-s[...,j,None])
    d[...,j]=np.inf
    g=d.min(axis=-1)
    sigma=s[...,j]
    relative=np.maximum(s[...,0],np.finfo(float).tiny)
    zero=sigma<=32*EPS*relative
    repeat=g<=32*EPS*relative
    near=g<=1e-6*relative
    # 1 simple positive; 2 near repeat; 3 repeated positive; 4 simple zero; 5 multiple zero.
    tag=np.where(zero,np.where(repeat,5,4),np.where(repeat,3,np.where(near,2,1)))
    return g,tag.astype(np.uint8)


def embed(prepared,label,q):
    method,side,j=label[0],label[1],int(label[2])-1
    s=prepared.s.reshape(-1,4)
    U=prepared.U.reshape(-1,4,4)
    V=prepared.Vh.reshape(-1,4,4).swapaxes(-2,-1)
    F=U if side=="U" else V
    G=V if side=="U" else U
    c=np.einsum("ijk,j->ik",F,REF)
    c/=np.linalg.norm(c,axis=1)[:,None]
    n=len(c)
    basis=np.broadcast_to(EYE[j],c.shape)
    old=feature(c,basis)
    bits=prepared.block_bits.ravel()
    ki=qim_indices(old,q,bits)
    chosen=ki.copy()
    if method=="G":
        low=np.full(n,np.inf)
        for k in range(4):
            if j==k:
                continue
            a=np.hypot(c[:,j],c[:,k])
            b=np.linalg.norm(c[:,[x for x in range(4) if x not in (j,k)]],axis=1)
            low=np.minimum(low,np.arctan2(b,a))
        wait=(ki<0)|(ki*q+16*EPS<low)
        chosen[wait]=qim_indices(old[wait],q,bits[wait],np.maximum(0.,low[wait]-8*EPS))
    active=chosen>=0
    target=np.where(active,chosen*q,old)
    norm=s[:,0]
    normalized=np.divide(s,norm[:,None],out=np.zeros_like(s),where=norm[:,None]!=0)
    w=normalized**2
    # Preserve the exact Phase 01 input matrix for every failed block.
    B=prepared.A.reshape(-1,4,4).copy()
    rotations=np.broadcast_to(EYE,(n,4,4)).copy()
    back={}
    if np.any(active):
        T,back=geometry(w[active],c[active],j,target[active],method)
        if method=="G":
            assert np.all(back["auxiliary0"]>=0)
        assert np.max(abs(feature(c[active], T[:,:,j])-target[active]))<=1e-8
        V0=(F[active]@T*s[active,None,:])@G[active].swapaxes(-2,-1)
        B[active]=V0 if side=="U" else V0.swapaxes(-1,-2)
        rotations[active]=T
    ideal=(s[:,None,:]*(rotations-EYE))
    loss=np.sum(ideal**2,axis=(1,2))
    theta=np.arctan2(np.linalg.norm(rotations[:,:,j]-rotations[:,j,j,None]*EYE[j],axis=1),rotations[:,j,j])
    pidx=prepared.A.shape[:2]
    record=dict(initial_phi=old,target_phi=target,unconstrained_index=ki,target_index=chosen,
                embed_success=active,reselected=active&(chosen!=ki),theoretical_loss=loss,
                actual_rotation_angle=theta,sigma_scale=norm)
    for key,value in back.items():
        fill=-1 if key=="auxiliary0" else 0
        v=np.full((n,*value.shape[1:]),fill,dtype=value.dtype)
        v[active]=value
        record[key]=v
    return B.reshape(prepared.A.shape),{key:v.reshape(*pidx,*v.shape[1:]) for key,v in record.items()}


def blind(bgr,label,q,upstream):
    ll,_=phase01.ll_channels(bgr)
    n=(ll.shape[1]//4)*(ll.shape[2]//4)
    shuffle=phase01.permutations(n,P2_CONFIG["password_img"])
    x=phase01.matrices(ll,shuffle)
    U,s,Vh=np.linalg.svd(x,full_matrices=False)
    j=int(label[2])-1
    used=U[...,j] if label[1]=="U" else Vh[...,j,:]
    phi=feature(REF,used)
    indices=qim_indices(phi,q)
    assert np.all(indices>=0)
    block=(indices%2).astype(np.uint8)
    length=P2_CONFIG["watermark_length"]
    votes=np.array([block[:,k::length].mean() for k in range(length)])
    coded=upstream.one_dim_kmeans(votes)
    bits=np.empty(length,dtype=np.uint8)
    bits[wm_order(length,P2_CONFIG["password_wm"])]=coded
    return bits,block,votes,phi,indices,x,(U,s,Vh.swapaxes(-1,-2))


def remaining(cal,label):
    f=cal/label/"selection.json"
    if not f.is_file():
        return True
    status=json.loads(f.read_text(encoding="utf-8"))
    return not status.get("done",False)


def calibration_one(args):
    run,source,upstream_path,label=args
    cv2.setNumThreads(1)
    g=gate(run)
    book=run/CALIB/label
    if not remaining(run/CALIB,label):
        return label
    book.mkdir(parents=True,exist_ok=True)
    image=PreparedImage(source)
    upstream=upstream_module(upstream_path)
    temp=book/"candidate.tmp.png"
    candidates={}
    csv_path=book/"candidates.csv"
    if csv_path.exists():
        for r in csv.DictReader(csv_path.open(encoding="utf-8",newline="")):
            r["q_rad"]=float(r["q_rad"]); r["psnr"]=float(r["psnr"])
            for key in ("q_degrees","seconds_embed_png","seconds_blind"):
                r[key]=float(r[key])
            r["errors"]=int(r["errors"]); r["round"]=int(r["round"]); r["failed_blocks"]=int(r["failed_blocks"])
            r["block_errors"]=int(r["block_errors"])
            candidates[r["q_rad"]]=r
    def test(q,rd):
        q=float(q)
        if q in candidates:
            return
        t=time.perf_counter()
        B,embed_diag=embed(image,label,q)
        im=png_roundtrip(temp,image.render(B))
        duration=time.perf_counter()-t
        t=time.perf_counter()
        b,bl,*_=blind(im,label,q,upstream)
        row=dict(label=label,round=rd,q_rad=q,q_degrees=float(np.degrees(q)),psnr=psnr(image.image,im),
            errors=int(np.count_nonzero(b!=image.original_bits)),
            block_errors=int(np.count_nonzero(bl!=image.block_bits)),
            failed_blocks=int(np.count_nonzero(~embed_diag["embed_success"])),
            seconds_embed_png=duration,seconds_blind=time.perf_counter()-t,png_sha256=phase01.sha(temp))
        with csv_path.open("a",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if not candidates: writer.writeheader()
            writer.writerow(row)
        candidates[q]=row
    for q in np.geomspace(np.deg2rad(.01),HALF_PI,CANDIDATES):
        test(q,0)
    for rd in range(1,LEVELS+1):
        keep=[r for r in candidates.values() if r["round"]<rd]
        ordered=sorted(keep,key=lambda v:v["q_rad"])
        relevant=[r for r in ordered if r["errors"]==0] or ordered
        best=min(relevant,key=lambda r:(abs(r["psnr"]-40),r["q_rad"]))
        k=ordered.index(best)
        start=ordered[max(0,k-1)]["q_rad"]
        end=ordered[min(k+1,len(ordered)-1)]["q_rad"]
        for q in np.linspace(start,end,SUBINTERVALS+1)[1:-1]:
            test(q,rd)
    formal=[r for r in candidates.values() if r["errors"]==0 and 39.9<=r["psnr"]<=40.1]
    subset=formal or [r for r in candidates.values() if r["errors"]==0] or list(candidates.values())
    chosen=min(subset,key=lambda r:(abs(r["psnr"]-40),r["q_rad"]))
    q=chosen["q_rad"]
    B,record=embed(image,label,q)
    dest=run/DATA/label
    dest.mkdir(parents=True,exist_ok=True)
    im=png_roundtrip(dest/"unattacked.png",image.render(B))
    assert phase01.sha(dest/"unattacked.png")==chosen["png_sha256"]
    b,blocks,votes,phi,ind,B0,K=blind(im,label,q,upstream)
    assert psnr(image.image,im)==chosen["psnr"] and np.count_nonzero(b!=image.original_bits)==chosen["errors"]
    U,s,Vh=np.linalg.svd(B)
    basis=U[...,int(label[2])-1] if label[1]=="U" else Vh[...,int(label[2])-1,:]
    with h5py.File(dest/"embedding.pending","w") as h:
        put(h,"/B/matrices",B); put(h,"/B0/matrices",B0)
        put(h,"/B/svd_feature",feature(REF,basis))
        for k,v in record.items(): put(h,"/geometry/"+k,v)
        for k,v in dict(final_bits=b,block_bits=blocks,votes=votes,feature=phi,index=ind).items():
            put(h,"/B0/"+k,v)
        for key,values in (("B",(U,s,Vh.swapaxes(-1,-2))),("B0",K)):
            for name,v in zip(("U","sigma","V"),values,strict=True):
                put(h,f"/{key}/{name}",v)
        h.attrs.update(q_rad=q,classification="formal" if formal else "exploratory")
    (dest/"embedding.pending").replace(dest/"embedding.h5")
    summary={**chosen,"done":True,"classification":"formal" if formal else "exploratory",
             "tested_candidates":len(candidates),"math_gate_sha256":phase01.sha(run/"01_數學驗證"/"gate.json")}
    t=book/"selection.pending"
    t.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    t.replace(book/"selection.json")
    temp.unlink(missing_ok=True)
    print(label,summary["classification"],"PSNR",chosen["psnr"],"errors",chosen["errors"],flush=True)
    return label


def calibration_main(args):
    gate(args.run)
    if any(label.startswith("P") for label in args.labels):
        stress=args.run/"01_數學驗證"/"stress_gate"
        for name in ("gate.json","batch_gate.json"):
            assert json.loads((stress/name).read_text(encoding="utf-8"))["passed"]
    for p in (CALIB,DATA,STATS):
        (args.run/p).mkdir(exist_ok=True)
    labels=[s for s in args.labels if remaining(args.run/CALIB,s)]
    with ProcessPoolExecutor(max_workers=args.workers) as p:
        tasks={p.submit(calibration_one,(args.run,args.source,args.upstream,s)):s for s in labels}
        failures=[]
        for w in as_completed(tasks):
            try:
                w.result()
            except Exception:
                label=tasks[w]
                error=(args.run/CALIB/label/"failure.log")
                with error.open("a",encoding="utf-8") as f:
                    f.write(traceback.format_exc())
                failures.append(label)
                print(label, "calibration error", flush=True)
    if failures:
        raise RuntimeError(f"Calibration failed: {', '.join(failures)}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("source",type=Path)
    p.add_argument("upstream",type=Path)
    p.add_argument("--labels",choices=LABELS,nargs="+",default=LABELS)
    p.add_argument("--workers",type=int,default=4)
    calibration_main(p.parse_args())
