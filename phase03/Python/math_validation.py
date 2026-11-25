"""Generate shared deterministic fixtures and validate the independent Python solver."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
from scipy.optimize import minimize_scalar

from procrustes import HALF_PI,feature,givens,householder,loss,nearest,qim_indices,solve,theta_rotation

REFERENCE=np.full(4,.5)
LIMITS=dict(orthogonality=1e-10,projection=1e-10,angle=1e-8,determinant=1e-10,
            sigma=1e-10,primal_dual_gap=1e-8,frobenius=1e-10)


def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def generate(output,phase01,include_stress=False):
    rng=np.random.default_rng(20261125)
    cases=[]
    def add(name,s,c,j,phi,side="U",base=None):
        c=np.asarray(c)/np.linalg.norm(c)
        if base is None:
            u,_=householder(REFERENCE,0)
            h,_=householder(c,0)
            selected=u@h.T
            other=np.linalg.qr(rng.normal(size=(4,4)))[0]
        else:
            selected,other=base
        cases.append(dict(id=len(cases)+1,category=name,side=side,j=int(j),phi=float(phi),
                          sigma=np.asarray(s).tolist(),c=c.tolist(),
                          basis=selected.tolist(),other=other.tolist()))
    for kind in ("full_rank","positive_repeat","near_repeat","rank_0","rank_1","rank_2","rank_3","parallel","perpendicular"):
        for i in range(112):
            j=i%4
            s=np.sort(rng.uniform(.08,3,4))[::-1]
            c=rng.normal(size=4)
            phi=rng.uniform(.025,HALF_PI-.025)
            if i%11==0:
                phi=0.
            if i%11==1:
                phi=HALF_PI
            if i%37==2:
                phi=1e-4
            if kind=="positive_repeat":
                s[2]=s[3]=s[1]
            if kind=="near_repeat":
                s[2]=s[1]*(1-1e-10)
                s[3]=s[1]*(1-1e-12)
                s=np.sort(s)[::-1]
            if kind.startswith("rank_"):
                s[int(kind[-1]):]=0
            if kind=="parallel":
                c=np.eye(4)[j]*(-1)**i
            if kind=="perpendicular":
                c[j]=0
            add(kind,s*10.**rng.integers(-8,9),c,j,phi,("U","V")[i%2])
    with h5py.File(phase01,"r") as h:
        original=h["/original/A"][:]
        for i in range(160):
            channel,idx=i%3,int(rng.integers(0,original.shape[1]))
            U,s,Vh=np.linalg.svd(original[channel,idx])
            side=("U","V")[i%2]
            basis,other=(U,Vh.T) if side=="U" else (Vh.T,U)
            phi=(0.,HALF_PI,.15,.55,1.2,.9,.024,.67)[i%8]
            add("platelet",s,basis.T@REFERENCE,i%4,phi,side,(basis,other))
            cases[-1].update(channel=channel,block_index0=idx)
        if include_stress:
            flat=original.reshape(-1,4,4)
            U,s,Vh=np.linalg.svd(flat)
            ratios=np.divide(s[:,1],s[:,0],where=s[:,0]>0,out=np.ones(len(s)))
            tail=np.argsort(ratios,kind="stable")[:256:8]
            probe=np.r_[tail,rng.choice(len(flat),size=24,replace=False),299*89]
            for side in ("U","V"):
                for j in range(4):
                    for i,idx in enumerate(probe):
                        basis,other=(U[idx],Vh[idx].T) if side=="U" else (Vh[idx].T,U[idx])
                        c=basis.T@REFERENCE
                        old=feature(c,np.eye(4)[j])
                        q=np.deg2rad((.01,.01,.03,.3,2.,20.)[i%6])
                        bit=i%2
                        phi=float(q*qim_indices(old,q,bit))
                        add("platelet_near_rank_stress",s[idx],c,j,phi,side,(basis,other))
                        channel,block=divmod(int(idx),original.shape[1])
                        cases[-1].update(channel=channel,block_index0=block)
    add("O4_duality_gap",[4,3,2,1],[1,0,0,0],0,np.arccos(.5))
    write_json(output/"cases.json",dict(seed=20261125,limits=LIMITS,cases=cases))
    print("Shared math cases",len(cases),Counter(x["category"] for x in cases),flush=True)


def gap_fixture():
    w=np.array([16.,9.,4.,1.])
    c=np.eye(4)[0]
    p=solve(w/max(w),c,0,np.arccos(.5))
    positive=p[0]
    # The nonsplit support is sum singular values. Its convex minimum is at lambda=-16.
    result=minimize_scalar(lambda l:abs(16+l)+14-.5*l,bounds=(-40,4),method="bounded",
                           options={"xatol":1e-12})
    unconstrained=abs(16-16)+14+8
    assert abs(result.fun-unconstrained)<1e-6
    assert abs(positive.primal*16-21.5)<1e-10
    assert abs(positive.dual*16-21.5)<1e-9
    assert abs(positive.multiplier*16+17)<1e-8
    assert abs(loss(w,positive.R)-17)<1e-10
    return dict(ordinary_dual=unconstrained,det_positive_primal=positive.primal*16,
                det_positive_dual=positive.dual*16,lambda_sample=positive.multiplier*16,
                squared_loss=loss(w,positive.R),dual_gap=.5)


def run(output,tag="python"):
    obj=json.loads((output/"cases.json").read_text(encoding="utf-8"))
    rows=[]
    failures=[]
    start=perf_counter()
    for a in obj["cases"]:
        try:
            s,c=np.array(a["sigma"]),np.array(a["c"])
            j,phi=a["j"],a["phi"]
            scale=s.max()
            sig=s/scale if scale>0 else s
            w=sig**2
            e=np.eye(4)
            x=nearest(c,j,phi)
            S=e.copy()
            S[:,j]=x
            R=theta_rotation(x,j)
            g,planes=givens(w,c,j,phi)
            branches=solve(w,c,j,phi)
            selected=max(branches,key=lambda p:p.primal)
            loss_P=loss(w,selected.R)
            d_S,d_R=loss(w,S),loss(w,R)
            d_G=None if g is None else g[0]
            normW=max(1,w.sum())
            angle_max=max(abs(feature(c,M[:,j])-phi) for M in [S,R]+([g[1]] if g else [])+[p.R for p in branches])
            orth=max(np.linalg.norm(M.T@M-e,"fro") for M in [R]+([g[1]] if g else [])+[p.R for p in branches])
            constraint=max(abs(c@p.R[:,j]-p.signed_cos) for p in branches)
            det=max(abs(np.linalg.det(p.R)-p.determinant) for p in branches)
            gap=max(max(0,p.dual-p.primal)/normW for p in branches)
            lower=max((p.primal-p.dual)/normW for p in branches)
            formula=max(abs(loss(w,p.R)-(2*w.sum()-2*p.primal))/normW for p in branches)
            selected_basis=np.array(a["basis"])
            other=np.array(a["other"])
            A=(selected_basis*sig)@other.T
            if a["side"]=="V":
                A=A.T
            s_err=0.
            for M in [R]+([g[1]] if g else [])+[p.R for p in branches]:
                trial=(selected_basis@M*sig)@other.T
                if a["side"]=="V":
                    trial=trial.T
                s_err=max(s_err,float(max(abs(np.linalg.svd(trial,compute_uv=False)-sig))))
            B=(selected_basis@selected.R*sig)@other.T
            if a["side"]=="V":
                B=B.T
            formula=max(formula,abs(np.linalg.norm(B-A,"fro")**2-loss_P)/normW)
            cap=np.arctan2(np.linalg.norm(x-e[:,j]*(x[j])),x[j])
            sin2=np.linalg.norm(x-e[:,j])**2
            formula=max(formula,abs(d_S-4*w[j]*np.sin(cap/2)**2)/normW)
            if sin2>0:
                aux=x.copy(); aux[j]=0
                if np.linalg.norm(aux)>0:
                    aux/=np.linalg.norm(aux)
                    formula=max(formula,abs(d_R-4*np.sin(cap/2)**2*(w[j]+(w*aux**2).sum()))/normW)
            if g:
                formula=max(formula,abs(d_G-4*(w[j]+w[g[2]])*np.sin(g[3]/2)**2)/normW)
            order_error=max(0,d_S-loss_P,loss_P-d_R,0 if d_G is None else loss_P-d_G)/normW
            row=dict(id=a["id"],category=a["category"],side=a["side"],j=j,phi=phi,
                     D_S=d_S,D_G=d_G,D_R=d_R,D_P=loss_P,
                     orthogonality=orth,projection=constraint,angle=angle_max,
                     determinant=det,sigma=s_err,primal_dual_gap=gap,frobenius=formula,
                     dual_below_primal=max(0,lower),inequality_error=order_error,
                     planes=planes,p_kind=selected.kind,det_branch=selected.determinant,
                     projected_value=selected.signed_cos,lamb_scaled=selected.multiplier,
                     all_svd_calls=sum(v.evaluations for v in branches))
            for i,p in enumerate(branches,1):
                row[f"F{i}"]=p.primal
                row[f"G{i}"]=p.dual
                row[f"branch{i}"]=p.kind
            bad={m:row[m] for m,b in LIMITS.items() if row[m]>b}
            if order_error>1e-10 or lower>1e-10:
                bad.update(order_error=order_error,lower=lower)
            if bad:
                failures.append(dict(case=a,errors=bad,row=row))
            rows.append(row)
        except (FloatingPointError,ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
            failures.append(dict(case=a,error=repr(exc)))
        if a["id"]%200==0:
            print("Python",a["id"],"failures",len(failures),flush=True)
    if rows:
        with (output/f"{tag}_cases.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    write_json(output/f"{tag}_failures.json",failures)
    if not failures:
        example=gap_fixture()
        assert (qim_indices(np.array([.9,.4]),.8)==[1,1]).all()
        assert qim_indices(.5,.5,0)==2
        assert qim_indices(.9,HALF_PI,1)==1
        assert qim_indices(.9,1.,0,minimum=.1)==-1
        write_json(output/f"{tag}_summary.json",dict(passed=True,case_count=len(rows),
                   maxima={m:max(x[m] for x in rows) for m in LIMITS},
                   elapsed_seconds=perf_counter()-start,example=example))
    assert not failures,(len(failures),output/f"{tag}_failures.json")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("--phase01",type=Path)
    p.add_argument("--stress",action="store_true")
    args=p.parse_args()
    if args.phase01:
        generate(args.out,args.phase01,args.stress)
    run(args.out)
