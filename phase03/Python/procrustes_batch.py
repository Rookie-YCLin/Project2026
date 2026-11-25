"""Batched determinant-split solver, with scalar recovery at difficult faces."""
import time
import numpy as np

from procrustes import branch, EPS, HALF_PI
from image_experiment import EYE, outer, target_many


def complete_many(w,x,j,det):
    signs=np.where(x[:,j]>=0,1.,-1.)
    z=EYE[j]+signs[:,None]*x
    z/=np.linalg.norm(z,axis=1)[:,None]
    H=EYE-2*outer(z,z)
    H[:,:,j]=x
    rest=[k for k in range(4) if k!=j]
    N=H[:,:,rest]
    U,_,Vh=np.linalg.svd(N[:,rest,:].transpose(0,2,1)*w[:,None,rest])
    U[:,:,-1]*=(det*signs*np.sign(np.linalg.det(U@Vh)))[:,None]
    H[:,:,rest]=N@(U@Vh)
    return H


def evaluate(w,c,j,lam,det):
    M=EYE*w[:,None,:]
    M[:,:,j]+=lam[:,None]*c
    L,g,Ht=np.linalg.svd(M)
    orient=det*np.sign(np.linalg.det(L@Ht))
    X=L.copy()
    X[:,:,-1]*=orient[:,None]
    col=np.einsum("ijk,ik->ij",X,Ht[:,:,j])
    h=(c*col).sum(axis=1)
    s=g[:,:3].sum(axis=1)+orient*g[:,3]
    return h,s,col,L,g,Ht,orient


def candidate_columns(st,c,j,t,cushion=192.):
    h,_,col,L,g,Ht,d=st
    tol=cushion*EPS*np.maximum(1,g[:,0])
    a=np.einsum("ijk,ij->ik",L,c)
    b=Ht[:,:,j]
    z=np.count_nonzero(g<=tol[:,None],axis=1)
    k=np.count_nonzero(g-g[:,3,None]<=tol[:,None],axis=1)
    kind=np.zeros(len(c),np.uint8)
    for size in (2,3,4):
        ix=np.flatnonzero(z==size)
        if len(ix):
            r=4-size
            ai,bi=a[ix,r:],b[ix,r:]
            alpha=np.linalg.norm(ai,axis=1)
            beta=np.linalg.norm(bi,axis=1)
            rest=t[ix]-(a[ix,:r]*b[ix,:r]).sum(axis=1)
            lim=alpha*beta
            valid=abs(rest)<=lim+cushion*EPS
            if np.any(valid):
                ii=ix[valid]
                A,B=ai[valid],bi[valid]
                an,bn=alpha[valid],beta[valid]
                scalar=np.divide(rest[valid],lim[valid],out=np.zeros(len(ii)),where=lim[valid]>0)
                scalar=np.clip(scalar,-1,1)
                aunit=np.divide(A,an[:,None],out=np.zeros_like(A),where=an[:,None]>0)
                pivot=np.argmin(abs(aunit),axis=1)
                safe=np.eye(size)[pivot]
                u=safe-(safe*aunit).sum(axis=1)[:,None]*aunit
                u/=np.linalg.norm(u,axis=1)[:,None]
                y=bn[:,None]*(scalar[:,None]*aunit+np.sqrt((1-scalar)*(1+scalar))[:,None]*u)
                y=np.where((lim[valid]>0)[:,None],y,B)
                coords=np.concatenate([b[ii,:r],y],axis=1)
                col[ii]=np.einsum("ijk,ik->ij",L[ii],coords)
                kind[ii]=1
        ix=np.flatnonzero((k==size)&(d<0)&(kind==0))
        if len(ix):
            r=4-size
            A,B=a[ix,r:],b[ix,r:]
            values,V=np.linalg.eigh((outer(A,B)+outer(B,A))/2)
            eta=((a[ix]*b[ix]).sum(axis=1)-t[ix])/2
            valid=(eta>=values[:,0]-cushion*EPS)&(eta<=values[:,-1]+cushion*EPS)
            if np.any(valid):
                item=ix[valid]
                ev=values[valid]
                R=V[valid]
                frac=np.divide(eta[valid]-ev[:,0],ev[:,-1]-ev[:,0],out=np.zeros(len(item)),
                               where=ev[:,-1]>ev[:,0])
                frac=np.clip(frac,0,1)
                v=np.sqrt(1-frac)[:,None]*R[:,:,0]+np.sqrt(frac)[:,None]*R[:,:,-1]
                B=B[valid]
                y=B-2*v*(v*B).sum(axis=1)[:,None]
                coords=np.concatenate([b[item,:r],y],axis=1)
                col[item]=np.einsum("ijk,ik->ij",L[item],coords)
                kind[item]=2
    return col,kind


def solve_one(w,c,j,phi,sgn,det):
    n=len(c)
    t=np.where(phi==HALF_PI,0.,sgn*np.cos(phi))
    ss=np.sin(phi)
    den=np.maximum(1,w.sum(axis=1))
    target=target_many(c,np.broadcast_to(EYE[j],c.shape),t,ss)
    R=complete_many(w,target,j,det)
    F=(w*np.diagonal(R,axis1=1,axis2=2)).sum(axis=1)
    G=np.full(n,np.inf)
    count=np.zeros(n,np.int16)
    lambda_value=np.zeros(n)
    kind=np.zeros(n,np.uint8)
    corner=(phi==0)|(w.sum(axis=1)==0)
    kind[corner]=3
    G[corner]=F[corner]
    not_corner=np.flatnonzero(~corner)
    goal=2e-10
    if len(not_corner):
        ix=not_corner
        st=evaluate(w[ix],c[ix],j,np.zeros(len(ix)),det)
        G[ix]=st[1]+32*EPS*(1+w[ix].sum(axis=1))
        count[ix]+=1
        # A face from near-zero weights proposes a feasible candidate. The bound
        # still uses the original weights, including the small positive entries.
        use_face=(w[ix,2]<2e-9) | (w[ix,2]-w[ix,3]<2e-9)
        if np.any(use_face):
            sub=ix[use_face]
            st=tuple(a[use_face] for a in st)
            x,form=candidate_columns(st,c[sub],j,t[sub],cushion=1e7)
            x=target_many(c[sub],x,t[sub],ss[sub])
            P=complete_many(w[sub],x,j,det)
            val=(w[sub]*np.diagonal(P,axis1=1,axis2=2)).sum(axis=1)
            better=val>F[sub]
            R[sub[better]],F[sub[better]],kind[sub[better]]=P[better],val[better],form[better]
    active=not_corner[(G[not_corner]-F[not_corner])/den[not_corner]>goal]
    ix=active
    initial=np.maximum(1e-12,w[:,j])
    low=-initial.copy()
    high=initial.copy()
    for _ in range(52):
        if not len(ix):
            break
        left=evaluate(w[ix],c[ix],j,low[ix],det)[0]-t[ix]
        right=evaluate(w[ix],c[ix],j,high[ix],det)[0]-t[ix]
        count[ix]+=2
        incomplete=(left>0)|(right<0)
        low[ix[left>0]]*=2
        high[ix[right<0]]*=2
        ix=ix[incomplete]
    for iteration in range(1,55):
        if not len(active): break
        mid=(low[active]+high[active])/2
        st=evaluate(w[active],c[active],j,mid,det)
        slope=st[0]-t[active]
        count[active]+=1
        upper=st[1]-mid*t[active]+32*EPS*(1+w[active].sum(axis=1)+abs(mid))
        improved=upper<G[active]
        lambda_value[active[improved]]=mid[improved]
        G[active]=np.minimum(G[active],upper)
        low[active[slope<0]]=mid[slope<0]
        high[active[slope>=0]]=mid[slope>=0]
        if iteration not in (8,14,22,32,43,54): continue
        x,form=candidate_columns(st,c[active],j,t[active],cushion=1e7)
        x=target_many(c[active],x,t[active],ss[active])
        P=complete_many(w[active],x,j,det)
        val=(w[active]*np.diagonal(P,axis1=1,axis2=2)).sum(axis=1)
        better=val>F[active]
        jx=active[better]
        R[jx],F[jx],kind[jx]=P[better],val[better],form[better]
        active=active[(G[active]-F[active])/den[active]>goal]
    for i in active:
        s=branch(w[i],c[i],j,float(phi[i]),sgn,det)
        if s.primal>F[i]:
            R[i],F[i]=s.R,s.primal
            kind[i]={"regular":4,"reflection_face":5,"null_face":6,"pole":3,"zero_objective":3}[s.kind]
        if s.dual<G[i]:
            G[i],lambda_value[i]=s.dual,s.bound_multiplier
        count[i]+=s.evaluations
    ort=np.linalg.norm(R.transpose(0,2,1)@R-EYE,axis=(1,2))
    proj=abs(np.sum(c*R[:,:,j],axis=1)-t)
    gap=(G-F)/den
    assert np.max(ort)<=1e-10 and np.max(proj)<=1e-10
    if np.max(gap)>1e-8 or np.min(gap)<-1e-10:
        k=int(np.argmax(gap) if np.max(gap)>1e-8 else np.argmin(gap))
        raise ArithmeticError(dict(row=k,weights=w[k].tolist(),coordinates=c[k].tolist(),
            phi=float(phi[k]),j=j,projection_sign=sgn,determinant=det,certificate_gap=float(gap[k]),
            primal=float(F[k]),dual=float(G[k]),multiplier=float(lambda_value[k]),kind=int(kind[k])))
    assert np.max(abs(np.linalg.det(R)-det))<=1e-10
    return R,F,G,lambda_value,kind,count,ort,proj


def solve_batch(w,c,j,phi):
    start=time.perf_counter()
    branches=[solve_one(w,c,j,phi,s,d) for s in (1,-1) for d in (1,-1)]
    values=np.stack([b[1] for b in branches],axis=1)
    bounds=np.stack([b[2] for b in branches],axis=1)
    ix=values.argmax(axis=1)
    x=np.arange(len(w))
    matrices=np.stack([b[0] for b in branches],axis=1)[x,ix]
    select=lambda z: np.stack([b[z] for b in branches],axis=1)[x,ix]
    stats=dict(primal=values[x,ix],dual=bounds.max(axis=1),chosen_branch0=ix.astype(np.uint8),
               branch_primal=values,branch_dual=bounds,scaled_lambda=select(3),degeneracy_code=select(4),
               branch_lambdas=np.stack([b[3] for b in branches],axis=1),
               svd_calls=np.stack([b[5] for b in branches],axis=1).sum(axis=1),
               orthogonality=select(6),signed_projection_error=select(7),
               solve_seconds_per_block_batch_average=np.full(len(w),(time.perf_counter()-start)/len(w)))
    return matrices,stats
