"""Direction constraints and determinant-split Procrustes in SVD coordinates."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

EPS = np.finfo(float).eps
HALF_PI = np.pi / 2


def feature(c, x):
    d = np.sum(c * x, axis=-1)
    return np.arctan2(np.linalg.norm(x - d[..., None] * c, axis=-1), abs(d))


def target_vector(c, x, signed_cos, sine):
    """Closest vector at the specified signed latitude, with a deterministic pole rule."""
    v = x - (c @ x) * c
    v -= (c @ v) * c
    length = np.linalg.norm(v)
    if length <= 16*EPS:
        v = np.eye(len(c))[np.argmin(abs(c))] - c[np.argmin(abs(c))] * c
        length = np.linalg.norm(v)
    return signed_cos*c + sine*v/length


def householder(x, j):
    e = np.eye(len(x))[j]
    flip = 1 if x[j]>=0 else -1
    d = e + flip*x
    d /= np.linalg.norm(d)
    H = np.eye(len(x)) - 2*np.outer(d,d)
    H[:,j] *= -flip
    return H,flip


def complete(w, x, j, sign):
    """Exact optimal weighted completion for the fixed column and determinant."""
    H, detH = householder(x, j)
    other = np.arange(len(x)) != j
    N = H[:,other]
    U, _, Vh = np.linalg.svd(N[other,:].T * w[other])
    U[:,-1] *= sign*detH*np.linalg.det(U@Vh)
    H[:,other] = N@(U@Vh)
    H[:,j] = x
    return H


def theta_rotation(x, j):
    e = np.eye(len(x))[j]
    u = x.copy()
    u[j] = 0
    step = np.linalg.norm(u)
    if step <= 8*EPS:
        if x[j] >= 0:
            return np.eye(len(x))
        u = np.eye(len(x))[(j+1)%len(x)]
    else:
        u /= step
    return np.eye(len(x)) + (x[j]-1)*(np.outer(e,e)+np.outer(u,u)) + step*(np.outer(u,e)-np.outer(e,u))


def nearest(c, j, phi):
    t = np.cos(phi) if phi != HALF_PI else 0.0
    return target_vector(c,np.eye(len(c))[j],t if c[j]>=0 else -t,np.sin(phi))


def loss(w,R):
    return float(np.sum((R-np.eye(len(w)))**2*w))


def givens(w,c,j,phi):
    """Enumerate all phase roots; infeasible planes are identified before computing them."""
    best=None
    solutions=0
    for k in range(len(c)):
        if j == k:
            continue
        amp=np.hypot(c[j],c[k])
        residual=np.linalg.norm(c[np.logical_and(np.arange(len(c))!=j,np.arange(len(c))!=k)])
        if phi+16*EPS<np.arctan2(residual,amp):
            continue
        solutions+=1
        if amp == 0:
            angles=[0.]
        else:
            p=np.arctan2(c[k],c[j])
            t=np.cos(phi) if phi!=HALF_PI else 0.
            # The sine formula avoids loss of precision at phi=0 and on tangency.
            beta=np.arctan2(np.sqrt(max(0.,(np.sin(phi)-residual)*(np.sin(phi)+residual))),t)
            angles=[p+sgn*beta+shift for sgn in (-1,1) for shift in (0.,np.pi)]
        for theta in angles:
            C,S=np.cos(theta),np.sin(theta)
            R=np.eye(len(c))
            R[j,j]=R[k,k]=C
            R[k,j]=S
            R[j,k]=-S
            q=loss(w,R)
            if best is None or q<best[0]:
                best=(q,R,k,abs(np.arctan2(S,C)))
    return best,solutions


@dataclass
class Branch:
    R: np.ndarray
    primal: float
    dual: float
    multiplier: float
    determinant: int
    signed_cos: float
    kind: str
    evaluations: int
    bound_multiplier: float = 0.


def support(w,c,j,s,lam):
    M=np.diag(w)
    M[:,j]+=lam*c
    L,g,Ht=np.linalg.svd(M)
    orient=1 if np.linalg.det(L@Ht)>0 else -1
    desired=orient*s
    X=L.copy()
    X[:,-1]*=desired
    R=X@Ht
    h=float(c@R[:,j])
    value=float(np.sum(g[:-1])+desired*g[-1])
    return h,value,R,L,g,Ht,desired


def null_rotation(a,b,h,det):
    a_norm=np.linalg.norm(a)
    b_norm=np.linalg.norm(b)
    k=len(a)
    if min(a_norm,b_norm)==0:
        Q=np.eye(k)
        Q[-1,-1]=det
        return Q
    ratio=np.clip(h/(a_norm*b_norm),-1,1)
    y=target_vector(a/a_norm,np.eye(k)[np.argmin(abs(a))],ratio,
                    np.sqrt(max(0.,(1-ratio)*(1+ratio))))
    X,dx=householder(y,0)
    Y,dy=householder(b/b_norm,0)
    X[:,-1]*=det*dx*dy
    return X@Y.T


def face_column(state,c,j,t,multiplier):
    _,_,base,L,g,Ht,desired=state
    a=L.T@c
    b=Ht[:,j]
    tol=multiplier*EPS*max(1.,g[0])
    zeros=np.count_nonzero(g<=tol)
    if zeros>=2:
        k=len(g)-zeros
        target=t-a[:k]@b[:k]
        cap=np.linalg.norm(a[k:])*np.linalg.norm(b[k:])
        if abs(target)<=cap+multiplier*EPS:
            v=np.r_[b[:k],null_rotation(a[k:],b[k:],target,desired)@b[k:]]
            return L@v,"null_face"
    if desired<0:
        k=int(np.count_nonzero(g-g[-1]>tol))
        if len(g)-k>=2:
            ak,bk=a[k:],b[k:]
            values,V=np.linalg.eigh((np.outer(ak,bk)+np.outer(bk,ak))/2)
            eta=(a@b-t)/2
            if values[0]-multiplier*EPS<=eta<=values[-1]+multiplier*EPS:
                w=(np.clip((eta-values[0])/(values[-1]-values[0]),0,1)
                    if values[-1]>values[0] else 0.)
                v=np.sqrt(1-w)*V[:,0]+np.sqrt(w)*V[:,-1]
                return L@np.r_[b[:k],bk-2*v*(v@bk)],"reflection_face"
    return base[:,j], "regular"


def branch(w,c,j,phi,projection_sign,det_sign):
    """Convex derivative bracketing plus an independently feasible completion."""
    t=projection_sign*(np.cos(phi) if phi != HALF_PI else 0.)
    sine=np.sin(phi)
    total=float(np.sum(w))
    if phi == 0 or total == 0:
        x=target_vector(c,np.eye(len(c))[j],t,sine)
        R=complete(w,x,j,det_sign)
        f=float(w@np.diag(R))
        return Branch(R,f,f,0.,det_sign,t,"pole" if phi == 0 else "zero_objective",0)
    count=0
    samples=[]
    def evaluate(l):
        nonlocal count
        count+=1
        v=support(w,c,j,det_sign,l)
        error=32*EPS*(total+abs(l)+1)
        samples.append((v[1]-l*t+error,l,v))
        return v[0]-t
    low,high=-1.,1.
    dl,dh=evaluate(low),evaluate(high)
    for _ in range(56):
        if dl<=0<=dh:
            break
        if dl>0:
            low*=2
            dl=evaluate(low)
        if dh<0:
            high*=2
            dh=evaluate(high)
    if not dl<=0<=dh:
        raise ArithmeticError(("dual derivative unbracketed",dl,dh,phi))
    for _ in range(95):
        mid=(low+high)*.5
        slope=evaluate(mid)
        if high-low<=4*EPS*max(np.finfo(float).tiny,abs(mid)) or abs(slope)<2*EPS:
            break
        if slope<0:
            low=mid
        else:
            high=mid
    bound_sample=min(samples,key=lambda item:item[0])
    upper=bound_sample[0]
    best=None
    # The endpoints and the best dual values can expose different subgradients.
    chosen=samples[-5:]+sorted(samples,key=lambda p:p[0])[:3]+sorted(samples,key=lambda p:abs(p[2][0]-t))[:1]
    for _,l,state in chosen:
        for mult in (12.,48.,192.,768.,1e5,1e7):
            x,kind=face_column(state,c,j,t,mult)
            x=target_vector(c,x,t,sine)
            R=complete(w,x,j,det_sign)
            p=float(w@np.diag(R))
            if best is None or p>best[0]:
                best=p,R,l,kind
            if upper-p<=5e-12*max(1,total):
                break
        if upper-best[0]<=5e-12*max(1,total):
            break
    f,R,l,kind=best
    return Branch(R,f,upper,l,det_sign,t,kind,count,bound_sample[1])


def solve(w,c,j,phi):
    return [branch(w,c,j,phi,s,d) for s in (1,-1) for d in (1,-1)]


def qim_indices(phi,q,bit=None,minimum=0.):
    """Integer legal endpoints, nearest distance then larger index."""
    phi=np.asarray(phi)
    parity=0 if bit is None else np.asarray(bit)
    step=1 if bit is None else 2
    U=np.floor(HALF_PI/q).astype(np.int64)
    if U*q>HALF_PI:
        U-=1
    hi=np.floor((U-parity)/step).astype(np.int64)
    lo=np.maximum(0,np.ceil((np.asarray(minimum)/q-parity)/step)).astype(np.int64)
    # Correct floating-point ceil at an exact legal minimum.
    prev=parity+step*(lo-1)
    lo=np.where((lo>0)&(prev*q>=minimum),lo-1,lo)
    lo+=((parity+step*lo)*q<minimum)
    flo=np.floor((phi/q-parity)/step).astype(np.int64)
    a=parity+step*np.maximum(lo,np.minimum(flo,hi))
    b=parity+step*np.maximum(lo,np.minimum(flo+1,hi))
    pick=np.where(abs(phi-a*q)<abs(phi-b*q),a,b)
    return np.where((lo<=hi)&(pick*q<=HALF_PI),pick,-1)


def timed_solve(w,c,j,phi):
    start=perf_counter()
    b=solve(w,c,j,phi)
    return b,perf_counter()-start
