"""奇偶量化中心、固定權重可行域與單純形投影。"""
import numpy as np

EPS = np.finfo(np.float64).eps
METHODS = ("G0","S1","S2","S3","S4","S5","S6","S7")


def index(sum_sigma, q):
    return np.floor(sum_sigma/q+0.5).astype(np.int64)


def simplex(s, t):
    """投影到 x>=0、sum(x)=t，使用排序後的活動集合。"""
    v = np.sort(s,axis=-1)[...,::-1]
    j = np.arange(1,v.shape[-1]+1)
    threshold = (t[...,None]-np.cumsum(v,axis=-1))/j
    active = np.count_nonzero(v+threshold>0,axis=-1)
    tau = np.take_along_axis(threshold,np.maximum(active-1,0)[...,None],axis=-1)
    return np.where(t[...,None]>0,np.maximum(s+tau,0),0)


def weights(s, method):
    if method == "S1":
        return np.full_like(s,.25)
    if method in ("S6","S7"):
        s_total = s.sum(axis=-1,keepdims=True)
        numer = s if method=="S6" else s[...,::-1]
        return np.divide(numer,s_total,out=np.full_like(s,.25),where=s_total>0)
    j = int(method[1])-2
    return np.broadcast_to(np.eye(4)[j],s.shape)


def allocate(s, bit, q, method):
    assert method in METHODS[1:] and q>0 and np.isfinite(q)
    bit = np.broadcast_to(np.asarray(bit,dtype=np.int64),s.shape[:-1])
    S = s.sum(axis=-1)
    lower = bit + 2*np.maximum(0,np.floor((S/q-bit)/2)).astype(np.int64)
    upper = lower+2
    k0 = np.where(abs(S-lower*q)<abs(upper*q-S),lower,upper)
    w = weights(s,method)
    tol = 64*EPS*np.maximum(np.maximum(1,S),np.maximum(q,k0*q))
    minimum = np.zeros_like(S)
    if method!="S1":
        ratio=np.divide(s,w,out=np.full_like(s,np.inf),where=w>0)
        minimum=np.maximum(0,S-np.min(ratio,axis=-1))
    required=bit+2*np.maximum(0,np.ceil(((minimum-tol)/q-bit)/2)).astype(np.int64)
    k=np.maximum(k0,required)
    T=k*q
    y=T-S
    x=s+y[...,None]*w
    nonnegative_trigger=np.any(s+(k0*q-S)[...,None]*w < -tol[...,None],axis=-1)
    projected=np.any(x < 0,axis=-1)&(method=="S1")
    if np.any(projected):
        x[projected]=simplex(s[projected],T[projected])
    assert np.all(x>=-tol[...,None])
    x=np.maximum(x,0)
    assert np.all(abs(x.sum(axis=-1)-T)<=tol)
    assert np.array_equal(index(x.sum(axis=-1),q)%2,bit)
    inversions=x[...,1:]-x[...,:-1]
    info=dict(weights=w,s_original=s,s_modified=x,S=S,T_unconstrained=k0*q,T=T,T_min=minimum,
              k_unconstrained=k0,k=k,y_unconstrained=k0*q-S,y=y,
              nonnegative_trigger=nonnegative_trigger,projected=projected,reselected=k!=k0,
              rank_reversal=np.any(inversions>tol[...,None],axis=-1),
              adjacent_tie=np.any(abs(inversions)<=tol[...,None],axis=-1),
              zero_sum_fallback=(S==0)&(method in ("S6","S7")))
    return x,info
