"""相同 C-order 係數洗牌、Common/Slow 逆轉換與盲式投票。"""
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pywt

from allocation import allocate, index
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(1,str(CODE.parent/"phase01"/"Python"))
import image_pipeline as phase01

CONFIG=json.loads((CODE/"Configs"/"phase02.json").read_text(encoding="utf-8"))


def upstream_module(path):
    sha=subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()
    assert sha==CONFIG["upstream_commit"]
    status=subprocess.check_output(["git","-C",str(path),"status","--porcelain","--untracked-files=no"],
                                   text=True).strip()
    assert not status, "上游原碼有未提交變更"
    core=path/"blind_watermark"/"bwm_core.py"
    assert hashlib.sha256(core.read_text(encoding="utf-8").encode()).hexdigest()==phase01.CONFIG["upstream_core_lf_sha256"]
    sys.path.insert(0,str(path))
    return importlib.import_module("blind_watermark.bwm_core")


def payload():
    bits=np.random.default_rng(CONFIG["watermark_seed"]).integers(0,2,CONFIG["watermark_length"],dtype=np.uint8)
    return bits, bits[wm_order(len(bits),CONFIG["password_wm"])]


def wm_order(length,key):
    order=np.arange(length)
    np.random.RandomState(key).shuffle(order)
    return order


def block_bit_indices(n,length):
    return np.arange(n)%length


def png_roundtrip(path, pixels):
    success, data=cv2.imencode(".png",pixels)
    assert success
    path.write_bytes(data.tobytes())
    image=phase01.decode(path.read_bytes())
    assert image.dtype==np.uint8 and image.shape==pixels.shape
    return image


class PreparedImage:
    """A、原本 LL 與細節頻帶只準備一次。"""
    def __init__(self, source):
        assert phase01.sha(source)==CONFIG["original_sha256"]
        self.image=phase01.decode(source.read_bytes())
        ll,padded=phase01.ll_channels(self.image)
        self.details=tuple(pywt.dwt2(padded[:,:,c],"haar")[1] for c in range(3))
        self.ll=ll
        self.rows,self.cols=ll.shape[1]//4,ll.shape[2]//4
        self.n=self.rows*self.cols
        assert self.image.shape==(1108,738,3) and self.n==12696
        self.shuffle=phase01.permutations(self.n,CONFIG["password_img"])
        self.A=phase01.matrices(ll,self.shuffle)
        self.U,self.s,self.Vh=np.linalg.svd(self.A,full_matrices=False)
        self.original_bits,self.encrypted_bits=payload()
        self.bit_index=block_bit_indices(self.n,len(self.original_bits))
        self.block_bits=np.broadcast_to(self.encrypted_bits[self.bit_index],(3,self.n))

    def render(self, B):
        inv=np.argsort(self.shuffle,axis=-1)
        flat=np.take_along_axis(B.reshape(3,self.n,16),inv[None,...],axis=-1)
        unshuffled=flat.astype(np.float32).reshape(3,self.n,4,4)
        out=[]
        for c in range(3):
            block=np.stack([cv2.idct(a) for a in unshuffled[c]])
            LL=self.ll[c].copy()
            LL[:4*self.rows,:4*self.cols]=block.reshape(self.rows,self.cols,4,4).transpose(0,2,1,3).reshape(4*self.rows,4*self.cols)
            out.append(pywt.idwt2((LL,self.details[c]),"haar"))
        h,w=self.image.shape[:2]
        yuv=np.stack(out,axis=-1)[:h,:w]
        return np.clip(cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR),0,255)

    def embed(self, method, parameter, upstream, capture=False):
        if method!="G0":
            x,info=allocate(self.s,self.block_bits,parameter,method)
            B=(self.U*x[...,None,:])@self.Vh
            return self.render(B), B if capture else None, info if capture else None
        core=upstream.WaterMarkCore(password_img=CONFIG["password_img"],mode="common")
        core.read_img_arr(self.image)
        core.read_wm(self.encrypted_bits)
        core.d1,core.d2=36*parameter,20*parameter
        if not capture:
            return core.embed(),None,None
        trace=[]
        real_idct=upstream.idct
        real_svd=upstream.svd
        orig_s=[]
        def observe(transform):
            trace.append(transform.copy())
            return real_idct(transform)
        def observe_svd(a):
            value=real_svd(a)
            orig_s.append(value[1].copy())
            return value
        with patch.object(upstream,"idct",observe),patch.object(upstream,"svd",observe_svd):
            image=core.embed()
        B=np.take_along_axis(np.asarray(trace).reshape(3,self.n,16),self.shuffle[None,...],axis=-1)
        s=np.asarray(orig_s).reshape(3,self.n,4)
        x=s.copy()
        for j,d in enumerate((core.d1,core.d2)):
            x[...,j]=s[...,j]//d*d
            # 原版先算括號內的量化格位置，再乘以步長。
            x[...,j]=np.array([(s0//d+1/4+1/2*b)*d for s0,b in
                               zip(s[...,j].ravel(),self.block_bits.ravel(),strict=True)]).reshape(3,self.n)
        inv=x[...,1:]-x[...,:-1]
        tol=32*np.finfo(np.float32).eps*np.maximum(1,x[...,:1])
        return image,B.reshape(3,self.n,4,4).astype(float),dict(
            s_original=s.astype(float),s_modified=x.astype(float),T_components=x[...,:2].astype(float),
            y_components=(x[...,:2]-s[...,:2]).astype(float),
            rank_reversal=np.any(inv>tol,axis=-1),adjacent_tie=np.any(abs(inv)<=tol,axis=-1))


def extract(bgr, method, parameter, length, key_img, key_wm, upstream):
    """盲式入口。所有取樣位置及排列皆由收到的影像、參數和金鑰取得。"""
    ll,_=phase01.ll_channels(bgr)
    n=(ll.shape[1]//4)*(ll.shape[2]//4)
    A=phase01.matrices(ll,phase01.permutations(n,key_img))
    if method=="G0":
        d1,d2=36*parameter,20*parameter
        s=np.linalg.svd(A.astype(np.float32),compute_uv=False)
        f1,f2=(s[...,0]%d1)>d1/2,(s[...,1]%d2)>d2/2
        scores=(3*f1.astype(float)+f2)/4
        hard=scores>.5
    else:
        s=np.linalg.svd(A,compute_uv=False)
        ki=index(s.sum(axis=-1),parameter)
        hard=(ki%2)!=0
        scores=hard.astype(float)
    # 沿用郭飛 extract_avg 的逐位元計算順序及平均分母。
    votes=np.array([scores[:,j::length].mean() for j in range(length)])
    labels=upstream.one_dim_kmeans(votes)
    restored=np.empty(length,dtype=np.uint8)
    restored[wm_order(length,key_wm)] = labels
    return restored,A,s,scores,hard,votes


def psnr(reference,image):
    mse=np.mean((reference.astype(float)-image)**2)
    return float(np.inf if mse==0 else 10*np.log10(255**2/mse))
