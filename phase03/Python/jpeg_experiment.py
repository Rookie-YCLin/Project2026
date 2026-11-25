"""全部角度步長凍結後，從各組正式 PNG 分別產生 QF 1–100。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import cv2
import h5py
import numpy as np
from PIL import features

from image_experiment import (CALIB, DATA, EYE, LABELS, P2_CONFIG, REF, STATS, PreparedImage,
                              blind, gate, geometry, phase01, psnr, put, rank_info, upstream_module)


def json_file(path,value):
    tmp=path.with_suffix(".pending")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    tmp.replace(path)


def csv_file(path,rows):
    tmp=path.with_suffix(".pending")
    with tmp.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def save_common(c,source):
    if c.is_file():
        return
    p=PreparedImage(source)
    with h5py.File(c.with_suffix(".pending"),"x") as f:
        f.attrs.update(original_sha256=P2_CONFIG["original_sha256"],channels="Y,U,V",
                       rows=p.rows,columns=p.cols,block_order="row-major, zero-based")
        for k,x in dict(matrices=p.A,U=p.U,sigma=p.s,V=p.Vh.swapaxes(-1,-2),
                        expected=p.block_bits).items():
            put(f,"/original/"+k,x)
        idx=np.arange(p.n,dtype=np.int32)
        for k,x in dict(shuffle=p.shuffle,bit_index=p.bit_index,block_row=idx//p.cols,
                        block_column=idx%p.cols,block_index=idx,
                        plaintext=p.original_bits,coded=p.encrypted_bits).items():
            d=f.create_dataset("/index/"+k,data=x)
            d.attrs["indices"]="zero-based; Y,U,V channels; row-major blocks"
    c.with_suffix(".pending").replace(c)


def save_sgr_geometry(run,common):
    """重建已保存名義旋轉的正交性，並由對應 B 核對。"""
    with h5py.File(common,"r") as f:
        U=f["/original/U"][:]
        s=f["/original/sigma"][:]
        V=f["/original/V"][:]
    shape=s.shape[:2]
    A=(U*s[...,None,:])@V.swapaxes(-1,-2)
    norm=np.maximum(np.linalg.norm(A,axis=(-2,-1)),1)
    weights=np.divide(s,s[...,:1],out=np.zeros_like(s),where=s[...,:1]>0).reshape(-1,4)**2
    for label in LABELS:
        if label.startswith("P"):
            continue
        base=run/DATA/label
        target=base/"geometry_audit.h5"
        if target.is_file():
            continue
        with h5py.File(base/"embedding.h5","r") as f:
            valid=f["/geometry/embed_success"][:].ravel()
            phi=f["/geometry/target_phi"][:].ravel()
            B=f["/B/matrices"][:]
        basis,other=(U,V) if label[1]=="U" else (V,U)
        c=np.einsum("ijk,j->ik",basis.reshape(-1,4,4),REF)
        c/=np.linalg.norm(c,axis=1)[:,None]
        transform=np.broadcast_to(EYE,(len(c),4,4)).copy()
        nominal,_=geometry(weights[valid],c[valid],int(label[2])-1,phi[valid],label[0])
        transform[valid]=nominal
        T=transform.reshape(*shape,4,4)
        predicted=(basis@T*s[...,None,:])@other.swapaxes(-1,-2)
        if label[1]=="V":
            predicted=predicted.swapaxes(-1,-2)
        error=np.linalg.norm(B-predicted,axis=(-2,-1))/norm
        assert float(error.max())<=1e-10,(label,float(error.max()))
        residual=np.linalg.norm(T.swapaxes(-1,-2)@T-EYE,axis=(-2,-1))
        with h5py.File(target.with_suffix(".pending"),"w") as f:
            put(f,"/orthogonality",residual)
            f.attrs["max_scaled_B_reconstruction_error"]=float(error.max())
        target.with_suffix(".pending").replace(target)


def freeze(run,source):
    gate(run)
    gate_pin=phase01.sha(run/"01_數學驗證"/"gate.json")
    stress=run/"01_數學驗證"/"stress_gate"
    for name in ("gate.json","batch_gate.json"):
        d=json.loads((stress/name).read_text(encoding="utf-8"))
        assert d["passed"] and max(d.get("fixtures",0),d.get("case_count",0))>=1000
    stress_pin=phase01.sha(stress/"gate.json")
    assert phase01.sha(source)==P2_CONFIG["original_sha256"]
    methods={}
    for label in LABELS:
        status=run/CALIB/label/"selection.json"
        d=json.loads(status.read_text(encoding="utf-8"))
        book=status.with_name("candidates.csv")
        with book.open(encoding="utf-8",newline="") as stream:
            rows=list(csv.DictReader(stream))
        assert len(rows)==d["tested_candidates"] and len(set(r["q_rad"] for r in rows))==len(rows)
        none=[r for r in rows if int(r["errors"])==0]
        successful=[r for r in none if 39.9<=float(r["psnr"])<=40.1]
        best=min(successful or none or rows,key=lambda r:(abs(float(r["psnr"])-40),float(r["q_rad"])))
        assert float(best["q_rad"])==d["q_rad"] and float(best["psnr"])==d["psnr"]
        assert int(best["errors"])==d["errors"] and best["png_sha256"]==d["png_sha256"]
        png=run/DATA/label/"unattacked.png"
        assert d["done"] and d["label"]==label and d["math_gate_sha256"]==gate_pin
        assert isinstance(d["q_rad"],float) and 0<d["q_rad"]<=np.pi/2
        assert phase01.sha(png)==d["png_sha256"]
        formally_calibrated=d["errors"]==0 and 39.9<=d["psnr"]<=40.1
        assert (d["classification"]=="formal")==formally_calibrated
        methods[label]={**d,"candidates_sha256":phase01.sha(book)}
    result=dict(
        methods=methods,phase02_commit="ecce4da07c7ba010a8bfe247c7e56273ddffc4cc",
        calibration=dict(calibration_uses_unattacked_only=True,log_grid_points=13,
                         refinements=2,subintervals_per_refinement=8),
        input=P2_CONFIG,math_gate_sha256=gate_pin,stress_gate_sha256=stress_pin,
        jpeg=dict(quality_factors=list(range(1,101)),subsampling=0),
    )
    config=run/CALIB/"frozen.json"
    if config.is_file():
        assert json.loads(config.read_text(encoding="utf-8"))==result
    else:
        json_file(config,result)
    (run/CALIB/"frozen.sha256").write_text(phase01.sha(config)+"  frozen.json\n",encoding="ascii")
    common=run/DATA/"common.h5"
    save_common(common,source)
    save_sgr_geometry(run,common)
    if not (run/"02_研究規格"/"environment.json").is_file():
        batch=run/"01_數學驗證"/"batch_gate.json"
        assert json.loads(batch.read_text(encoding="utf-8"))["passed"]
        json_file(run/"02_研究規格"/"environment.json",dict(
            host=platform.platform(),processor=platform.processor(),python=platform.python_version(),
            libraries={n:version(n) for n in ("numpy","opencv-python","PyWavelets","Pillow","h5py","scipy")},
            libjpeg=features.version_codec("jpg"),hdf5=h5py.version.hdf5_version,run_id=run.name))
    print("Frozen",len(methods),"methods,",sum(m["classification"]=="formal" for m in methods.values()),
          "formal; all candidates evaluated before JPEG.",flush=True)
    return methods


def process_one(task):
    run,source,upstream_path,label,frozen=task
    cv2.setNumThreads(1)
    base=run/DATA/label
    stats=run/STATS/(label+"_ber_psnr.csv")
    if stats.exists():
        assert (base/"attacks.h5").is_file()
        return
    upstream=upstream_module(upstream_path)
    png=base/"unattacked.png"
    assert phase01.sha(png)==frozen["png_sha256"]
    original=phase01.decode(source.read_bytes())
    clean=phase01.decode(png.read_bytes())
    rgb=np.ascontiguousarray(clean[:,:,::-1])
    j=int(label[2])-1
    q=frozen["q_rad"]
    with h5py.File(run/DATA/"common.h5","r") as common:
        expected=common["/original/expected"][:]
        payload=common["/index/plaintext"][:]

    def record(quality,received,bits,blocks,seconds,encoder_seconds=""):
        wrong=blocks!=expected
        by_channel=wrong.sum(axis=1)
        errors=int(np.count_nonzero(bits!=payload))
        return dict(method=label,classification=frozen["classification"],qf=quality or "",
                    q_rad=q,psnr_original_db=psnr(original,received),
                    psnr_vs_unattacked_db=psnr(clean,received) if quality else "",
                    final_errors=errors,bit_count=len(payload),ber=errors/len(payload),
                    block_errors_y=int(by_channel[0]),block_errors_u=int(by_channel[1]),
                    block_errors_v=int(by_channel[2]),block_count=int(wrong.size),
                    block_errors=int(wrong.sum()),block_ber=float(wrong.mean()),seconds_blind=seconds,
                    seconds_jpeg_encode_decode=encoder_seconds)

    with h5py.File(base/"embedding.h5","r") as h:
        rows=[record(None,clean,h["/B0/final_bits"][:],h["/B0/block_bits"][:],
                     float(frozen["seconds_blind"]))]
    assert rows[0]["final_errors"]==frozen["errors"]
    directory=base/"JPEG"
    directory.mkdir(exist_ok=True)
    final=base/"attacks.h5"
    partial=base/"attacks.partial"
    metadata=[]
    with h5py.File(final if final.is_file() else partial,"a") as h:
        h.attrs.update(method=label,q_rad=q,png_sha256=frozen["png_sha256"],
                       classification=frozen["classification"])
        for quality in range(1,101):
            name=f"/jpeg/QF{quality:03d}"
            jpeg=directory/f"QF{quality:03d}.jpg"
            if name in h:
                group=h[name]
                if group.attrs.get("complete",False) and jpeg.is_file():
                    assert phase01.sha(jpeg)==group.attrs["jpeg_sha256"]
                    rows.append(json.loads(group.attrs["row_json"]))
                    metadata.append(json.loads(group.attrs["jpeg_metadata"]))
                    continue
                del h[name]
            t=time.perf_counter()
            img,meta=phase01.encode(rgb,quality,jpeg)
            encoder_seconds=time.perf_counter()-t
            t=time.perf_counter()
            bits,block,votes,phi,idx,C,(U,s,V)=blind(img,label,q,upstream)
            elapsed=time.perf_counter()-t
            gap,tag=rank_info(s,j)
            group=h.create_group(name)
            for k,x in dict(matrices=C,U=U,sigma=s,V=V,block_bits=block,
                            final_bits=bits,block_error=block!=expected,votes=votes,phi=phi,
                            nearest_index=idx,nearest_gap=gap,rank_class=tag).items():
                put(group,k,x)
            row=record(quality,img,bits,block,elapsed,encoder_seconds)
            group.attrs.update(qf=quality,jpeg_sha256=meta["sha256"],
                               row_json=json.dumps(row,allow_nan=False),
                               jpeg_metadata=json.dumps(meta,allow_nan=False))
            h.flush()
            group.attrs["complete"]=True
            h.flush()
            rows.append(row)
            metadata.append(meta)
            if quality%10==0:
                print(label,"JPEG",quality,flush=True)
    if not final.is_file():
        partial.replace(final)
    json_file(directory/"jpeg_metadata.json",metadata)
    csv_file(stats,rows)


def main(args):
    methods=freeze(args.run,args.source)
    if not args.freeze_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(process_one,[(args.run,args.source,args.upstream,label,methods[label])
                                       for label in args.labels]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("run","source","upstream"):
        p.add_argument(name,type=Path)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--labels",nargs="+",choices=LABELS,default=LABELS)
    p.add_argument("--freeze-only",action="store_true")
    main(p.parse_args())
