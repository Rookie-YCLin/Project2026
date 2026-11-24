"""只使用正式 PNG 的未攻擊 PSNR 和盲式 BER，先完成八組驗收。"""
import argparse
import csv
import importlib.metadata
import json
import platform
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import features

from allocation import METHODS
from watermark import CODE,CONFIG,PreparedImage,extract,phase01,png_roundtrip,psnr,upstream_module,wm_order

FOLDERS=("01_實驗配置","02_步長校準","03_含浮水印影像","04_JPEG攻擊影像",
         "05_原始矩陣","06_嵌入與擾動數據","07_統計分析","08_MATLAB圖表",
         "09_Python驗證","10_交叉驗證","11_研究報告","12_執行紀錄")


def json_file(p,data):
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def write_csv(path,rows):
    with path.open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def put(f,name,a):
    dims={2:"channel,block",3:"channel,block,component",4:"channel,block,row,column"}
    a=np.asarray(a)
    axes=dims.get(a.ndim,"element")
    if name.endswith(("/U","/V")):
        axes="channel,block,row,singular_index"
    if name.startswith("/index/"):
        axes={"shuffle":"block,coefficient","grid":"rows,columns","block_id":"channel,block"}.get(name.split("/")[-1],"element")
    return phase01.dataset(f,name,a,axes,"HDF5 axes use C-order; block and coefficient IDs are zero-based")


def candidate_order(rows):
    return sorted(rows,key=lambda r:(abs(r["psnr_db"]-40),r["parameter"]))


def calibrate_one(args):
    run, path, upstream_path, method=args
    cv2.setNumThreads(1)
    image=PreparedImage(path)
    upstream=upstream_module(upstream_path)
    out=run/FOLDERS[1]/method
    out.mkdir()
    temp=out/"_candidate.png"
    best_bytes=None
    best=None
    records={}
    low,high=CONFIG["acceptance_psnr"]
    def scan(parameter,round_id):
        nonlocal best,best_bytes
        p=float(parameter)
        if p in records:
            return
        raw,_,_=image.embed(method,p,upstream)
        pixels=png_roundtrip(temp,raw)
        bits,*_=extract(pixels,method,p,len(image.original_bits),CONFIG["password_img"],CONFIG["password_wm"],upstream)
        perr=int(np.count_nonzero(bits!=image.original_bits))
        score=psnr(image.image,pixels)
        accepted=perr==0 and low<=score<=high
        row=dict(method=method,round=round_id,parameter=p,psnr_db=score,bit_errors=perr,
                 ber=perr/len(bits),accepted=accepted,png_sha256=phase01.sha(temp))
        records[p]=row
        if accepted and (best is None or (abs(score-40),p)<(abs(best["psnr_db"]-40),best["parameter"])):
            best=row
            best_bytes=temp.read_bytes()
    a,b,n=CONFIG["g0_search" if method=="G0" else "s_search"]
    for p in np.geomspace(a,b,int(n)):
        scan(p,0)
    for rd in range(1,CONFIG["refinement_rounds"]+1):
        ordered=sorted(records)
        pick=candidate_order([r for r in records.values() if r["bit_errors"]==0] or list(records.values()))[:2]
        intervals=set()
        for item in pick:
            i=ordered.index(item["parameter"])
            if i>0:
                intervals.add((ordered[i-1],ordered[i]))
            if i<len(ordered)-1:
                intervals.add((ordered[i],ordered[i+1]))
        for u,v in zip(ordered,ordered[1:]):
            if (records[u]["psnr_db"]-40)*(records[v]["psnr_db"]-40)<0:
                intervals.add((u,v))
        for a,b in sorted(intervals,key=lambda x:(min(abs(records[z]["psnr_db"]-40) for z in x),x))[:4]:
            for p in np.linspace(a,b,CONFIG["refinement_subdivisions"]+1)[1:-1]:
                scan(p,rd)
    write_csv(out/"candidates.csv",list(records.values()))
    temp.unlink()
    print(method,len(records),"best accepted",best,flush=True)
    if best is None:
        return None
    formal=run/FOLDERS[2]/f"{method}.png"
    with formal.open("xb") as f:
        f.write(best_bytes)
    assert phase01.sha(formal)==best["png_sha256"]
    pixels=phase01.decode(formal.read_bytes())
    raw,B,info=image.embed(method,best["parameter"],upstream,capture=True)
    again=png_roundtrip(out/"replay.png",raw)
    assert np.array_equal(again,pixels) and phase01.sha(out/"replay.png")==best["png_sha256"]
    (out/"replay.png").unlink()
    bits,B0,s,scores,hard,votes=extract(pixels,method,best["parameter"],len(image.original_bits),
                                      CONFIG["password_img"],CONFIG["password_wm"],upstream)
    assert np.array_equal(bits,image.original_bits) and psnr(image.image,pixels)==best["psnr_db"]
    if method=="G0":
        core=upstream.WaterMarkCore(password_img=CONFIG["password_img"],mode="common")
        core.wm_size=len(image.original_bits)
        core.d1,core.d2=36*best["parameter"],20*best["parameter"]
        ref=core.extract_raw(pixels)
        assert np.array_equal(scores,ref)
        assert np.array_equal(votes,core.extract_avg(ref))
    with h5py.File(run/FOLDERS[5]/f"{method}_embed.h5","x") as h:
        h.attrs.update(method=method,parameter=best["parameter"],png_sha256=best["png_sha256"])
        put(h,"/B/A",B)
        put(h,"/B0/A",B0)
        for key,value in info.items():
            put(h,f"/embedding/{key}",value)
        for key,value in dict(final_bits=bits,block_scores=scores,block_hard=hard,votes=votes,
                              s_extracted=s).items():
            put(h,f"/unattacked/{key}",value)
    return best


def main(root,source,upstream_path,phase1_matrix,workers,marker):
    assert root.is_dir()
    run=root/f"phase02_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}"
    run.mkdir()
    for folder in FOLDERS:
        (run/folder).mkdir()
    if marker:
        marker.write_text(str(run),encoding="utf-8")
    print("RUN="+str(run),flush=True)
    config_dir=run/FOLDERS[0]
    shutil.copyfile(CODE/"Configs"/"phase02.json",config_dir/"phase02.json")
    for name in ("RESEARCH_SPEC.md","DECISIONS_65_72.md"):
        shutil.copyfile(CODE.parent/"docs"/"phase02"/name,config_dir/name)
    img=PreparedImage(source)
    with h5py.File(phase1_matrix) as prior:
        assert np.array_equal(img.A,prior["original/A"][:])
        assert np.array_equal(img.shuffle,prior["index/shuffle"][:])
    watermark=img.original_bits
    config_dir.joinpath("watermark.bin").write_bytes(watermark.tobytes())
    config_dir.joinpath("watermark_encrypted.bin").write_bytes(img.encrypted_bits.tobytes())
    write_csv(config_dir/"watermark.csv",[dict(index0=i,plain=int(watermark[i]),
        encrypted=int(img.encrypted_bits[i]),encrypted_source_index0=int(j))
        for i,j in enumerate(wm_order(len(watermark),CONFIG["password_wm"]))])
    json_file(config_dir/"inputs.json",dict(original=str(source),original_sha256=phase01.sha(source),
        phase01_matrix=str(phase1_matrix),phase01_matrix_sha256=phase01.sha(phase1_matrix),
        watermark_sha256=phase01.sha(config_dir/"watermark.bin"),
        encrypted_sha256=phase01.sha(config_dir/"watermark_encrypted.bin"),
        python=sys.version,os=platform.platform(),cpu=platform.processor(),
        jpeg_lib=features.version_codec("jpg"),jpeg_turbo=features.version_feature("libjpeg_turbo"),
        hdf5=h5py.version.hdf5_version,libraries={n:importlib.metadata.version(n)
            for n in ("numpy","opencv-python","PyWavelets","Pillow","h5py","pandas","scipy")}))
    with h5py.File(run/FOLDERS[4]/"common.h5","x") as f:
        put(f,"/original/A",img.A)
        for key,value in dict(U=img.U,sigma=img.s,V=img.Vh.swapaxes(-1,-2)).items():
            put(f,f"/python_original_svd/{key}",value)
        positions=np.arange(img.n,dtype=np.int32)
        for key,value in dict(shuffle=img.shuffle.astype(np.uint8),grid=np.array([img.rows,img.cols]),
              block_id=np.arange(3*img.n).reshape(3,img.n),block_row=positions//img.cols,
              block_col=positions%img.cols,bit_index0=img.bit_index).items():
            put(f,f"/index/{key}",value)
        put(f,"/payload/original",watermark)
        put(f,"/payload/encrypted",img.encrypted_bits)
        put(f,"/payload/expected_blocks",img.block_bits)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        result=list(pool.map(calibrate_one,[(run,source,upstream_path,method) for method in METHODS]))
    accepted=[r for r in result if r is not None]
    if accepted:
        write_csv(run/FOLDERS[1]/"selected.csv",accepted)
    json_file(run/FOLDERS[1]/"acceptance_status.json",dict(
        result=dict(zip(METHODS,result,strict=True)),all_passed=len(accepted)==len(METHODS)))
    if len(accepted)!=len(METHODS):
        raise RuntimeError("至少一組未通過，已保存搜尋歷程，維持 JPEG 攻擊關閉")
    json_file(config_dir/"frozen_parameters.json",dict(methods={r["method"]:{
        **r,**({"d1":36*r["parameter"],"d2":20*r["parameter"]} if r["method"]=="G0" else {"q":r["parameter"]})}
        for r in accepted},calibration_uses_unattacked_only=True))
    frozen=config_dir/"frozen_parameters.json"
    (config_dir/"frozen_parameters.sha256").write_text(phase01.sha(frozen)+"  frozen_parameters.json\n",
                                                      encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("root","original","upstream","phase01_matrices"):
        p.add_argument(key,type=Path)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--marker",type=Path)
    a=p.parse_args()
    main(a.root,a.original,a.upstream,a.phase01_matrices,a.workers,a.marker)
