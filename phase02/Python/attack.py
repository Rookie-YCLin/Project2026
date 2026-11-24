"""八組全部通過 PNG 驗收後，執行獨立的 Pillow JPEG 4:4:4。"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np

from allocation import METHODS
from calibrate import FOLDERS, json_file, put, write_csv
from watermark import CONFIG, extract, phase01, psnr, upstream_module


def process_one(task):
    run,source,upstream_path,method,frozen=task
    cv2.setNumThreads(1)
    upstream=upstream_module(upstream_path)
    original=phase01.decode(source.read_bytes())
    assert phase01.sha(source)==CONFIG["original_sha256"]
    payload=np.frombuffer((run/FOLDERS[0]/"watermark.bin").read_bytes(),np.uint8)
    coded=np.frombuffer((run/FOLDERS[0]/"watermark_encrypted.bin").read_bytes(),np.uint8)
    png=run/FOLDERS[2]/f"{method}.png"
    assert phase01.sha(png)==frozen["png_sha256"]
    clean=phase01.decode(png.read_bytes())
    rgb=clean[:,:,::-1].copy()
    parameter=frozen["parameter"]
    settings=(method,parameter,len(payload),CONFIG["password_img"],CONFIG["password_wm"],upstream)
    bits,A,s,scores,hard,votes=extract(clean,*settings)
    assert np.array_equal(bits,payload)
    assert psnr(original,clean)==frozen["psnr_db"] and 39.9<=frozen["psnr_db"]<=40.1
    expected=coded[np.arange(A.shape[1])%len(payload)]
    expected=np.broadcast_to(expected,scores.shape)
    directory=run/FOLDERS[3]/method
    directory.mkdir(exist_ok=True)
    data=run/FOLDERS[5]/f"{method}_jpeg.h5"
    partial=data.with_suffix(".partial")
    summary=[]
    table_meta=[]
    def record(q,decoded,bits,s,h):
        errors=int(np.count_nonzero(bits!=payload))
        block_errors=np.count_nonzero(h!=expected,axis=1)
        summary.append(dict(method=method,condition="unattacked" if q is None else f"QF{q:03d}",
            qf="" if q is None else q,psnr_original_db=psnr(original,decoded),
            psnr_vs_unattacked_db="" if q is None else psnr(clean,decoded),bit_errors=errors,
            bit_count=len(payload),ber=errors/len(payload),ber_pct=100*errors/len(payload),
            block_errors_y=int(block_errors[0]),block_errors_u=int(block_errors[1]),
            block_errors_v=int(block_errors[2]),block_errors=int(block_errors.sum()),
            block_count=scores.size,block_ber=float(block_errors.sum()/scores.size),
            mean_nuclear=float(s.sum(axis=-1).mean())))
    record(None,clean,bits,s,hard)
    with h5py.File(partial,"x") as h5:
        h5.attrs.update(method=method,parameter=parameter,unattacked_png_sha256=phase01.sha(png))
        for q in range(1,101):
            decoded,meta=phase01.encode(rgb,q,directory/f"QF{q:03d}.jpg")
            bits,C,s,scores,hard,votes=extract(decoded,*settings)
            group=f"/jpeg/QF{q:03d}"
            for name,values in dict(A=C,block_scores=scores,block_hard=hard,votes=votes,
                                     final_bits=bits,s_extracted=s).items():
                put(h5,group+"/"+name,values)
            h5[group].attrs.update(qf=q,jpeg_sha256=meta["sha256"])
            table_meta.append(meta)
            record(q,decoded,bits,s,hard)
            if q%10==0:
                print(f"{method} JPEG {q}/100",flush=True)
    partial.rename(data)
    write_csv(run/FOLDERS[6]/f"{method}_ber_psnr.csv",summary)
    json_file(directory/"jpeg_metadata.json",table_meta)


def main(run,source,upstream,workers):
    frozen_file=run/FOLDERS[0]/"frozen_parameters.json"
    pin=(run/FOLDERS[0]/"frozen_parameters.sha256").read_text().split()[0]
    assert phase01.sha(frozen_file)==pin
    frozen=json.loads(frozen_file.read_text(encoding="utf-8"))
    assert frozen["calibration_uses_unattacked_only"]
    assert set(frozen["methods"])==set(METHODS)
    assert all(m["bit_errors"]==0 and 39.9<=m["psnr_db"]<=40.1 for m in frozen["methods"].values())
    model=run/FOLDERS[8]/"matlab_allocation_gate.json"
    assert json.loads(model.read_text(encoding="utf-8"))["passed"], "先完成 MATLAB 獨立驗收"
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(process_one,[(run,source,upstream,m,frozen["methods"][m]) for m in METHODS]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("run","source","upstream"):
        p.add_argument(k,type=Path)
    p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    main(a.run,a.source,a.upstream,a.workers)
