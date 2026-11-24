"""Validate representative G0 JPEG predictions against the pinned upstream extractor."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image

from calibrate import FOLDERS
from watermark import CONFIG,phase01,upstream_module,wm_order


def check(run,upstream_path):
    upstream=upstream_module(upstream_path)
    frozen=json.loads((run/FOLDERS[0]/"frozen_parameters.json").read_text(encoding="utf-8"))["methods"]["G0"]
    rows=[]
    with h5py.File(run/FOLDERS[5]/"G0_jpeg.h5","r") as h:
        for q in (1,50,80,100):
            label=f"QF{q:03d}"
            jpeg=run/FOLDERS[3]/"G0"/f"{label}.jpg"
            with Image.open(jpeg) as im:
                bgr=np.asarray(im)[:,:,::-1].copy()
            core=upstream.WaterMarkCore(password_img=CONFIG["password_img"],mode="common")
            core.wm_size=CONFIG["watermark_length"]
            core.d1,core.d2=frozen["d1"],frozen["d2"]
            actual=core.extract_raw(bgr)
            vote=core.extract_avg(actual)
            decoded=np.empty(len(vote),dtype=np.uint8)
            decoded[wm_order(len(vote),CONFIG["password_wm"])]=upstream.one_dim_kmeans(vote)
            f=h[f"jpeg/{label}"]
            assert np.array_equal(actual,f["block_scores"][:])
            assert np.array_equal(vote,f["votes"][:])
            assert np.array_equal(decoded,f["final_bits"][:])
            rows.append(dict(qf=q,block_scores_match=int(actual.size),votes_match=int(vote.size),
                             final_bits_match=int(decoded.size),image_sha256=phase01.sha(jpeg)))
    (run/FOLDERS[9]/"G0_upstream_reference.json").write_text(
        json.dumps(rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Pinned G0 original extractor",rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("upstream",type=Path)
    a=p.parse_args()
    check(a.run,a.upstream)
