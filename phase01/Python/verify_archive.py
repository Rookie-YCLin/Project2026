"""核對 JPEG、主 HDF5、統計表與封存依據。"""
import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin
from image_pipeline import sha, json_file


def verify(run):
    log = run/"10_執行紀錄"
    env = json.loads((log/"environment_python.json").read_text(encoding="utf-8"))
    original = Path(json.loads((log/"run.json").read_text(encoding="utf-8"))["original_path"])
    assert sha(original) == env["original"]["sha256"]
    n = env["blocks_per_channel"]
    w,h = env["original"]["width"],env["original"]["height"]
    qf = [f"QF{q:03d}" for q in range(1,101)]
    j = run/"01_JPEG影像"
    meta = json.loads((j/"jpeg_metadata.json").read_text(encoding="utf-8"))
    assert len(meta) == 100 and len(list(j.glob("*.jpg"))) == 100
    for q,row in zip(qf,meta,strict=True):
        img = j/f"{q}.jpg"
        assert int(q[2:]) == row["qf"] and sha(img) == row["sha256"]
        with Image.open(img) as im:
            assert im.size == (w,h) and JpegImagePlugin.get_sampling(im) == 0
            assert [list(item) for item in im.layer] == row["sampling_factors"]
            assert {str(k):v for k,v in im.quantization.items()} == row["quantization_tables"]
    dataset_count = 0
    with ExitStack() as context:
        inputs = context.enter_context(h5py.File(run/"02_原始矩陣"/"block_matrices.h5"))
        factors = context.enter_context(h5py.File(run/"03_SVD分解結果"/"svd_matlab.h5"))
        errors = context.enter_context(h5py.File(run/"04_逐區塊擾動數據"/"perturbation_matlab.h5"))
        def check(_, obj):
            nonlocal dataset_count
            if isinstance(obj,h5py.Dataset):
                assert obj.chunks is not None and obj.compression == "gzip"
                assert {"dtype","axes","indices"}.issubset(obj.attrs)
                dataset_count += 1
        for f in (inputs,factors,errors):
            assert set(f["jpeg"]) == set(qf)
            f.visititems(check)
        assert inputs["original/A"].id == inputs["identity/A"].id
        for group in ("original", *[f"jpeg/{q}" for q in qf]):
            assert inputs[f"{group}/A"].shape == (3,n,4,4)
            for k in ("U","V"):
                assert factors[f"{group}/{k}"].shape == (3,n,4,4)
            assert factors[f"{group}/sigma"].shape == (3,n,4)
        for group in ("identity", *[f"jpeg/{q}" for q in qf]):
            assert errors[f"{group}/metrics"].shape[:2] == (3,n)
            assert errors[f"{group}/diagnostics"].shape[:2] == (3,n)
        for q,row in zip(qf,meta,strict=True):
            assert inputs[f"jpeg/{q}"].attrs["jpeg_sha256"] == row["sha256"]
        assert np.array_equal(inputs["index/block_id"][:].ravel(),np.arange(3*n))
    stats = pd.read_csv(run/"05_統計分析"/"statistics_matlab.csv")
    assert set(stats.condition) == {"identity",*qf}
    assert not stats.duplicated(["condition","scope","metric"]).any()
    p = stats[stats.metric.isin(["D_F","D_sigma",*[f"theta_{side}_{i}" for side in "uv" for i in range(1,5)]])]
    assert len(p) == 101*4*10
    assert np.array_equal(p["count"],np.where(p.scope=="All",3*n,n))
    json_file(log/"archive_checks.json",dict(qf_count=100, jpeg_size=[w,h],
              jpeg_files_verified=100,hdf5_datasets_checked=dataset_count,
              statistics_rows=len(stats),core_statistics_rows=len(p),blocks_per_channel=n,
              total_jpeg_block_pairs=3*n*100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    verify(parser.parse_args().run)
