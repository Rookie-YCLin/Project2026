"""產生可追溯的 4×4 區塊矩陣。磁碟維度採 NumPy / HDF5 次序。"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np
import pywt
from PIL import Image, JpegImagePlugin, features

CODE = Path(__file__).resolve().parents[1]
CONFIG = json.loads((CODE / "Configs" / "phase01.json").read_text(encoding="utf-8"))
CT = ("Y", "U", "V")
IMAGE_DIR = "01_JPEG影像"
MATRIX_DIR = "02_原始矩陣"
LOG_DIR = "10_執行紀錄"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_file(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def dataset(group, name, data, axes, indices):
    d = group.create_dataset(name, data=data, compression="gzip", compression_opts=1,
                             shuffle=True, fletcher32=True)
    d.attrs.update(axes=axes, dtype=str(d.dtype), indices=indices)
    return d


def ll_channels(bgr):
    """沿用 OpenCV float32、原版 0…255 輸入尺度與補邊順序。"""
    yuv = cv2.cvtColor(bgr.astype(np.float32), cv2.COLOR_BGR2YUV)
    h, w = bgr.shape[:2]
    padded = cv2.copyMakeBorder(yuv, 0, h % 2, 0, w % 2, cv2.BORDER_CONSTANT,
                                value=(0, 0, 0))
    return np.stack([pywt.dwt2(padded[:, :, channel], "haar")[0]
                     for channel in range(3)]), padded


def permutations(n, seed):
    return np.random.RandomState(seed).random_sample((n, 16)).argsort(axis=1)


def matrices(ll, shuffle):
    rows, cols = ll.shape[1] // 4, ll.shape[2] // 4
    blocks = ll[:, :4*rows, :4*cols].reshape(3, rows, 4, cols, 4)
    blocks = blocks.transpose(0, 1, 3, 2, 4).reshape(3, rows*cols, 4, 4)
    assert blocks.dtype == np.float32 and shuffle.shape == (rows*cols, 16)
    transformed = np.empty_like(blocks)
    for c in range(3):
        for i, block in enumerate(blocks[c]):
            transformed[c, i] = cv2.dct(block).ravel()[shuffle[i]].reshape(4, 4)
    return transformed.astype(np.float64)


def upstream_reference(upstream, bgr, ll, shuffle, expected):
    # 取已審閱的上游方法主體，避免匯入整套程式或執行浮水印。
    source = upstream / "blind_watermark" / "bwm_core.py"
    text = source.read_text(encoding="utf-8")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == CONFIG["upstream_core_lf_sha256"], (
        "上游前處理原碼 SHA-256 不符"
    )
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WaterMarkCore")
    names = {"read_img_arr", "init_block_index", "block_get_wm_slow"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    slow = next(n for n in methods if n.name == "block_get_wm_slow")
    slow.body = slow.body[:2] + [ast.Return(value=ast.Name(id="block_dct_shuffled", ctx=ast.Load()))]
    cls.body = methods
    rng = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "random_strategy1")
    module = ast.fix_missing_locations(ast.Module(body=[cls, rng], type_ignores=[]))
    scope = dict(np=np, cv2=cv2, dwt2=pywt.dwt2, dct=cv2.dct)
    exec(compile(module, str(source), "exec"), scope)
    core = scope["WaterMarkCore"]()
    core.block_shape, core.wm_size = np.array([4, 4]), 0
    core.ca, core.hvd, core.ca_block = [None]*3, [None]*3, [None]*3
    core.read_img_arr(bgr)
    core.init_block_index()
    reference_shuffle = scope["random_strategy1"](CONFIG["password_img"], core.block_num, 16)
    assert np.array_equal(shuffle, reference_shuffle)
    assert np.array_equal(ll, np.asarray(core.ca))
    largest = 0.0
    for c in range(3):
        for i, loc in enumerate(core.block_index):
            before_svd = core.block_get_wm_slow((core.ca_block[c][loc], reference_shuffle[i]))
            largest = max(largest, float(np.max(np.abs(before_svd - expected[c, i]))))
    assert largest == 0
    return largest


def decode(payload):
    return cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)


def encode(rgb, q, path):
    Image.fromarray(rgb).save(path, format="JPEG", quality=q, subsampling=CONFIG["jpeg_subsampling"],
                              optimize=False, progressive=False)
    with Image.open(path) as img:
        assert img.size == (rgb.shape[1], rgb.shape[0]) and img.mode == "RGB"
        assert JpegImagePlugin.get_sampling(img) == 0 and all(x[1:3] == (1, 1) for x in img.layer)
        result = np.asarray(img)[:, :, ::-1].copy()
        meta = dict(qf=q, sampling_factors=img.layer, quantization_tables=img.quantization,
                    sha256=sha(path), bytes=path.stat().st_size)
    return result, meta


def save_preprocess(group, bgr, ll, padded, rows, cols):
    # 角落、中心及邊界附近，保留 MATLAB 可獨立重算的 8×8 輸入。
    n = rows * cols
    i = np.unique([0, cols-1, cols+1, (rows//2)*cols+cols//2, n-cols, n-1])
    def crops(image, stride):
        return np.stack([image[:, (k//cols)*stride:(k//cols+1)*stride,
                              (k%cols)*stride:(k%cols+1)*stride] for k in i])
    h, w = bgr.shape[:2]
    bgr_pad = np.zeros(padded.shape, dtype=np.uint8)
    bgr_pad[:h, :w] = bgr
    fixtures = group.create_group("preprocess_check")
    dataset(fixtures, "block", i.astype(np.int32), "sample", "0-based block")
    dataset(fixtures, "bgr", crops(bgr_pad.transpose(2, 0, 1), 8),
            "sample,channel,row,column", "0-based; B,G,R")
    dataset(fixtures, "yuv", crops(padded.transpose(2, 0, 1), 8),
            "sample,channel,row,column", "0-based; Y,U,V")
    dataset(fixtures, "LL", crops(ll, 4), "sample,channel,row,column", "0-based; Y,U,V")


def generate(root, upstream, quality):
    source = upstream / CONFIG["upstream_image"]
    assert sha(source) == CONFIG["image_sha256"], "原始影像 SHA-256 不符"
    payload = source.read_bytes()
    bgr = decode(payload)
    with Image.open(source) as original:
        assert original.mode == "RGB" and original.size == (bgr.shape[1], bgr.shape[0])
        assert np.array_equal(np.asarray(original), bgr[:, :, ::-1])
        source_jpeg = dict(sampling_factors=original.layer,
                           quantization_tables=original.quantization)
    rgb = bgr[:, :, ::-1].copy()
    ll, padded = ll_channels(bgr)
    rows, cols = ll.shape[1] // 4, ll.shape[2] // 4
    n = rows * cols
    shuffle = permutations(n, CONFIG["password_img"])
    A = matrices(ll, shuffle)
    delta = upstream_reference(upstream, bgr, ll, shuffle, A)
    assert np.array_equal(A, matrices(ll_channels(bgr.copy())[0], shuffle))
    h, w = bgr.shape[:2]
    print(f"Original {w}×{h}, LL {ll.shape[1:]}, grid {rows}×{cols}; reference {delta:g}", flush=True)
    jpeg_meta = []
    library_names = ("numpy", "opencv-python", "PyWavelets", "Pillow", "h5py", "scipy",
                     "pandas", "matplotlib")
    environment = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                       machine=platform.machine(), processor=platform.processor(), python=sys.version,
                       libraries={n: importlib.metadata.version(n) for n in library_names},
                       pillow_jpeg=features.version_codec("jpg"),
                       pillow_libjpeg_turbo=features.version_feature("libjpeg_turbo"),
                       hdf5=h5py.version.hdf5_version, password_img=CONFIG["password_img"],
                       rng="numpy RandomState MT19937; argsort independently per block",
                       upstream_sha=CONFIG["upstream_sha"], project_spec_sha=CONFIG["project_spec_sha"],
                       original=dict(filename=CONFIG["upstream_image"], sha256=sha(source),
                                     width=w, height=h, **source_jpeg),
                       ll_shape=list(ll.shape[1:]), block_grid=[rows, cols],
                       blocks_per_channel=n, upstream_max_abs_error=delta,
                       pillow_opencv_source_decodes_equal=True, qualities=list(quality))
    json_file(root / LOG_DIR / "environment_python.json", environment)
    (root / LOG_DIR / "opencv_build.txt").write_text(cv2.getBuildInformation(), encoding="utf-8")
    target = root / MATRIX_DIR / "block_matrices.h5"
    with h5py.File(target.with_suffix(".partial"), "x") as f:
        f.attrs.update(schema="phase01/1", channel_order="Y,U,V",
                       source_sha256=sha(source), seed=CONFIG["password_img"])
        index = f.create_group("index")
        base = np.arange(n, dtype=np.int32)
        for name, values in (("block", base), ("block_row", base//cols), ("block_col", base%cols),
                             ("ll_row", 4*(base//cols)), ("ll_col", 4*(base%cols))):
            dataset(index, name, values, "block", "0-based")
        dataset(index, "channel", np.arange(3, dtype=np.int32), "channel", "0=Y; 1=U; 2=V")
        dataset(index, "block_id", np.arange(3*n, dtype=np.int32).reshape(3, n),
                "channel,block", "0-based global ID; channel*n+block")
        dataset(index, "shuffle", shuffle.astype(np.uint8), "block,coefficient",
                "0-based; C-order gather, reused across YUV and all conditions")
        dataset(index, "grid", np.array([rows, cols], np.int32), "grid_dimension", "rows,columns")
        original = f.create_group("original")
        axes = "channel,block,row,column"
        indexed = "0-based; Y,U,V; block row-major; matrix row-major"
        dataset(original, "A", A, axes, indexed)
        save_preprocess(original, bgr, ll, padded, rows, cols)
        identity = f.create_group("identity")
        identity.attrs.update(A_source="/original/A", jpeg="none")
        identity["A"] = original["A"]
        # 非對稱、全元素不同，外加獨立的通道和區塊編碼。
        c, i, r, k = np.indices((3, 2, 4, 4))
        dataset(f, "layout_sentinel", 1000*c + 100*i + 10*r + k, axes, indexed)
        for q in quality:
            path = root / IMAGE_DIR / f"QF{q:03d}.jpg"
            decoded, meta = encode(rgb, q, path)
            ll_q, yuv_q = ll_channels(decoded)
            B = matrices(ll_q, shuffle)
            group = f.create_group(f"jpeg/QF{q:03d}")
            group.attrs.update(qf=q, jpeg_sha256=meta["sha256"])
            dataset(group, "A", B, axes, indexed)
            if q in CONFIG["pilot_qf"]:
                save_preprocess(group, decoded, ll_q, yuv_q, rows, cols)
                repeat = path.with_suffix(".repro.jpg")
                decoded2, meta2 = encode(rgb, q, repeat)
                regen = matrices(ll_channels(decoded2)[0], shuffle)
                assert meta == meta2 and np.array_equal(decoded2, decoded) and np.array_equal(regen, B)
                repeat.unlink()
                meta["repeat_byte_and_matrix_exact"] = True
                meta["upstream_max_abs_error"] = upstream_reference(upstream, decoded, ll_q, shuffle, B)
            jpeg_meta.append(meta)
            print(f"Matrix input QF {q:3d}: {meta['bytes']:7d} bytes", flush=True)
    target.with_suffix(".partial").replace(target)
    json_file(root / IMAGE_DIR / "jpeg_metadata.json", jpeg_meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    generate(args.run, args.upstream, CONFIG["pilot_qf"] if args.pilot else range(1, 101))
