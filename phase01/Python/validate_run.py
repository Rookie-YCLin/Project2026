"""逐塊交叉核對 MATLAB，並以 NumPy/SciPy 獨立產生摘要。"""
import argparse
import csv
import json
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from image_pipeline import CONFIG, LOG_DIR, MATRIX_DIR, json_file
from numerics import (EPS, STAT, SCOPES, bins, correlations, perturb, select, summary,
                      subspace_errors, svd, validate_svd, fv, distribution)

STAT_KEYS = ("condition", "scope", "metric")
CORR_KEYS = ("condition", "scope", "j", "side", "feature")
BIN_KEYS = ("condition", "scope", "j", "side", "bin")
DIAG_KEYS = ("condition", "scope", "flag")


def from_matlab(group):
    return tuple(group[k][:] for k in ("U", "sigma", "V"))


def fields(d):
    names = d.attrs["fields"]
    if isinstance(names, bytes):
        names = names.decode()
    assert isinstance(names, str)
    values = d[:]
    return dict(zip(names.split(","), np.moveaxis(values, -1, 0), strict=True))


def compare_factor(A, py, mat, near, maxima):
    u, s, v = py
    U, sm, V = mat
    err = np.max(np.abs(s-sm) / np.maximum(1, s[..., :1]))
    maxima["singular_value_scaled"] = max(maxima["singular_value_scaled"], float(err))
    assert err <= 128*EPS
    check = validate_svd(A, mat)
    maxima["stored_matlab_reconstruction"] = max(maxima["stored_matlab_reconstruction"], float(check[0]))
    maxima["stored_matlab_orthogonality"] = max(maxima["stored_matlab_orthogonality"], float(check[1]))
    stable, err, ratio, by_size, counts = subspace_errors(py, mat, near)
    maxima["projection_frobenius"] = max(maxima["projection_frobenius"], err)
    maxima["projection_tolerance_ratio"] = max(maxima["projection_tolerance_ratio"], ratio)
    assert ratio <= 1, f"子空間投影誤差超出尺度容許值: {ratio}"
    for k in range(4):
        maxima[f"projection_size_{k+1}_max"] = max(maxima[f"projection_size_{k+1}_max"], float(by_size[k]))
        maxima[f"projection_size_{k+1}_count"] += int(counts[k])
    return stable


def compare_metrics(m, ref, stable, geometry_scale, maxima, name, direction_errors):
    assert m.keys() == ref.keys()
    assert all(x.shape == geometry_scale.shape for x in m.values())
    for key, x in m.items():
        y = ref[key]
        assert np.array_equal(np.isfinite(x), np.isfinite(y)), (name, key)
        keep = np.isfinite(x)
        err = np.abs(x-y)
        if key.startswith("theta"):
            j = int(key[-1])-1
            mask = stable[..., j]
            reliable = float(err[mask].max(initial=0))
            ambiguous = float(err[~mask].max(initial=0))
            # 額外以投影差建立逐角度界，沿用 acos 在 0 附近的雙精度誤差。
            if reliable > 4e-6:
                limit = direction_errors[key[:7]][..., j] + 4e-6
                assert np.all(err[mask] <= limit[mask]), (name, key, reliable, np.max(limit))
            maxima["angle_identifiable_deg"] = max(maxima["angle_identifiable_deg"], reliable)
            maxima["angle_ambiguous_deg"] = max(maxima["angle_ambiguous_deg"], ambiguous)
            maxima["angle_ambiguous_count"] += int(np.count_nonzero(~mask))
            continue
        if key.startswith("perturbation_over_gap"):
            j = key[-1]
            g, gm = m["gap_"+j], ref["gap_"+j]
            with np.errstate(invalid="ignore", divide="ignore"):
                scale = np.maximum(1, np.maximum(np.abs(x), np.abs(y)))
                # 商的誤差主要來自分母。用實測分母差分離放大效應。
                unc = np.where(keep, np.abs(x)*np.abs(g-gm)/gm + 128*EPS*scale, 0)
            assert np.all(err[keep] <= unc[keep]), (name, key)
            maxima["ratio_scaled_difference"] = max(maxima["ratio_scaled_difference"],
                                                     float(np.max((err/scale)[keep], initial=0)))
            continue
        if key in ("D_F", "D_sigma", "R"):
            scale = geometry_scale**2
        elif key.startswith(("gap_norm", "gap_q_norm")):
            scale = 1.0
        else:
            scale = geometry_scale
        normalized = float(np.max(np.broadcast_to(err/scale, x.shape)[keep], initial=0))
        maxima["metric_scaled"] = max(maxima["metric_scaled"], normalized)
        assert normalized <= 256*EPS, (name, key, normalized)


def direction_bounds(left_py, left_mat, right_py, right_mat):
    result = {}
    for side, index in (("theta_u", 0), ("theta_v", 2)):
        def per_input(p, q):
            a, b = p[index], q[index]
            d = np.minimum(np.linalg.norm(a-b, axis=-2), np.linalg.norm(a+b, axis=-2))
            return np.degrees(2*np.arcsin(np.minimum(1, d/2)))
        result[side] = per_input(left_py, left_mat) + per_input(right_py, right_mat)
    return result


def read_summary(path, keys):
    return pd.read_csv(path, float_precision="round_trip").set_index(list(keys)).sort_index()


def compare_tables(python_dir, mat_dir, validation_dir):
    specs = (("statistics", STAT_KEYS), ("correlations", CORR_KEYS),
             ("gap_bins", BIN_KEYS), ("diagnostics", DIAG_KEYS),
             ("identifiable_angles", ("condition","scope","j","side")))
    comparison = {}
    for name, keys in specs:
        a = read_summary(python_dir / f"{name}_python.csv", keys)
        b = read_summary(mat_dir / f"{name}_matlab.csv", keys)
        assert a.index.equals(b.index)
        checks = []
        for column in a.columns.intersection(b.columns):
            x, y = a[column].to_numpy(dtype=float), b[column].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            diff = np.abs(x[finite] - y[finite])
            scale = np.maximum(1, np.maximum(np.abs(x[finite]), np.abs(y[finite])))
            checks.append([column, len(x), int(finite.sum()), int(np.count_nonzero(np.isfinite(x)^np.isfinite(y))),
                           float(diff.max(initial=0)), float((diff/scale).max(initial=0))])
        table = pd.DataFrame(checks, columns=["field","rows","finite_pairs","finite_mismatch","max_abs","max_scaled"])
        table.to_csv(validation_dir / f"{name}_differences.csv", index=False)
        comparison[name] = {r[0]: dict(rows=r[1], finite_pairs=r[2], finite_mismatch=r[3],
                                      max_abs=r[4], max_scaled=r[5]) for r in checks}
    # 同一 MATLAB 逐塊值再以 NumPy 統計，獨立核對平均、母體 SD 及分位數實作。
    a = read_summary(python_dir / "statistics_from_matlab_blocks.csv", STAT_KEYS)
    b = read_summary(mat_dir / "statistics_matlab.csv", STAT_KEYS)
    assert a.index.equals(b.index) and a.columns.equals(b.columns)
    x, y = a.to_numpy(), b.to_numpy()
    assert np.array_equal(np.isfinite(x), np.isfinite(y))
    keep = np.isfinite(x)
    scale = np.maximum(1, np.maximum(np.abs(x), np.abs(y)))
    scaled = np.abs(x-y)[keep]/scale[keep]
    assert scaled.max(initial=0) <= 2e-12, scaled.max()
    comparison["reaggregation_same_input_max_scaled"] = float(scaled.max(initial=0))
    comparison["reaggregation_same_input_rows"] = int(len(a))
    return comparison


def run(root, pilot):
    source_dir = root / MATRIX_DIR
    output = root / "07_Python驗證"
    validation = root / "08_交叉驗證"
    mat_stats = root / "05_統計分析"
    checked = defaultdict(float)
    numerical_rows = []
    conditions = ["identity", *[f"QF{q:03d}" for q in (CONFIG["pilot_qf"] if pilot else range(1, 101))]]
    meta = json.loads((root / LOG_DIR / "environment_python.json").read_text(encoding="utf-8"))
    jpeg_meta = json.loads((root/"01_JPEG影像"/"jpeg_metadata.json").read_text(encoding="utf-8"))
    expected = sorted(int(c[2:]) for c in conditions[1:])
    assert meta["qualities"] == expected and [r["qf"] for r in jpeg_meta] == expected
    assert {p.stem for p in (root/"01_JPEG影像").glob("*.jpg")} == set(conditions[1:])
    near = CONFIG["relative_near_threshold"]
    with ExitStack() as stack:
        source = stack.enter_context(h5py.File(source_dir / "block_matrices.h5", "r"))
        decomp = stack.enter_context(h5py.File(root / "03_SVD分解結果" / "svd_matlab.h5", "r"))
        mat_metrics = stack.enter_context(h5py.File(root / "04_逐區塊擾動數據" / "perturbation_matlab.h5", "r"))
        assert len(conditions)-1 == len(source["jpeg"]) == len(decomp["jpeg"]) == len(mat_metrics["jpeg"])
        n = meta["blocks_per_channel"]
        idx = source["index"]
        assert np.array_equal(idx["block_id"][:], np.arange(3*n).reshape(3,n))
        assert np.all(np.sort(idx["shuffle"][:], axis=1) == np.arange(16))
        assert np.array_equal(idx["block_row"][:]*meta["block_grid"][1]+idx["block_col"][:], idx["block"][:])
        A = source["original/A"][:]
        assert A.shape == (3, n, 4, 4)
        base = svd(A)
        base_check = validate_svd(A, base)
        base_mat = from_matlab(decomp["original"])
        original_stable = compare_factor(A, base, base_mat, near, checked)
        paths = [
            ("statistics_python", (*STAT_KEYS, *STAT)),
            ("statistics_from_matlab_blocks", (*STAT_KEYS, *STAT)),
            ("correlations_python", (*CORR_KEYS, "count", "pearson", "spearman")),
            ("gap_bins_python", (*BIN_KEYS, "lower", "upper", "count", "median", "p10", "p90")),
            ("diagnostics_python", (*DIAG_KEYS, "count", "true_count")),
            ("identifiable_angles_python", ("condition","scope","j","side",*STAT)),
        ]
        writers = []
        for key, header in paths:
            handle = stack.enter_context((output / f"{key}.csv").open("x", newline="", encoding="utf-8"))
            writer = csv.writer(handle)
            writer.writerow(header)
            writers.append(writer)
        for condition in conditions:
            group = condition if condition == "identity" else "jpeg/" + condition
            B = source[f"{group}/A"][:]
            assert B.shape == A.shape and np.isfinite(B).all()
            result = svd(B)
            score = validate_svd(B, result)
            m, diagnostic, checks = perturb(A, B, base, result)
            ref = fields(mat_metrics[f"{group}/metrics"])
            ref_diag = fields(mat_metrics[f"{group}/diagnostics"])
            assert diagnostic.keys() == ref_diag.keys()
            for k in diagnostic:
                if not np.array_equal(diagnostic[k], ref_diag[k]):
                    checked["diagnostic_boolean_mismatches"] += int(np.count_nonzero(diagnostic[k] != ref_diag[k]))
            rmat, stable = base_mat, original_stable
            if condition == "identity":
                assert np.array_equal(A,B) and not np.any(m["D_F"]) and not np.any(m["D_sigma"])
                assert max(m[f"theta_{side}_{j}"].max() for side in ("u","v") for j in range(1,5)) <= np.degrees(np.arccos(1-base_check[1]-8*EPS))
            else:
                rmat = from_matlab(decomp[group])
                stable = compare_factor(B, result, rmat, near, checked)
            scale = np.maximum(1, np.sqrt(fv(A)) + np.sqrt(fv(B)))
            directions = direction_bounds(base, base_mat, result, rmat)
            compare_metrics(m, ref, stable & original_stable, scale, checked, condition, directions)
            numerical_rows.append([condition, *score, *checks])
            writers[0].writerows(summary(m, condition))
            writers[1].writerows(summary(ref, condition))
            writers[2].writerows(correlations(m, condition))
            writers[3].writerows(bins(m, condition))
            writers[4].writerows([condition, scope, key, select(flag,c).size, np.count_nonzero(select(flag,c))]
                                 for c, scope in enumerate(SCOPES) for key, flag in diagnostic.items())
            writers[5].writerows([condition,scope,j,side,*distribution(
                select(m[f"theta_{side}_{j}"],c)[select(
                    (m[f"gap_{j}"] > near*m["sigma_1"]) & (m[f"gap_q_{j}"] > near*m["sigma_q_1"]) &
                    (m[f"sigma_{j}"] > near*m["sigma_1"]) & (m[f"sigma_q_{j}"] > near*m["sigma_q_1"]),c)])]
                for c,scope in enumerate(SCOPES) for j in range(1,5) for side in ("u","v"))
            print(f"Python independently verified {condition}: {B.shape[0]*B.shape[1]} blocks", flush=True)
    comparison = compare_tables(output, mat_stats, validation)
    result = dict(conditions=conditions, blocks_per_channel=n, total_block_pairs=n*3*len(expected),
                  original_python_svd_check=list(base_check), max_errors=dict(checked), tables=comparison)
    json_file(validation/"cross_validation.json", result)
    pd.DataFrame(numerical_rows, columns=["condition","svd_residual","orthogonality",
                                         "frobenius_delta","mirsky_violation"]).to_csv(
                   output/"numerical_checks_python.csv", index=False)
    print(json.dumps(dict(checked), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--pilot", action="store_true")
    a = parser.parse_args()
    run(a.run, a.pilot)
