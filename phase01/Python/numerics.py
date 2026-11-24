"""Python 的獨立矩陣分析與描述統計。磁碟上 V 一律儲存列向量。"""
import numpy as np
from scipy.stats import rankdata

EPS = np.finfo(np.float64).eps
REL_NEAR = 1e-6
STAT = ("count", "mean", "median", "std", "min", "max", "p10", "p25", "p75", "p90", "p95")
PROBS = (.10, .25, .75, .90, .95)
SCOPES = ("Y", "U", "V", "All")
CORR_FEATURES = ("gap", "gap_norm", "gap_q", "gap_q_norm", "perturbation_over_gap")
BIN_EDGES = np.array([0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1, np.inf])


def angles(u, w):
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(u*w, axis=-2)), 0, 1)))


def svd(A):
    U, s, vh = np.linalg.svd(A, full_matrices=False)
    return U, s, np.swapaxes(vh, -2, -1)


def fv(A):
    return np.sum(A*A, axis=(-2, -1))


def validate_svd(A, result):
    U, s, V = result
    scale = np.maximum(1, np.sqrt(fv(A)))
    reconstruction = np.max(np.sqrt(fv((U*s[..., None, :]) @ V.swapaxes(-2, -1) - A))/scale)
    orthogonality = max(np.max(np.sqrt(fv(X.swapaxes(-2, -1) @ X - np.eye(4)))) for X in (U, V))
    assert reconstruction <= 128*EPS and orthogonality <= 128*EPS
    assert np.all(s >= 0) and np.all(np.diff(s, axis=-1) <= 0)
    return np.array([reconstruction, orthogonality])


def spectral(s):
    d = -np.diff(s, axis=-1)
    gap = np.stack([d[..., 0], np.minimum(d[..., 0], d[..., 1]),
                    np.minimum(d[..., 1], d[..., 2]), d[..., 2]], axis=-1)
    scale = s[..., :1]
    relative = np.divide(gap, scale, out=np.full(gap.shape, np.nan), where=scale != 0)
    return d, gap, relative


def diagnostics(s, gap):
    scale = s[..., :1]
    tol = 32*EPS*scale
    near = REL_NEAR*scale
    return dict(repeated=gap <= tol, near_repeated=gap <= near,
                zero_singular=s <= tol, near_rank_deficient=s[..., -1] <= near[..., 0],
                zero_matrix=s[..., 0] == 0)


def perturb(A, B, left, right):
    u, s, v = left
    uq, sq, vq = right
    inc = sq-s
    E = B-A
    D = fv(E)
    Ds = np.sum(inc**2, axis=-1)
    original_norm, new_norm = np.sum(s, axis=-1), np.sum(sq, axis=-1)
    nuclear_diff = new_norm - original_norm
    d, gap, rel = spectral(s)
    dq, gapq, relq = spectral(sq)
    spectral_ratio = np.divide(np.sqrt(D)[..., None], gap, out=np.full_like(gap, np.nan), where=gap > REL_NEAR*s[..., :1])
    df_error = np.max(np.abs(D - np.linalg.norm(E, axis=(-2, -1))**2) / np.maximum(1, D))
    scale = np.maximum(1, fv(A) + fv(B))
    violation = np.max(np.maximum(0, Ds-D)/scale)
    assert df_error <= 32*EPS and violation <= 128*EPS
    m = dict(D_F=D, D_sigma=Ds, R=D-Ds, nuclear_original=original_norm, nuclear_q=new_norm,
             nuclear_delta=nuclear_diff, nuclear_abs=np.abs(nuclear_diff))
    vectors = dict(sigma=s, sigma_q=sq, delta_sigma=inc, abs_delta_sigma=np.abs(inc),
                   theta_u=angles(u, uq), theta_v=angles(v, vq), adjacent=d, adjacent_q=dq,
                   gap=gap, gap_q=gapq, gap_norm=rel, gap_q_norm=relq,
                   perturbation_over_gap=spectral_ratio)
    for name, vector in vectors.items():
        for j in range(vector.shape[-1]):
            m[f"{name}_{j+1}"] = vector[..., j]
    flags = {}
    for prefix, values in (("original", diagnostics(s, gap)), ("jpeg", diagnostics(sq, gapq))):
        for name, value in values.items():
            if value.ndim == s.ndim:
                for j in range(4):
                    flags[f"{prefix}_{name}_{j+1}"] = value[..., j]
            else:
                flags[f"{prefix}_{name}"] = value
    assert np.all((m["theta_u_1"] >= 0) & (m["theta_u_1"] <= 90))
    for name in ("theta_u", "theta_v"):
        assert np.all((vectors[name] >= 0) & (vectors[name] <= 90))
    return m, flags, np.array([df_error, violation])


def distribution(a):
    finite = np.sort(a[np.isfinite(a)])
    if finite.size == 0:
        return [0] + [np.nan]*10
    def quantiles(ps):
        h = (finite.size-1)*np.array(ps)
        f = h.astype(int)
        return finite[f] + (h-f)*(finite[np.minimum(f+1, finite.size-1)] - finite[f])
    return [finite.size, finite.mean(), quantiles([.5])[0], finite.std(ddof=0),
            finite[0], finite[-1], *quantiles(PROBS)]


def select(a, k):
    return a[k].ravel() if k < 3 else a.ravel()


def summary(m, name):
    for c, scope in enumerate(SCOPES):
        for key, array in m.items():
            yield [name, scope, key, *distribution(select(array, c))]


def correlations(m, name):
    for c, scope in enumerate(SCOPES):
        for j in range(1, 5):
            for feature in CORR_FEATURES:
                predictor = select(m[f"{feature}_{j}"], c)
                for side in ("u", "v"):
                    theta = select(m[f"theta_{side}_{j}"], c)
                    keep = np.isfinite(predictor) & np.isfinite(theta)
                    x, y = predictor[keep], theta[keep]
                    usable = x.size > 1 and x.max() > x.min() and y.max() > y.min()
                    pearson = np.corrcoef(x, y)[0, 1] if usable else np.nan
                    spearman = np.corrcoef(rankdata(x), rankdata(y))[0, 1] if usable else np.nan
                    yield [name, scope, j, side, feature, x.size, pearson, spearman]


def bins(m, name):
    for c, scope in enumerate(SCOPES):
        for j in range(1, 5):
            x = select(m[f"gap_norm_{j}"], c)
            cats = np.searchsorted(BIN_EDGES, x, side="right")-1
            for side in ("u", "v"):
                y = select(m[f"theta_{side}_{j}"], c)
                for b in range(len(BIN_EDGES)-1):
                    stat = distribution(y[cats == b])
                    yield [name, scope, j, side, b+1, BIN_EDGES[b], BIN_EDGES[b+1],
                           stat[0], stat[2], stat[6], stat[9]]


def subspace_errors(reference, other, tol_relative):
    """逐頁、逐相鄰群組比較雙側投影矩陣。共用兩種分解得到的群組。"""
    u, s, v = reference
    uq, sq, vq = other
    boundary = np.minimum(-np.diff(s, axis=-1), -np.diff(sq, axis=-1))
    scale = np.maximum(s[..., 0], sq[..., 0])
    joined = boundary <= tol_relative*scale[..., None]
    projector_max = 0.
    ratio_max = 0.
    size_max = np.zeros(4)
    group_counts = np.zeros(4, dtype=int)
    direct = np.zeros_like(s, dtype=bool)
    for a in range(4):
        for b in range(a, 4):
            keep = np.all(joined[..., a:b], axis=-1)
            if a > 0:
                keep &= ~joined[..., a-1]
            if b < 3:
                keep &= ~joined[..., b]
            group_counts[b-a] += np.count_nonzero(keep)
            if not np.any(keep):
                continue
            if b == a:
                # 保守地把相對接近零的向量留在子空間比較。
                direct[..., a] = keep & (np.minimum(s[..., a], sq[..., a]) > tol_relative*scale)
            sep = np.full_like(scale, np.inf)
            if a > 0:
                sep = np.minimum(sep, boundary[..., a-1])
            if b < 3:
                sep = np.minimum(sep, boundary[..., b])
            # 對 AA^T/A^TA 使用外部平方奇異值間距，並納入雙方量級。
            squared_sep = np.full_like(scale, np.inf)
            for j in (a-1, b):
                if 0 <= j < 3:
                    squared_sep = np.minimum(
                        squared_sep, np.minimum((s[..., j]-s[..., j+1])*(s[..., j]+s[..., j+1]),
                                                (sq[..., j]-sq[..., j+1])*(sq[..., j]+sq[..., j+1])))
            denom = squared_sep[keep]
            bound = 512*EPS*(1 + scale[keep]**2/denom)
            for p, q in ((u, uq), (v, vq)):
                P, Q = p[keep][:, :, a:b+1], q[keep][:, :, a:b+1]
                distance = np.sqrt(fv(P@P.swapaxes(-2, -1) - Q@Q.swapaxes(-2, -1)))
                projector_max = max(projector_max, float(np.max(distance)))
                ratio_max = max(ratio_max, float(np.max(distance/bound)))
                size_max[b-a] = max(size_max[b-a], float(np.max(distance)))
    # 稍強的交集，零空間與重根的個別角度仍保留於主統計。
    return direct, projector_max, ratio_max, size_max, group_counts
