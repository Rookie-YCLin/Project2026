"""從兩套正式 HDF5 定位初次檢核尚未分類的區塊位元差異。"""
import h5py
import numpy as np
import pandas as pd

from image_experiment import DATA, LABELS, STATS


def inspect(run, frozen):
    records=[]
    eps=np.finfo(float).eps
    for method in LABELS:
        checks=pd.read_csv(run/STATS/f"{method}_checks_matlab.csv")
        flagged=checks[checks.unexplained_bit_disagreement>0]
        if flagged.empty:
            continue
        base=run/DATA/method
        q=frozen["methods"][method]["q_rad"]
        upper=int(np.floor((np.pi/2)/q))
        if upper*q>np.pi/2:
            upper-=1
        for row in flagged.itertuples(index=False):
            quality=row.qf
            part=(quality+9)//10
            py_file="attacks.h5" if quality else "embedding.h5"
            py_group=f"jpeg/QF{quality:03d}" if quality else "B0"
            mat_group=py_group if quality else "unattacked"
            py_angle="phi" if quality else "feature"
            py_center="nearest_index" if quality else "index"
            with h5py.File(base/py_file) as py, h5py.File(base/"matlab_checks"/f"part{part:02d}.h5") as mat:
                group=py[py_group]
                ds=mat[mat_group+"/decision"]
                fields=ds.attrs["fields"]
                if isinstance(fields,bytes):
                    fields=fields.decode()
                values=ds[0]
                fields=dict(zip(fields.split(","),np.moveaxis(values,-1,0),strict=True))
                angle=fields["phi"]
                tag=fields["rank_class"]
                lower=np.clip(np.floor(angle/q),0,upper)
                distance=np.abs(angle-(lower+.5)*q)
                mask=(fields["decision"]!=group["block_bits"][:]) & ~np.isin(tag,[3,5]) & (distance>128*eps)
                assert mask.sum()==row.unexplained_bit_disagreement
                for channel,block in np.argwhere(mask):
                    pos=(int(channel),int(block))
                    phi_python=float(group[py_angle][pos])
                    phi_matlab=float(angle[pos])
                    py_index=int(group[py_center][pos])
                    mat_index=int(fields["decoded_index"][pos])
                    midpoint=(min(py_index,mat_index)+.5)*q
                    crossed=abs(py_index-mat_index)==1 and min(phi_python,phi_matlab)<=midpoint<=max(phi_python,phi_matlab)
                    sigma1=float(group["sigma"][pos+(0,)])
                    nearest_gap=float(fields["nearest_gap"][pos])
                    records.append(dict(
                        method=method,qf=quality,channel="YUV"[channel],block_index0=block,
                        rank_class=int(tag[pos]),nearest_gap=nearest_gap,sigma1=sigma1,
                        relative_gap=nearest_gap/sigma1 if sigma1 else np.nan,
                        angle_python_rad=phi_python,angle_matlab_rad=phi_matlab,
                        angle_difference_rad=abs(phi_python-phi_matlab),quantizer_midpoint_rad=midpoint,
                        matlab_midpoint_distance_rad=abs(phi_matlab-midpoint),
                        python_index=py_index,matlab_index=mat_index,midpoint_crossed=crossed))
    return pd.DataFrame.from_records(records)
