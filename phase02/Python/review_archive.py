"""Build the independent review archive and byte-exact CSV viewing copies."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import h5py

from allocation import METHODS
from calibrate import FOLDERS
from watermark import CODE,CONFIG,phase01


def split_csv(path, root, key):
    base=path.read_bytes()
    head,sep,body=base.partition(b"\n")
    assert sep==b"\n"
    header=head+sep
    rows=body.splitlines(keepends=True)
    assert b"".join(rows)==body
    folder=root/key/path.stem
    folder.mkdir(parents=True)
    chunk=bytearray(header)
    size=0
    files=[]
    def flush():
        folder_part=folder/f"{len(files)+1:03d}.csv"
        folder_part.write_bytes(chunk)
        files.append((folder_part,size))
    for row in rows:
        assert len(row)+len(header)<450_000
        if len(chunk)+len(row)>=450_000:
            flush()
            size=0
            chunk=bytearray(header)
        chunk.extend(row)
        size+=1
    flush()
    rebuilt=header+b"".join(p.read_bytes()[len(header):] for p,_ in files)
    assert rebuilt==base
    return [[key+"/"+path.name,phase01.sha(path),len(rows),order,p.relative_to(root).as_posix(),
             count,p.stat().st_size,phase01.sha(p)] for order,(p,count) in enumerate(files,1)]


def review_csv(root,study):
    entries=[]
    for f in sorted(study.rglob("*.csv")):
        if f.name.endswith(".reassembled.csv") or f.stat().st_size < 500_000:
            continue
        entries.extend(split_csv(f,root,f.relative_to(study).parent.as_posix()))
    for entry in entries:
        assert 0<entry[6]<500_000
    path=root/"csv_index.csv"
    with path.open("x",encoding="utf-8",newline="") as out:
        writer=csv.writer(out,lineterminator="\n")
        writer.writerow(("original","original_sha256","original_rows","part_order","part_path","rows","bytes","sha256"))
        writer.writerows(entries)
    return len(entries)


def subset_one(data, review_dir, method, labels):
    out=review_dir/f"{method}_review.h5"
    with h5py.File(out,"x") as review:
        for section,suffix in (("embedding","embed"),("matrices","jpeg"),("analysis","matlab")):
            src=data/f"{method}_{suffix}.h5"
            with h5py.File(src) as file:
                group=review.create_group(section)
                for k,v in file.attrs.items():
                    group.attrs[k]=v
                selected=list(file.keys()) if suffix=="embed" else [k for k in file.keys() if k!="jpeg"]
                for k in selected:
                    file.copy(k,group)
                if suffix!="embed":
                    j=group.create_group("jpeg")
                    for label in labels:
                        file.copy("jpeg/"+label,j)
    with h5py.File(out) as f:
        for section,suffix in (("embedding","embed"),("matrices","jpeg"),("analysis","matlab")):
            with h5py.File(data/f"{method}_{suffix}.h5") as src:
                def scan(name,obj):
                    if isinstance(obj,h5py.Dataset):
                        s=src[name]
                        assert s.dtype==obj.dtype and s.shape==obj.shape
                        assert hashlib.sha256(s[:].tobytes()).digest()==hashlib.sha256(obj[:].tobytes()).digest()
                f[section].visititems(scan)
    print(method,"review HDF5 copied and checked",flush=True)


def validate_figures(run):
    from PIL import Image
    from xml.etree import ElementTree
    folder=run/"08_MATLAB圖表"
    for j in range(1,19):
        stem=f"Fig{j:02d}"
        assert all((folder/f"{stem}.{e}").stat().st_size>100 for e in ("fig","svg","pdf","png","csv"))
        assert ElementTree.parse(folder/f"{stem}.svg").getroot().tag.endswith("svg")
        assert (folder/f"{stem}.pdf").read_bytes().startswith(b"%PDF")
        with Image.open(folder/f"{stem}.png") as im:
            assert min(im.size)>750
            im.verify()


def package(run,source):
    validate_figures(run)
    reporting=run/"11_研究報告"
    review=run/"13_獨立審查"
    review.mkdir()
    separate=review/"逐區塊審查HDF5"
    separate.mkdir()
    samples=review/"影像"
    samples.mkdir()
    shutil.copy2(source,samples/"original.jpeg")
    labels=[f"QF{q:03d}" for q in CONFIG["review_qf"]]
    shutil.copy2(run/"05_原始矩陣"/"common.h5",separate/"common.h5")
    for m in METHODS:
        subset_one(run/"06_嵌入與擾動數據",separate,m,labels)
        (samples/m).mkdir()
        shutil.copy2(run/"03_含浮水印影像"/f"{m}.png",samples/m/"unattacked.png")
        for q in labels:
            f=run/"04_JPEG攻擊影像"/m/f"{q}.jpg"
            shutil.copy2(f,samples/m/f.name)
            assert phase01.sha(f)==phase01.sha(samples/m/f.name)
    git_csv=review/"GitHub_CSV"
    git_csv.mkdir()
    part_count=review_csv(git_csv,run)
    shutil.copy2(CODE/"Python"/"verify_split.py",git_csv/"verify_split.py")
    readme=review/"README.md"
    readme.write_text(
        "# 第二階段獨立審查\n\n"
        "ZIP 收錄正式統計、配置、圖表、報告、驗證和代表性影像。"
        "獨立 HDF5 目錄依方法存原始嵌入資料及代表性 QF 的全部區塊，原始 A 在 common.h5。"
        "每個分法以 embedding、matrices、analysis 命名空間區分原始檔案。"
        "HDF5 的 bytes、SHA-256 與絕對路徑見正式 file_index.csv。"
        "大型 CSV 以 GitHub_CSV/csv_index.csv 逐段定位，verify_split.py 可驗證並精確重組。\n",
        encoding="utf-8")
    files=[]
    for p in sorted(run.rglob("*")):
        if p.is_file():
            files.append((p.relative_to(run).as_posix(),str(p),p.stat().st_size,phase01.sha(p)))
    index=reporting/"file_index.csv"
    with index.open("x",encoding="utf-8-sig",newline="") as out:
        w=csv.writer(out,lineterminator="\n")
        w.writerow(["path_from_run","absolute_path","bytes","sha256"])
        w.writerows(files)
    sha=reporting/"manifest.sha256"
    sha.write_text("".join(f"{d}  {p}\n" for p,_,_,d in files),encoding="utf-8")
    target=review/"Phase02_Review_Package.zip"
    include=(FOLDERS[0],FOLDERS[1],FOLDERS[6],FOLDERS[7],FOLDERS[8],FOLDERS[9],FOLDERS[10],FOLDERS[11])
    with ZipFile(target,"x",compression=ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(run.rglob("*")):
            if not p.is_file():
                continue
            relative=p.relative_to(run)
            if relative.parts[0] in include or relative.parts[:2] in ((review.name,samples.name),(review.name,git_csv.name)) or p==readme:
                z.write(p,relative.as_posix())
    with ZipFile(target) as z:
        assert z.testzip() is None
        length=len(z.namelist())
    stats=dict(run=str(run),review_zip=str(target),review_zip_bytes=target.stat().st_size,
               review_zip_sha256=phase01.sha(target),review_zip_members=length,
               files_in_manifest=len(files),bytes_before_zip=sum(row[2] for row in files),
               csv_parts=part_count,review_qf=CONFIG["review_qf"])
    (review/"delivery.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(stats,ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("source",type=Path)
    a=p.parse_args()
    package(a.run,a.source)
