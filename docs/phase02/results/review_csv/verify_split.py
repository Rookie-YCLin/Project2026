"""Verify and optionally restore the CRC-independent SHA-256 CSV archive."""
import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

def digest(x):
    return hashlib.sha256(x).hexdigest()

def check(root,restore):
    groups=defaultdict(list)
    with (root/"csv_index.csv").open(encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):
            groups[r["original"]].append(r)
    for name,rows in groups.items():
        ordered=sorted(rows,key=lambda x:int(x["part_order"]))
        assert [int(r["part_order"]) for r in ordered]==list(range(1,len(rows)+1))
        head=None
        whole=bytearray()
        for row in ordered:
            payload=(root/row["part_path"]).read_bytes()
            assert 0<len(payload)==int(row["bytes"])<500_000
            assert digest(payload)==row["sha256"]
            h,sep,body=payload.partition(b"\n")
            assert sep==b"\n"
            if head is None:
                head=h
                whole.extend(head+sep)
            assert h==head
            assert len(body.splitlines())==int(row["rows"])
            whole.extend(body)
        assert sum(int(r["rows"]) for r in rows)==int(ordered[0]["original_rows"])
        assert digest(whole)==ordered[0]["original_sha256"]
        if restore:
            target=restore/name
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open("xb") as f:
                f.write(whole)
        print(name,len(rows),"parts verified")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--restore-dir",type=Path)
    a=p.parse_args()
    check(Path(__file__).resolve().parent,a.restore_dir)
