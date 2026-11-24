"""Verify SHA-256 for every CSV part and restore the exact source bytes."""
import argparse
import csv
import hashlib
import io
from collections import defaultdict
from pathlib import Path


def verified_sources(root):
    parts = defaultdict(list)
    with (root/"index"/"csv_split_index.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parts[row["original_file"]].append(row)
    for name, rows in parts.items():
        rows.sort(key=lambda row: int(row["part_order"]))
        assert [int(row["part_order"]) for row in rows] == list(range(1,len(rows)+1))
        merged = bytearray()
        header = None
        count = 0
        conditions = {}
        for row in rows:
            data = (root/row["part_file"]).read_bytes()
            assert len(data) == int(row["part_bytes"]) and len(data) < 500_000
            assert hashlib.sha256(data).hexdigest() == row["part_sha256"]
            part_header, sep, body = data.partition(b"\n")
            assert sep == b"\n"
            if header is None:
                header = part_header + sep
                merged.extend(header)
            assert header == part_header + sep
            records = list(csv.reader(io.StringIO(body.decode("utf-8"), newline=""), strict=True))
            assert len(records) == int(row["rows"]) and records
            assert records[0][0] == row["first_condition"]
            assert records[-1][0] == row["last_condition"]
            for record in records:
                conditions[record[0]] = conditions.get(record[0],0) + 1
            merged.extend(body)
            count += len(records)
        assert list(conditions) == ["identity", *[f"QF{q:03d}" for q in range(1,101)]]
        assert count == int(rows[0]["original_rows"]) and len(merged) == int(rows[0]["original_bytes"])
        assert hashlib.sha256(merged).hexdigest() == rows[0]["original_sha256"]
        yield name, bytes(merged), count, len(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--restore-dir", type=Path, help="Optional new directory for byte-identical CSVs")
    a = p.parse_args()
    if a.restore_dir:
        a.restore_dir.mkdir()
    for name, data, count, parts in verified_sources(Path(__file__).resolve().parent):
        if a.restore_dir:
            (a.restore_dir/name).write_bytes(data)
        print(f"{name}: {parts} parts, {count} rows, SHA-256 {hashlib.sha256(data).hexdigest()}")
