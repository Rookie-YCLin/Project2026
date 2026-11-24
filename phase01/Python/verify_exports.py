"""核對所有 MATLAB 交付圖的格式、圖號、資料覆蓋及實體輸出。"""
import argparse
from pathlib import Path
from xml.etree import ElementTree as ET
import pandas as pd
from PIL import Image
from image_pipeline import json_file


def verify(run):
    folder = run/"06_MATLAB圖表"
    report = []
    for number in range(1,23):
        label = f"Fig{number:02d}"
        fig, svg, pdf, png, csv = [folder/f"{label}.{ext}" for ext in ("fig","svg","pdf","png","csv")]
        assert all(p.stat().st_size > 100 for p in (fig,svg,pdf,png,csv))
        assert pdf.read_bytes().startswith(b"%PDF-")
        root = ET.parse(svg).getroot()
        assert root.tag.endswith("svg")
        assert any(n.tag.endswith(("path","polyline","image")) for n in root.iter())
        with Image.open(png) as bitmap:
            assert bitmap.format=="PNG" and min(bitmap.size)>700
            size = bitmap.size
            bitmap.verify()
        data = pd.read_csv(csv)
        if number <= 17 or number == 22:
            assert set(data.QF) == set(range(1,101))
        if number <= 13:
            assert len(data)==400
        report.append(dict(figure=label, csv_rows=len(data), png_size=size,
                           file_bytes={p.suffix[1:]:p.stat().st_size for p in (fig,svg,pdf,png,csv)}))
    json_file(run/"10_執行紀錄"/"figure_checks.json", report)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    verify(p.parse_args().run)
