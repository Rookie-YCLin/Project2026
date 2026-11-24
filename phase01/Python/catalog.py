"""封存第一階段資料與檔案雜湊。正式出版前後皆可生成索引。"""
import argparse
import csv
import json
from pathlib import Path

from image_pipeline import sha

FOLDERS = dict([
    ("01_JPEG影像","逐 QF JPEG、色度取樣與量化表"),
    ("02_原始矩陣","原始矩陣與逐 QF 矩陣、位置、排列及驗證樣本"),
    ("03_SVD分解結果","MATLAB 逐塊 SVD 的 U、sigma、V"),
    ("04_逐區塊擾動數據","MATLAB 逐塊指標與退化診斷"),
    ("05_統計分析","MATLAB 逐 QF 統計、關聯與分箱"),
    ("06_MATLAB圖表","正式與補充圖表及逐圖 CSV"),
    ("07_Python驗證","Python 獨立數學、統計與驗證圖"),
    ("08_交叉驗證","逐塊與逐統計量軟體交叉核對"),
    ("09_實驗報告","第一階段研究報告與圖表索引"),
    ("10_執行紀錄","版本、種子、運算與測試紀錄"),
])
DERIVED = {"file_index.csv","manifest.sha256","github_publication.json","成果索引.md"}


def catalog(run, code, specs, references):
    base = run.resolve().parents[1]
    record = json.loads((run/"10_執行紀錄"/"run.json").read_text(encoding="utf-8"))
    original = Path(record["original_path"])
    paths = {p: FOLDERS.get(p.relative_to(run).parts[0],"執行專屬檔")
             for p in run.rglob("*") if p.is_file() and p.name not in DERIVED}
    paths[original] = "上游原始血小板 JPEG"
    for folder, description in ((code,"第一階段可重現程式與設定"),
                               (specs,"Project2026 基準研究規格快照"),
                               (references,"第一階段的參考文獻與軟體來源")):
        paths.update({p:description for p in folder.rglob("*") if p.is_file()
                      and "__pycache__" not in p.parts and ".git" not in p.parts})
    report = run/"09_實驗報告"
    meta = []
    for path, purpose in sorted(paths.items(),key=lambda x: x[0].relative_to(base).as_posix()):
        meta.append((path.relative_to(base).as_posix(),str(path),purpose,path.stat().st_size,sha(path)))
    with (report/"manifest.sha256").open("x",encoding="utf-8",newline="\n") as f:
        for relative,full,use,length,digest in meta:
            f.write(f"{digest}  {relative}\n")
    return meta


def write_index(run, code, rows):
    base = run.resolve().parents[1]
    out = run/"09_實驗報告"
    derived_purpose = {
        "file_index.csv":"含絕對路徑、用途、大小與 SHA-256 的逐檔目錄",
        "manifest.sha256":"已封存科學及程式檔案的 SHA-256 清單",
        "github_publication.json":"GitHub 分支、commit 與 PR",
        "成果索引.md":"依目錄定位第一階段成果",
    }
    for filename,purpose in derived_purpose.items():
        p = out/filename
        if not p.exists() and filename == "github_publication.json":
            continue
        record_hash = sha(p) if p.exists() and filename not in {"成果索引.md","file_index.csv"} else ""
        length = p.stat().st_size if p.exists() and filename not in {"成果索引.md","file_index.csv"} else ""
        rows.append((p.relative_to(base).as_posix(),str(p),purpose,length,record_hash))
    with (out/"file_index.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_to_project","absolute_path","purpose","bytes","sha256"])
        writer.writerows(sorted(rows))
    lines = ["# 第一階段成果索引", "", f"執行識別碼為 `{run.name}`。", "",
             f"執行資料根目錄為 `{run}`。","",f"程式碼快照為 `{code}`。","",
             "完整逐檔索引見 [file_index.csv](file_index.csv)。檔案雜湊見 [manifest.sha256](manifest.sha256)。",
             "", "| 絕對路徑 | 用途 |","|---|---|"]
    for folder,description in FOLDERS.items():
        lines.append(f"| `{run/folder}` | {description} |")
    g = out/"github_publication.json"
    if g.exists():
        data = json.loads(g.read_text(encoding="utf-8"))
        lines.extend(["",f"GitHub 提交 `{data['commit']}`。[審閱 PR]({data['pr_url']})"])
    lines += ["","圖表由 [圖表索引.md](圖表索引.md) 進入。分析正文見 [第一階段研究報告.md](第一階段研究報告.md)。",""]
    (out/"成果索引.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run","code","specs","references"):
        p.add_argument(name,type=Path)
    p.add_argument("--refresh-index",action="store_true")
    a=p.parse_args()
    if a.refresh_index:
        path=a.run/"09_實驗報告"/"file_index.csv"
        with path.open(encoding="utf-8-sig",newline="") as stream:
            rows=[tuple(row) for row in list(csv.reader(stream))[1:] if Path(row[0]).name not in DERIVED]
    else:
        rows=catalog(a.run,a.code,a.specs,a.references)
    write_index(a.run,a.code,rows)


if __name__ == "__main__":
    main()
