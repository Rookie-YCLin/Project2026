"""建立每一張正式及補充圖的索引。"""
import argparse
from pathlib import Path

TITLES = [
    "平方 Frobenius 擾動的平均值",
    "Y 通道的四個平均絕對奇異值差",
    "U 通道的四個平均絕對奇異值差",
    "V 通道的四個平均絕對奇異值差",
    "三通道合併的四個平均絕對奇異值差",
    "Y 通道的四個平均左奇異向量夾角",
    "U 通道的四個平均左奇異向量夾角",
    "V 通道的四個平均左奇異向量夾角",
    "三通道合併的四個平均左奇異向量夾角",
    "Y 通道的四個平均右奇異向量夾角",
    "U 通道的四個平均右奇異向量夾角",
    "V 通道的四個平均右奇異向量夾角",
    "三通道合併的四個平均右奇異向量夾角",
    "四個奇異值的平均有號差",
    "核範數的平均有號差與平均絕對差",
    "平方 Frobenius 擾動、奇異值平方差與差額",
    "JPEG 後正規化最近間距的中位數與 P10–P90",
    "原始正規化最近間距與左角度的分箱統計",
    "原始正規化最近間距與右角度的分箱統計",
    "QF 10、50、90 的三通道區塊空間擾動",
    "QF 1–20 的平方 Frobenius 擾動",
    "整體平方 Frobenius 擾動中位數與百分位帶",
]


def create(run):
    front = ["# MATLAB 圖表索引", "",
             "圖 1–13 為主要結果，圖 14–22 為補充結果。所有圖均由 MATLAB 產生。角度使用 degree。"
             "各圖的繪圖 CSV 保存未平滑資料。", "",
             "| 圖號 | 內容 | 檔案 |","|---:|---|---|"]
    for i, title in enumerate(TITLES,start=1):
        links = " · ".join(f"[{ext.upper()}](../06_MATLAB圖表/Fig{i:02d}.{ext})"
                           for ext in ("fig","svg","pdf","png","csv"))
        front.append(f"| {i} | {title} | {links} |")
    front += ["","22 張圖共用 [generate_figures.m](generate_figures.m) 入口；程式在正式程式碼快照的 MATLAB 子目錄。",""]
    front[-2] = front[-2].replace("[generate_figures.m](generate_figures.m)","`generate_figures.m`")
    (run/"09_實驗報告"/"圖表索引.md").write_text("\n".join(front),encoding="utf-8")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    create(p.parse_args().run)
