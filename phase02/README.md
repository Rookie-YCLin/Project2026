# Project2026 第二階段重現說明

研究分支自第一階段的 `47ca4df851b08ff515c527f64861ca03535a3215` 衍生。第一階段共用的前處理和 Pillow 編碼器位於 `phase01/Python/image_pipeline.py`，MATLAB 頁面 SVD 和方向夾角函式位於 `phase01/MATLAB`。本階段使用同一張原始 JPEG，須通過 SHA-256 與第一階段原始矩陣逐元素比對。[第二階段規格](../docs/phase02/RESEARCH_SPEC.md) · [第65～72題決策](../docs/phase02/DECISIONS_65_72.md)

## 執行順序

從倉庫根目錄使用 64 位元 Python 3.13、授權 MATLAB R2026b、第一階段相依版本，以及 SHA 為 `322c90dc02d45644692095b34d3a6b735b6edbd8` 的乾淨 `blind_watermark` checkout。以實際本機路徑設定以下參數。

```powershell
$root='C:\Users\macuz\OneDrive\文件\Project1124\06_實驗二_奇異值嵌入'
$original='C:\Users\macuz\OneDrive\文件\Project1124\04_原始影像\17bed8cfeb553435\ori_img.jpeg'
$phase1='C:\Users\macuz\OneDrive\文件\Project1124\05_實驗一_JPEG矩陣擾動\phase01_20261124_194030_975258\02_原始矩陣\block_matrices.h5'
$upstream=(Resolve-Path 'path-to-pinned-blind-watermark').Path
$matlab='C:\Program Files\MATLAB\R2026b\bin\matlab.exe'
$env:PYTHONUTF8='1'
python -m pip install -r phase01\Configs\requirements.txt
python -m unittest discover -s phase02\Tests -v
python phase02\Python\calibrate.py $root $original $upstream $phase1 --marker phase02-run.txt
$run=Get-Content -LiteralPath phase02-run.txt -Encoding UTF8 -Raw
```

在 MATLAB 將 `phase01/MATLAB`、`phase02/MATLAB` 加入路徑，呼叫 `validate_allocations(runDir)`。這個函式讀取正式校準結果，使用獨立 MATLAB SVD、非負可行中心與單純形投影重算。然後依序執行下列命令。

```powershell
python phase02\Python\attack.py $run $original $upstream
python phase02\Python\verify_upstream.py $run $upstream
python phase02\Python\run_matlab.py $run $matlab
python phase02\Python\validate_run.py $run $original $upstream
python phase02\Python\verify_subspaces.py $run
python phase02\Python\report_evidence.py $run
```

在 MATLAB 呼叫 `summarize_reversals(runDir)` 完成分層統計，再呼叫 `generate_phase02_figures(runDir)` 產生正式圖。也可用 `generate_phase02_figures(runDir,[1 9],newFigureDir)` 重建指定圖。以 `python phase02/Python/make_report.py $run` 產生研究報告後，用 `python phase02/Python/review_archive.py $run $original` 建立獨立審查包。用同包內的 `python verify_split.py --restore-dir restored` 驗證並重組大型 CSV。

## 設定與驗收

- payload 為 `default_rng(20261124).integers(0,2,512,dtype=np.uint8)`。升冪索引經 RandomState(1) 洗牌後取位元，索引 i 的區塊載入加密後的第 i mod 512 項。三通道共享各位置的位元與 DCT 係數排列。
- G0 直接呼叫固定上游的 Slow 嵌入及 K-means。提取的合成分數為 `(3*b1+b2)/4`，區塊硬判決為分數大於 0.5。兩個參數均乘以共同倍率 alpha。
- S1～S7 每組唯一步長 q。量化中心只看原始奇異值總和、payload 奇偶與非負可行性，同距取較大中心。S1 的非負投影採活動集合，MATLAB 以二分根獨立驗證。正向與反向比例分配在零總和時使用平均權重。
- 校準以對數粗網格開始，接續三輪細化已實測鄰點。對每個候選寫入並重讀 8-bit PNG，以 39.9～40.1 dB 且未攻擊 0/512 為驗收條件。正式候選按與 40 dB 的距離、步長由小到大決勝。全部候選與 PNG SHA-256 均存檔。候選凍結後才統一進行 JPEG 攻擊。
- 第一階段以單精度做影像前處理。S1～S7 的矩陣分解及重建使用 double， inverse DCT 使用 float32。G0 保持上游的單精度區塊分解及重建。各組均由 OpenCV 將 clip 後的像素寫入 8-bit PNG，重新讀取後才進行評估。
- 試驗報告使用 Pillow 4:4:4 並保存每檔 JPEG 的量化表、採樣因子及 SHA-256。[郭飛固定上游版本](https://github.com/guofei9987/blind_watermark/tree/322c90dc02d45644692095b34d3a6b735b6edbd8)

## HDF5 排列

HDF5 採 C-order 的 channel, block, row, column，三通道順序為 Y、U、V。奇異值張量末軸為 singular_index。U、V 存列向量。MATLAB 讀取時反轉 HDF5 軸且換回 row、column。指標最後一軸的欄名在 `fields` 屬性。

`common.h5` 保存共同 A、Python 參考 SVD、索引及 payload。每組的 `*_embed.h5` 保存 B、B0、嵌入目標、理想修改量與未攻擊盲式結果；`*_jpeg.h5` 保存該組全部 C(Q) 與盲式判決；`*_matlab.h5` 保存從矩陣獨立重算的 SVD 與數學指標。原始 A 的 MATLAB 分解由各方法共用，保存於 G0 檔的 original 群組。
