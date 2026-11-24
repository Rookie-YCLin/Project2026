# Project2026 第一階段｜JPEG 區塊矩陣擾動

在單張未嵌入浮水印的血小板影像上，逐一測量 JPEG QF 1–100、4:4:4 與三通道所有完整區塊。研究規格以 Project2026 主分支 8686ebc8cc1533a0c1fc55fac6cbd6842293c335 與作者本次第一階段指示為準。

## 重現

需要 Git、64 位元 Python 3.13、授權 MATLAB R2026b 與可用的 Times New Roman。分析使用基礎 MATLAB。可在此目錄建立獨立 Python 虛擬環境，再於 PowerShell 執行下列指令。請依電腦實際安裝位置指定 MATLAB 可執行檔。Project1124 必須是可寫入的既有目錄。

```powershell
python -m venv .venv
$py = (Resolve-Path .venv\Scripts\python.exe).Path
& $py -m pip install -r Configs\requirements.txt
$project = 'C:\Users\macuz\OneDrive\文件\Project1124'
$matlab = 'C:\Program Files\MATLAB\R2026b\bin\matlab.exe'
& $py Python\run_phase01.py --project $project --matlab $matlab --pilot
& $py Python\run_phase01.py --project $project --matlab $matlab
```

程式依固定 SHA 抓取上游並核對影像與核心程式的 SHA-256。可選用 `--upstream` 提供已在該 commit 的乾淨上游 checkout。每次執行建立新識別碼。先跑 QF 1、50、100 與 identity control，再跑全部 100 個 QF；兩次使用各自的輸出目錄。若單階段失敗，保留日誌與已產生的檔案，重新執行主程式會建立新的目錄。

正式工作順序為 Python 圖像處理與資料交換、MATLAB 的 SVD 和統計、Python 獨立數學驗證、MATLAB 配對分析與正式圖表、Matplotlib 驗證圖。各步驟的獨立命令寫入執行日誌。建議預留至少數 GB。所有正式 JPEG 均從同一原圖的解碼像素編碼。設定 `password_img=1` 是上游預設值。

## 單圖重建

在 MATLAB 加入 `phase01/MATLAB` 路徑後執行。例如，從完整資料重建圖 4 與圖 18 到新的輸出資料夾。

```matlab
generate_figures(runDir,[4 18],fullfile(runDir,'重新繪圖'));
```

`generate_figures.m` 按圖號選擇資料，CSV 內容可核對實際繪圖資料列。正式圖 1–13 和補充圖 14–22 共用此入口。

## HDF5 檔案

| 檔案 | 路徑與維度 |
|---|---|
| block_matrices.h5 | /original/A、/jpeg/QFxxx/A，皆為 channel, block, row, column。/identity/A 為原始矩陣的 HDF5 硬連結。另保存 /index、非對稱 /layout_sentinel，與 /preprocess_check 的代表性像素、YUV、LL。 |
| svd_matlab.h5 | /original、/jpeg/QFxxx 各有 U、sigma、V。U、V 為 channel, block, row, column；sigma 為 channel, block, singular_index。可用 U @ diag(sigma) @ V.T 還原。|
| perturbation_matlab.h5 | /identity、/jpeg/QFxxx 的 metrics、diagnostics 為 channel, block, field，欄名依 dataset 的 fields 屬性排列。diagnostics 為 0/1。|

資料操作採 HDF5 檔案的 0-based C-order 描述。數學／CSV 的 j 用 1–4。channel 依序是 Y、U、V。block 由上而下、每列由左至右。同一位置的排列跨通道及 QF 共用。全域 block_id 為 channel*n+block。MATLAB h5read 回傳的維度順序相反，程式在運算前校正 row 與 column，並以 sentinel 及逐頁 SVD 重建驗證。正式 HDF5 採分 JPEG QF 資料群組、chunking 與無損 Deflate。identity 沿用同一份 A，沒有 QF 值。

## 統計與容許誤差

- 統計量依逐塊值計算，母體標準差使用分母 N，分位數使用 h=1+(N−1)p 線性插值。All 直接合併三個通道。固定間距分箱左閉右開。
- 原始 σ₁=0 的正規化間距記 NaN。補充比例 √D_F/g 只在原始 g>10⁻⁶σ₁ 時計算。NaN 只在相應統計中視為無定義，逐塊資料仍完整保存。相關係數對非有限值成對略過，常數列無相關係數。Spearman 平手用平均秩。
- 重根與零奇異值依 32εσ₁ 的數值容差標記。近重根與近秩虧以 10⁻⁶σ₁ 標記。主角度保留全部排序方向，輔助敏感度表只保留前後最近間距與該奇異值均大於相應 10⁻⁶σ₁ 的方向。
- SVD 重建尺度為 max(1,||A||_F)，正交性使用 ||UᵀU−I||_F 及 V 對應式，容許值 128ε。Mirsky 核對除以 max(1,||A||_F²+||Ã||_F²)，容許值 128ε。Frobenius 逐元素與 norm 的平方獨立計算，容許值 32ε。雙精度 ε=2⁻⁵²。
- 跨平台比較存檔 SVD 重建、排序奇異值、單一方向與相鄰群組的子空間投影。在重根群組比較 UUᵀ 與 VVᵀ。投影差容許值使用外部平方奇異值間距作尺度調整。個別報告主統計在退化區塊的基底相依差異。
- 圖表採 A4 相容寬度、全 QF 原始曲線、PNG 350 dpi 與向量 PDF/SVG。熱圖色塊是 PDF/SVG 內的影像，座標與文字為向量。圖內採英文與數字，使用 Times New Roman。

## 參考來源

1. [郭飛 blind_watermark 固定版本](https://github.com/guofei9987/blind_watermark/tree/322c90dc02d45644692095b34d3a6b735b6edbd8)。
2. L. Mirsky, “Symmetric gauge functions and unitarily invariant norms,” *Q. J. Math.*, vol. 11, pp. 50–59, 1960. [DOI](https://doi.org/10.1093/qmath/11.1.50)。
3. P.-Å. Wedin, “Perturbation bounds in connection with singular value decomposition,” *BIT*, vol. 12, pp. 99–111, 1972. [DOI](https://doi.org/10.1007/BF01932678)。
4. T. Kassis, V. Agarwal, Y. He, D. Patel, and A. M. Brueckner, “Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents,” arXiv, 2026. MATLAB、Matplotlib 操作技能參考。 [arXiv](https://arxiv.org/abs/2609.00065)。
