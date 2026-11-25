# 第三階段重現方式

使用已固定的第二階段提交 `ecce4da07c7ba010a8bfe247c7e56273ddffc4cc` 所指定的原圖與郭飛 upstream Common 模式。原圖核對採 `phase02/Configs/phase02.json` 的 SHA-256。使用同一 Windows Python 3.13 與 MATLAB R2026b，Python 套件依第二階段環境鎖定。以空的第三階段執行目錄啟動可重現各階段，既有完成標記可用來續跑。

## 驗證及影像順序

下列 PowerShell 變數分別指向第三階段執行目錄、第一階段正式原始區塊檔、原始 JPEG、郭飛 upstream 工作樹及第二階段執行目錄。每個命令的返回碼都應為 0。

```powershell
$phase = (Get-Location).Path
$matlab = 'C:\Program Files\MATLAB\R2026b\bin\matlab.exe'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$math = Join-Path $run '01_數學驗證'
$stress = Join-Path $math 'stress_gate'
New-Item -ItemType Directory -Path $math,$stress -Force | Out-Null

python phase03/Python/math_validation.py $math --phase01 $phase01H5
& $matlab -wait -batch "addpath('$phase/phase03/MATLAB'); phase03_math_validation('$math')"
python phase03/Python/check_gate.py $math
python phase03/Python/check_batch.py $math

python phase03/Python/math_validation.py $stress --phase01 $phase01H5 --stress
& $matlab -wait -batch "addpath('$phase/phase03/MATLAB'); phase03_math_validation('$stress')"
python phase03/Python/check_gate.py $stress
python phase03/Python/check_batch.py $stress

python phase03/Python/image_experiment.py $run $source $upstream --workers 6
python phase03/Python/jpeg_experiment.py $run $source $upstream --workers 5
python phase03/Python/matlab_runner.py $run $matlab --phase02 $phase02 --workers 2
python phase03/Python/review.py $run $phase02 --phase01 $phase01 --publish-to $review
```

校準從 13 個對數步長開始，每輪按已有未攻擊結果最接近門檻的位置取左右相鄰步長，將區間等分為八份並補測中間點，共兩輪。重複步長只計一次。若未達正式門檻，先在零 BER 候選選最接近 40 dB 的步長。沒有零 BER 候選時，從全候選取 PSNR 最接近 40 dB 者。步長平手取較小者。每個正式及探索候選的 PNG 都經存檔與重新讀取。

首輪與擴充數學驗證通過、32 組 PNG 的雜湊都與所選候選一致後，JPEG 程式才寫入凍結組態。所有 JPEG 從同組 PNG 獨立產生，採同一 Pillow 設定與 1:1 色度取樣比例。每個品質值保存量化表及雜湊。MATLAB 獨立計算的矩陣擾動、方向與奇異值結果保存於逐十品質值的檢查點中。Python 盲式提取只讀收到的影像、方法、凍結步長及金鑰。評分時另以相同浮水印核對。MATLAB 驗證時會保存跨軟體 SVD 選基及判決差異。

共有 3,200 個 JPEG 條件，每組使用 38,088 個區塊。依原始矩陣、SVD、逐塊轉移量及判決資料的 float64 欄位估算，壓縮前約 125 GB。實際空間需同時計入 JPEG、PNG、檢查點、圖表與雜湊，預留 200 GB。HDF5 採無損壓縮，攻擊與 MATLAB 程式可從已核對的檢查點續跑。

## 矩陣資料

| 資料 | 內容 |
|---|---|
| `04_影像與矩陣/common.h5` | A、原始 SVD、洗牌、預期區塊位元、零基底的區塊和浮水印索引 |
| `04_影像與矩陣/<方法>/embedding.h5` | B 與 B₀、重新分解的 U/σ/V、理論旋轉、目標角、P 證書或 G 平面索引 |
| `04_影像與矩陣/<方法>/attacks.h5` | 每個 C(Q) 的原始矩陣、SVD、區塊與最終判決、實際間距及 JPEG 資訊 |
| `04_影像與矩陣/<方法>/matlab_checks/part*.h5` | MATLAB 的六種矩陣轉移量、方向偏移、判決裕度及光譜分類 |
| `05_統計分析` | 逐方法、逐 QF、逐通道描述統計、BER 和數值交叉核對 |
| `06_MATLAB圖表` | 24 個研究圖，附 FIG、PNG、SVG、PDF 及各圖繪圖資料 |

HDF5 的 U、V 均以欄存放奇異向量。P 的跡、乘數與證書以 σ₁² 正規化。分支索引 0～3 的內積符號及行列式依序為 (+,+)、(+,−)、(−,+)、(−,−)。乘數欄記錄產生最小取樣對偶上界的乘數。批次計時欄是該批總耗時平均分攤到每一區塊。校準 CSV 另留嵌入含 PNG 與盲式提取的耗時。分支端點用固定欄補全，乘數在端點以零作狀態標記。G 的輔助向量使用零基底索引，失敗原因 1 是無合法奇偶中心，原因 2 是沒有同位元的可行平面中心。

光譜分類 1～5 依序是正且單重、近重、重複正值、單重零、多重零。數值零與數值重根採用雙精度 `32 eps × σ₁`，近重診斷採用 `10⁻⁶ × σ₁`。分類保留原始值及各比對條件的間距，判決流程使用收到的區塊角度。

圖可透過 `phase03_figures(runDir,phase02Dir,ID)` 單獨重建。統計中 qf=0 或空值各表都代表 PNG 條件，JPEG 只使用 1～100。CSV 分割索引的資料列從 1 開始，重組時順序串接各檔，第二個分割檔起省去第一個標頭。所列 SHA-256 可檢查位元組重組結果。大型 HDF5 的雜湊及相對目錄附於版本控制資料索引。
