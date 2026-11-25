# 第三階段 MATLAB 圖表索引

本機執行資料夾的 `06_MATLAB圖表` 內有每張圖的 FIG、PNG、SVG、PDF 和製圖 CSV。圖號可傳給 `phase03_figures(runDir, phase02Dir, ID)` 重新產生。
曲線保留每個 QF 的原始量值。配對矩陣失真使用相同目標角度，正式 BER 比較採符合 PNG 校準門檻的組別。

| 圖號 | 內容 | 製圖 CSV |
|---|---|---|
| Fig01 | 全部方法的最終 512-bit BER | `06_MATLAB圖表/Fig01.csv` |
| Fig02 | 全部方法相對原圖的 PSNR | `06_MATLAB圖表/Fig02.csv` |
| Fig03 | 凍結角度步長與未受攻擊 PNG PSNR | `06_MATLAB圖表/Fig03.csv` |
| Fig04 | 未受攻擊 PNG 區塊 BER | `06_MATLAB圖表/Fig04.csv` |
| Fig05 | 理論嵌入平均平方 Frobenius 失真 | `06_MATLAB圖表/Fig05.csv` |
| Fig06 | A→B、B→B₀、A→B₀ 三種失真 | `06_MATLAB圖表/Fig06.csv` |
| Fig07 | 同一目標角度的 S、G、R、P 配對平方失真 | `06_MATLAB圖表/Fig07.csv` |
| Fig08 | P 相對 G 與 R 的配對失真改善 | `06_MATLAB圖表/Fig08.csv` |
| Fig09 | P 的原始與對偶最佳值差 | `06_MATLAB圖表/Fig09.csv` |
| Fig10 | P 的每區塊批次平均求解時間 | `06_MATLAB圖表/Fig10.csv` |
| Fig11 | 名義旋轉因子的正交性殘差 | `06_MATLAB圖表/Fig11.csv` |
| Fig12 | 理論矩陣 B 的奇異值有號變化 | `06_MATLAB圖表/Fig12.csv` |
| Fig13 | 重新 SVD 後已嵌入區塊的目標方向絕對偏移 | `06_MATLAB圖表/Fig13.csv` |
| Fig14 | G 的中心重選與無法嵌入比例 | `06_MATLAB圖表/Fig14.csv` |
| Fig15 | PNG 判決正確率與五類光譜可辨識性 | `06_MATLAB圖表/Fig15.csv` |
| Fig16 | B₀→C(Q) 的平均平方 Frobenius 擾動 | `06_MATLAB圖表/Fig16.csv` |
| Fig17 | B₀→C(Q) 的左奇異向量角度擾動 | `06_MATLAB圖表/Fig17.csv` |
| Fig18 | B₀→C(Q) 的右奇異向量角度擾動 | `06_MATLAB圖表/Fig18.csv` |
| Fig19 | 方向 QIM 的平均正規化絕對偏移 | `06_MATLAB圖表/Fig19.csv` |
| Fig20 | 方向 QIM 的目標判決裕度與區塊正確率 | `06_MATLAB圖表/Fig20.csv` |
| Fig21 | B₀ 的局部最近間距與 JPEG 角度擾動之 Pearson 相關 | `06_MATLAB圖表/Fig21.csv` |
| Fig22 | 區塊 BER 與最終 512-bit BER | `06_MATLAB圖表/Fig22.csv` |
| Fig23 | QF 1～30 的最終 512-bit BER | `06_MATLAB圖表/Fig23.csv` |
| Fig24 | 第二階段正式方法與第三階段正式校準方法的最終 BER | `06_MATLAB圖表/Fig24.csv` |
