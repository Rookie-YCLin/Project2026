# Project2026 第一階段獨立審查

來源是正式執行 `phase01_20261124_194030_975258`，既有實驗的 Git 版本為 `d17004331acce952480c4f74415a80c74423b440`。所有資料由正式成果複製或提取。審查子集使用 QF 1、10、50、79、80、81、90、100，每個條件包含全部 Y、U、V 的完整區塊。

## 小型 CSV

可直接瀏覽 [統計](csv/statistics_matlab)、[相關](csv/correlations_matlab)與[間距分箱](csv/gap_bins_matlab)三組目錄。先讀 [分割資料索引](index/csv_split_index.csv)。索引有分段順序、起訖 QF、行數、位元組大小及 SHA-256。Identity 各自有一份，QF 欄留白。每份分段保留完整欄名，整個 QF 的資料列保存在同一份。分段保留原始文字表示與行尾。

以 Python 執行 `python verify_csv.py --restore-dir restored` 可驗證每段 SHA-256，並按索引拼回三份完整 CSV。重組結果經位元組逐一比對及 SHA-256 核對。[實際重組結果](index/csv_merge_verification.json)

## 區塊索引與 HDF5

- [正式 HDF5 檔案索引](index/formal_hdf5_files.csv)記錄三份正式 HDF5 的本機完整路徑、大小及 SHA-256。[全部 dataset 路徑](index/formal_hdf5_datasets.csv)
- [條件與通道索引](index/formal_block_groups.csv)涵蓋原始、identity、QF 1–100，列出每個條件與通道可讀取的矩陣、SVD、指標與診斷 dataset。原始 SVD 同樣適用於 identity。三色通道共用相同區塊位置序列與同位置洗牌。[區塊位置、全域 ID 與係數排列](index/block_positions)
- 獨立檔案 `Phase01_Review_Subset.h5` 在原本的 dataset 路徑前加上 `/matrices`、`/svd` 或 `/perturbations` 命名空間。例如 `/matrices/original/A`、`/svd/jpeg/QF080/U`、`/perturbations/jpeg/QF080/metrics`。它含原始矩陣及 SVD、八個 QF 的矩陣與 SVD、逐區塊指標及診斷、原有代表性前處理樣本與 identity control。
- Python 以 HDF5 原順序讀取。矩陣的維度是 channel, block, row, column，奇異值是 channel, block, singular_index，逐塊指標為 channel, block, field。channel 依 Y、U、V 排列，CSV 中名尾為 0 的索引以零起算。MATLAB 讀取後依各 dataset 的 axes 屬性校正維度。完整欄名見 metrics 或 diagnostics 的 fields 屬性。
- 邏輯資料雜湊依 dataset dtype 讀取陣列，將 C 順序的原始位元組取 SHA-256。提取後重新開啟子集，逐一比對雜湊、形狀、dtype、屬性、chunk 與壓縮設定。[提取驗證表](index/review_hdf5_dataset_verification.csv)
- 原圖及八個正式 JPEG 在本機 `JPEG/`，ZIP 內也有相同複本。[JPEG 來源與 SHA-256](index/jpeg_extract_index.csv) [取樣、量化表與大小](index/jpeg_review_metadata.json)

## 使用本機審查包

審查資料夾附有 `review/index/package_delivery.json`，列出獨立 ZIP 與 HDF5 的本機路徑和 SHA-256。ZIP 內 `formal/` 按正式目錄保存全 QF 統計、相關、分箱、診斷、Python 驗證、交叉驗證、研究報告、索引、MATLAB 圖及原始日誌。`code/` 是正式 Python、MATLAB、配置及測試的複本。`review/` 是小型資料及索引。ZIP 的 `ZIP_CONTENTS.csv` 提供每個有內容檔案的大小與 SHA-256。HDF5 子集以獨立檔案提供，三份大型 HDF5 可由正式來源索引定位。
