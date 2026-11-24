# 文獻公式與實驗邊界

Huang、Chen、Lin、Chen 與 Hsu（2015）式 (15) 令量化索引為 floor(S/q+1/2)。式 (16)–(18) 使用相符奇偶的最近中心。當 S/q 恰為整數且奇偶不符，式 (17) 使中心往上移一格。第 3.2 節從收到影像的奇異值總和重算索引並取 mod 2。本文沿用此奇偶格點及盲式判決。[原始論文](https://doi.org/10.1186/s13640-015-0073-6)

本課堂專題沿用郭飛 Common/Slow 的 YUV–Haar LL–DCT–洗牌管線。七組課堂方法都依本專題決策另施奇異值非負條件。S1 以非負單純形投影找到固定中心下的最小平方改變。其餘組別於方法自身的可行半直線上選最近中心。此部份取捨依 `RESEARCH_SPEC.md` 與 `DECISIONS_65_72.md`。[郭飛原始碼](https://github.com/guofei9987/blind_watermark/blob/322c90dc02d45644692095b34d3a6b735b6edbd8/blind_watermark/bwm_core.py)

G0 用原始兩奇異值的四分之一及四分之三中心，原版雙特徵加權投票和 K-means 維持同一流程。G0 與總和 QIM 的步長各有不同意義。
