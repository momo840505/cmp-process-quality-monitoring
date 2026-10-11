# Step 02B：建立可信的分析用資料表

## 為什麼分成兩張表？

- **感測事件表（每個 split 一個 `02b_sensor_events_*.csv.gz`）**：一筆代表一列原始製程感測資料（來源可追溯），供之後 FDC 使用。
- **晶圓 × 階段品質表（`02b_wafer_stage_quality.csv`）**：同一 `WAFER_ID + STAGE` 在同一 split **只有一筆** MRR 目標，供 SPC 與品質分析使用。三組合計 2,829 筆，而不是 973,154 筆品質標籤。

## 清理的四個決策（保守）

1. **保留原始檔**：不修改 `PHM-Data-Challenge`。Step 02A 的報告也保留。所有輸出放 `outputs/`，預設由 `.gitignore` 排除。
2. **衍生感測表僅去除 25 個原始欄位完全相同的額外紀錄**：training 1,819、test 98、validation 434（預期合計 2,351）。每筆去除的位置存入 `02b_exact_duplicate_removal_log.csv`。這是一個*分析假設*，不代表能證明原本紀錄沒有物理意義；以後做敏感度分析仍可以使用原始資料比較。
3. **相同 wafer/stage/機台位置/時間但數值不同**：全部保留，並設 `same_time_multiple_sensor_values=True`；training 401、test 89、validation 64 組。**不能只用時間去重**。
4. **MRR > 1,000 的四筆**：全部保留，設 `mrr_review_gt_1000=True`，這是檢查旗標，**不是規格線或已確定的異常定義**。

## 為什麼要保留 `source_file`、`source_row`？

發現可疑的機台訊號時，可以回原始 CSV 檢查。`source_row` 是從 2 開始的原始 CSV 資料行號（第 1 行為欄名）。

## 時間怎麼處理？

`TIMESTAMP` 以 wafer、stage、機台位置（`MACHINE_DATA` / `CHAMBER`）分開排序，同一時間不同值按照原始檔案與行號穩定排序。這**不證明**物理事件誰先誰後；只有資料表中的明確呈現順序。`recorded_time_span_seconds` 是最大減最小時間戳，不保證設備真的連續運轉這麼久。

## 這一步刻意沒有做什麼？

- **沒有**硬判定製程好壞、沒有設 SPC 管制界限、沒有定義良率異常。
- **沒有**把原始訊號 0 當缺失、沒有替缺值補平均、沒有依 MRR 刪除資料。
- **沒有**在 train、test、validation 之間交換資料，或在全部資料上訓練模型。
- **沒有**把不同製程階段 A/B 或不同機台位置的訊號當成同一個時間序列進行統計建模。

## 後續提醒

`02b_wafer_stage_quality.csv` 有原始的跨切分 wafer ID 重疊，不能把既有三組直接視為完全 wafer-independent。Step 03 做 SPC/FDC 時須分階段與設備條件建立正常基準；進入建模後還要評估 wafer-grouped holdout 與測試集隔離。
